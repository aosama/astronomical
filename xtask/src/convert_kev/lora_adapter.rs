//! A PEFT LoRA adapter: which base tensors it changes, and by how much.
//!
//! Each adapted module carries two matrices, `A` (`[rank, inputs]`) and `B`
//! (`[outputs, rank]`), and the module's weight becomes
//! `W + scale · B · A`, with `scale = alpha / rank`. Only that plain form is
//! read: the variants that change the formula (rank-stabilized scaling,
//! DoRA, transposed weights, trained biases) are refused, because merging
//! them with this formula would silently produce a different model.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::model_files::safetensors_file::{SourceCheckpoint, TensorElementType, TensorRecord};

/// What PEFT prefixes every adapter tensor name with.
const ADAPTER_NAME_PREFIX: &str = "base_model.model.";
/// Where a multimodal Qwen3.5 checkpoint keeps its text model; an adapter
/// trained on the text model alone names its modules without it.
const BASE_TEXT_MODEL_PREFIXES: [&str; 3] = ["model.language_model.", "model.", ""];

/// One adapted module: the base tensor it changes and its two factors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdaptedModule {
    pub(crate) base_tensor_name: String,
    pub(crate) down_projection: TensorRecord,
    pub(crate) up_projection: TensorRecord,
}

/// The adapter as the merge needs it.
pub(crate) struct LoraAdapter {
    pub(crate) adapter_checkpoint: SourceCheckpoint,
    /// By base tensor name, so the shard rewrite can look each tensor up.
    pub(crate) adapted_modules: BTreeMap<String, AdaptedModule>,
    pub(crate) scale: f32,
    pub(crate) rank: u64,
    pub(crate) alpha: f64,
}

impl LoraAdapter {
    /// Reads the adapter in `adapter_directory` and matches each adapted
    /// module to one of `base_tensor_names`.
    pub(crate) fn read(
        adapter_directory: &Path,
        base_tensor_names: &[String],
    ) -> Result<Self, String> {
        let adapter_config: Value = serde_json::from_slice(
            &std::fs::read(adapter_directory.join("adapter_config.json")).map_err(
                |read_error| format!("could not read adapter_config.json: {read_error}"),
            )?,
        )
        .map_err(|json_error| format!("adapter_config.json is not JSON: {json_error}"))?;
        let (rank, alpha) = plain_lora_rank_and_alpha(&adapter_config)?;
        let adapter_checkpoint =
            SourceCheckpoint::open(&adapter_directory.join("adapter_model.safetensors"))?;
        let adapted_modules = match_adapted_modules(&adapter_checkpoint, base_tensor_names, rank)?;
        Ok(Self {
            adapter_checkpoint,
            adapted_modules,
            scale: (alpha / rank as f64) as f32,
            rank,
            alpha,
        })
    }
}

/// The rank and alpha of a plain LoRA adapter, refusing every variant whose
/// merge is not `W + alpha / rank · B · A`.
pub(crate) fn plain_lora_rank_and_alpha(adapter_config: &Value) -> Result<(u64, f64), String> {
    if adapter_config.get("peft_type").and_then(Value::as_str) != Some("LORA") {
        return Err("the adapter is not a LoRA adapter".to_owned());
    }
    for (variant_switch, variant_name) in [
        ("use_rslora", "rank-stabilized scaling"),
        ("use_dora", "DoRA"),
        ("fan_in_fan_out", "transposed weights"),
    ] {
        if adapter_config.get(variant_switch).and_then(Value::as_bool) == Some(true) {
            return Err(format!(
                "the adapter uses {variant_name}, which this converter does not merge"
            ));
        }
    }
    if adapter_config
        .get("bias")
        .and_then(Value::as_str)
        .is_some_and(|bias| bias != "none")
    {
        return Err("the adapter trains biases, which this converter does not merge".to_owned());
    }
    for per_module_override in ["rank_pattern", "alpha_pattern"] {
        if adapter_config
            .get(per_module_override)
            .and_then(Value::as_object)
            .is_some_and(|overrides| !overrides.is_empty())
        {
            return Err(format!(
                "the adapter sets a per-module {per_module_override}, which this converter does not merge"
            ));
        }
    }
    let rank = adapter_config
        .get("r")
        .and_then(Value::as_u64)
        .filter(|rank| *rank > 0);
    let alpha = adapter_config
        .get("lora_alpha")
        .and_then(Value::as_f64)
        .filter(|alpha| *alpha > 0.0);
    match (rank, alpha) {
        (Some(rank), Some(alpha)) => Ok((rank, alpha)),
        _ => Err("adapter_config.json must give a positive r and lora_alpha".to_owned()),
    }
}

/// Pairs every `lora_A` with its `lora_B` and with the base tensor it changes.
fn match_adapted_modules(
    adapter_checkpoint: &SourceCheckpoint,
    base_tensor_names: &[String],
    rank: u64,
) -> Result<BTreeMap<String, AdaptedModule>, String> {
    let mut adapted_modules = BTreeMap::new();
    for down_projection in &adapter_checkpoint.tensor_records {
        let Some(module_path) = down_projection
            .tensor_name
            .strip_prefix(ADAPTER_NAME_PREFIX)
            .and_then(|adapter_name| adapter_name.strip_suffix(".lora_A.weight"))
        else {
            continue;
        };
        let up_projection_name = format!("{ADAPTER_NAME_PREFIX}{module_path}.lora_B.weight");
        let Some(up_projection) = adapter_checkpoint
            .tensor_records
            .iter()
            .find(|tensor_record| tensor_record.tensor_name == up_projection_name)
        else {
            return Err(format!("the adapter has no lora_B for {module_path}"));
        };
        check_factor_shapes(module_path, down_projection, up_projection, rank)?;
        let base_tensor_name = base_tensor_name_for(module_path, base_tensor_names)?;
        adapted_modules.insert(
            base_tensor_name.clone(),
            AdaptedModule {
                base_tensor_name,
                down_projection: down_projection.clone(),
                up_projection: up_projection.clone(),
            },
        );
    }
    let factor_count = adapter_checkpoint.tensor_records.len();
    if adapted_modules.is_empty() || factor_count != adapted_modules.len() * 2 {
        return Err("the adapter's tensors are not all lora_A and lora_B pairs".to_owned());
    }
    Ok(adapted_modules)
}

fn check_factor_shapes(
    module_path: &str,
    down_projection: &TensorRecord,
    up_projection: &TensorRecord,
    rank: u64,
) -> Result<(), String> {
    if down_projection.shape.len() != 2
        || up_projection.shape.len() != 2
        || down_projection.shape[0] != rank
        || up_projection.shape[1] != rank
        || !is_floating_point(down_projection)
        || !is_floating_point(up_projection)
    {
        return Err(format!(
            "the adapter's factors for {module_path} do not have rank {rank}"
        ));
    }
    Ok(())
}

fn is_floating_point(tensor_record: &TensorRecord) -> bool {
    matches!(
        tensor_record.element_type,
        TensorElementType::Float32 | TensorElementType::BFloat16 | TensorElementType::Float16
    )
}

fn base_tensor_name_for(module_path: &str, base_tensor_names: &[String]) -> Result<String, String> {
    for text_model_prefix in BASE_TEXT_MODEL_PREFIXES {
        let candidate_name = format!("{text_model_prefix}{module_path}.weight");
        if base_tensor_names.contains(&candidate_name) {
            return Ok(candidate_name);
        }
    }
    Err(format!(
        "the base model has no tensor for adapted module {module_path}"
    ))
}

#[cfg(test)]
#[path = "lora_adapter_tests.rs"]
mod tests;
