//! Family-owned shallow classification and executable discovery for K2 Horizon MoVA.
//!
//! Recognition is `model_type` only. Completeness requires a stacked affine
//! index, shards, tokenizer, and standalone chat template. Unstacked per-expert
//! tensors stay unpublished.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::Deserialize;

/// Family-derived metadata returned to neutral discovery orchestration.
pub(super) struct K2HorizonMoVADiscoveredModelMetadata {
    pub context_window: u32,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub has_vision: bool,
    pub supports_reasoning: bool,
    pub supports_tool_calls: bool,
    pub model_size_bytes: u64,
}

#[derive(Deserialize)]
struct K2HorizonMoVAShallowConfig {
    model_type: String,
    #[serde(default)]
    max_position_embeddings: Option<u32>,
}

#[derive(Deserialize)]
struct K2HorizonMoVAShallowIndex {
    weight_map: serde_json::Map<String, serde_json::Value>,
}

/// Recognizes the K2 Horizon MoVA family marker without claiming execution support.
pub(super) fn recognizes_model_type(model_type: Option<&str>) -> bool {
    model_type == Some("k2_horizon_mova")
}

/// Predicts whether startup can execute one stacked affine family member.
pub(super) fn discover_model_metadata(
    model_directory: &Path,
    config_bytes: &[u8],
) -> Option<K2HorizonMoVADiscoveredModelMetadata> {
    let config: K2HorizonMoVAShallowConfig = serde_json::from_slice(config_bytes).ok()?;
    if config.model_type != "k2_horizon_mova" {
        return None;
    }
    let context_window = config
        .max_position_embeddings
        .filter(|window| *window >= 2)?;
    if !model_directory.join("tokenizer.json").is_file()
        || !model_directory.join("chat_template.jinja").is_file()
        || !model_directory
            .join("model.safetensors.index.json")
            .is_file()
    {
        return None;
    }
    let index_bytes = fs::read(model_directory.join("model.safetensors.index.json")).ok()?;
    let index: K2HorizonMoVAShallowIndex = serde_json::from_slice(&index_bytes).ok()?;
    let mut saw_stacked = false;
    let mut saw_unstacked_only = false;
    let mut shard_file_names = HashSet::new();
    for (tensor_name, shard_file_name) in &index.weight_map {
        if let Some(shard_file_name) = shard_file_name.as_str() {
            shard_file_names.insert(shard_file_name.to_owned());
        }
        if tensor_name.contains(".mlp.switch_mlp.")
            || tensor_name.contains(".self_attn.v_experts.weight")
        {
            saw_stacked = true;
        }
        if tensor_name.contains(".mlp.experts.0.")
            || tensor_name.contains(".self_attn.v_experts.0.")
        {
            saw_unstacked_only = true;
        }
    }
    if saw_unstacked_only && !saw_stacked {
        return None;
    }
    if !saw_stacked {
        return None;
    }
    for shard_file_name in &shard_file_names {
        if !model_directory.join(shard_file_name).is_file() {
            return None;
        }
    }
    let mut model_size_bytes = 0_u64;
    for shard_file_name in shard_file_names {
        model_size_bytes = model_size_bytes.checked_add(
            fs::metadata(model_directory.join(shard_file_name))
                .ok()?
                .len(),
        )?;
    }
    Some(K2HorizonMoVADiscoveredModelMetadata {
        context_window,
        max_input_tokens: context_window - 1,
        max_output_tokens: u32::from(u16::MAX).min(context_window - 1),
        has_vision: false,
        supports_reasoning: true,
        supports_tool_calls: true,
        model_size_bytes,
    })
}
