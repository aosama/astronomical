//! What the model's `config.json` says about module naming: how routed
//! experts are wrapped in the mlx-lm module tree, and the group sizes an
//! already-quantized source was written with.
//!
//! The expert rules are OptiQ's `_index_bf16_layers` architecture branches.
//! Each one exists because a family's on-disk names and its mlx-lm module
//! names disagree, and a mismatch makes every expert bank silently invisible
//! to the sweep: they never reach the allocator and keep the default width.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::QuantizationError;

/// How routed experts are named in the mlx-lm module tree for one family.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertNamingRules {
    /// `switch_mlp` (Qwen3/Qwen4, LFM2, Nemotron-H, Laguna, Flint, DeepSeek,
    /// Xing4) or `switch_glu` (the Gemma-4 default).
    pub wrapper_segment: &'static str,
    /// Qwen drops the `experts` segment when it fuses: `mlp.experts.*`
    /// becomes `mlp.switch_mlp.*`.
    pub drops_experts_segment: bool,
    /// DiffusionGemma keeps fused expert tensors under their own names.
    pub maps_fused_experts_directly: bool,
    /// Nemotron-H's `up_proj` and `down_proj` are `fc1` and `fc2`.
    pub projection_renames: BTreeMap<&'static str, &'static str>,
}

/// Group sizes an already-quantized source declares in its `quantization`
/// document: the global default and per-module overrides.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SourceQuantizationGroupSizes {
    pub default_group_size: Option<u32>,
    pub group_size_by_module: BTreeMap<String, u32>,
}

impl SourceQuantizationGroupSizes {
    /// The group size the source used for `module_name`.
    #[must_use]
    pub fn group_size_for(&self, module_name: &str) -> Option<u32> {
        match self.group_size_by_module.get(module_name) {
            Some(group_size) => Some(*group_size),
            None => self.default_group_size,
        }
    }
}

/// Both readings of `config.json`. A source without a config gets the
/// Gemma-style defaults and no source quantization, which is what OptiQ does.
pub fn read_model_config(
    source_directory: &Path,
) -> Result<(ExpertNamingRules, SourceQuantizationGroupSizes), QuantizationError> {
    let config_path = source_directory.join("config.json");
    if !config_path.is_file() {
        return Ok((
            rules_for_family(""),
            SourceQuantizationGroupSizes::default(),
        ));
    }
    let config_value = parse_config(&config_path)?;
    let family_text = family_text(&config_value);
    Ok((
        rules_for_family(&family_text),
        source_group_sizes(&config_value),
    ))
}

/// The lower-cased `architectures[0]` plus the top-level and text-tower
/// `model_type`. The text tower matters: MiniCPM-V 4.7 is `minicpmv4_7` on
/// top of a Qwen3.5-MoE, and reading only the top level named every expert
/// bank wrong.
fn family_text(config_value: &serde_json::Value) -> String {
    let mut family_parts = Vec::new();
    if let Some(architecture) = config_value["architectures"][0].as_str() {
        family_parts.push(architecture.to_lowercase());
    }
    if let Some(model_type) = config_value["model_type"].as_str() {
        family_parts.push(model_type.to_lowercase());
    }
    if let Some(text_model_type) = config_value["text_config"]["model_type"].as_str() {
        family_parts.push(text_model_type.to_lowercase());
    }
    family_parts.join(" ")
}

/// OptiQ's architecture branches, first match wins.
fn rules_for_family(family_text: &str) -> ExpertNamingRules {
    let mut rules = ExpertNamingRules {
        wrapper_segment: "switch_glu",
        drops_experts_segment: false,
        maps_fused_experts_directly: false,
        projection_renames: BTreeMap::new(),
    };
    if family_text.contains("qwen3") || family_text.contains("qwen4") {
        rules.wrapper_segment = "switch_mlp";
        rules.drops_experts_segment = true;
    } else if family_text.contains("diffusion_gemma") {
        rules.maps_fused_experts_directly = true;
    } else if family_text.contains("nemotron_h") || family_text.contains("nemotronh") {
        rules.wrapper_segment = "switch_mlp";
        rules.projection_renames.insert("up_proj", "fc1");
        rules.projection_renames.insert("down_proj", "fc2");
    } else if is_switch_mlp_family(family_text) {
        rules.wrapper_segment = "switch_mlp";
    }
    rules
}

/// Families that fuse unfused experts into `switch_mlp` with no other rename.
fn is_switch_mlp_family(family_text: &str) -> bool {
    for family_marker in ["lfm2", "laguna", "flint", "deepseek", "xing4"] {
        if family_text.contains(family_marker) {
            return true;
        }
    }
    false
}

/// The `quantization` (or `quantization_config`) document's group sizes.
fn source_group_sizes(config_value: &serde_json::Value) -> SourceQuantizationGroupSizes {
    let mut group_sizes = SourceQuantizationGroupSizes::default();
    let quantization_document = if config_value["quantization"].is_object() {
        &config_value["quantization"]
    } else {
        &config_value["quantization_config"]
    };
    let Some(quantization_entries) = quantization_document.as_object() else {
        return group_sizes;
    };
    for (entry_name, entry_value) in quantization_entries {
        if entry_name == "group_size" {
            group_sizes.default_group_size = entry_value.as_u64().map(narrow_group_size);
        } else if let Some(module_group_size) = entry_value["group_size"].as_u64() {
            group_sizes
                .group_size_by_module
                .insert(entry_name.clone(), narrow_group_size(module_group_size));
        }
    }
    group_sizes
}

/// Group sizes are small; anything that does not fit `u32` is malformed and
/// reads as zero, which no tensor divides into.
fn narrow_group_size(group_size: u64) -> u32 {
    u32::try_from(group_size).unwrap_or(0)
}

fn parse_config(config_path: &Path) -> Result<serde_json::Value, QuantizationError> {
    let config_text = match std::fs::read_to_string(config_path) {
        Ok(config_text) => config_text,
        Err(read_error) => return Err(config_unreadable(config_path, &read_error.to_string())),
    };
    match serde_json::from_str(&config_text) {
        Ok(config_value) => Ok(config_value),
        Err(parse_error) => Err(config_unreadable(config_path, &parse_error.to_string())),
    }
}

fn config_unreadable(config_path: &Path, description: &str) -> QuantizationError {
    QuantizationError::ModelConfigUnreadable {
        config_path: config_path.to_path_buf(),
        description: description.to_owned(),
    }
}
