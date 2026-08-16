//! Family-owned shallow discovery rules for executable Qwen3.5 artifacts.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Family-derived metadata returned to neutral discovery orchestration.
pub(super) struct Qwen3_5DiscoveredModelMetadata {
    pub context_window: u32,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub has_vision: bool,
    pub supports_reasoning: bool,
    pub supports_tool_calls: bool,
    pub model_size_bytes: u64,
}

/// Recognizes every Qwen model type served by the existing runtime.
pub(super) fn recognizes_model_type(model_type: Option<&str>) -> bool {
    matches!(
        model_type,
        Some("qwen3_5") | Some("qwen3_5_moe") | Some("qwen3_5_moe_vision")
    )
}

/// Validates shallow Qwen completeness and derives public discovery metadata.
pub(super) fn discover_model_metadata(
    model_directory: &Path,
    config_value: &serde_json::Value,
    max_output_tokens: u32,
) -> Option<Qwen3_5DiscoveredModelMetadata> {
    if !model_directory
        .join("model.safetensors.index.json")
        .is_file()
        || !model_directory.join("tokenizer.json").is_file()
    {
        return None;
    }

    // A missing MTP-only shard is valid for target-only serving. Every shard
    // that carries at least one target or vision tensor remains mandatory.
    let index_path = model_directory.join("model.safetensors.index.json");
    let has_vision = if let Ok(index_bytes) = fs::read(&index_path)
        && let Ok(index_document) = serde_json::from_slice::<serde_json::Value>(&index_bytes)
        && let Some(weight_map) = index_document
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
    {
        let mut shard_file_names = HashSet::new();
        let mut required_shard_file_names = HashSet::new();
        for (tensor_name, tensor_shard_file_name) in weight_map {
            if let Some(tensor_shard_file_name) = tensor_shard_file_name.as_str() {
                shard_file_names.insert(tensor_shard_file_name.to_owned());
                if !contains_mtp_component(tensor_name) {
                    required_shard_file_names.insert(tensor_shard_file_name.to_owned());
                }
            }
        }
        for shard_file_name in &shard_file_names {
            if !model_directory.join(shard_file_name).is_file()
                && required_shard_file_names.contains(shard_file_name)
            {
                return None;
            }
        }
        weight_map
            .keys()
            .any(|tensor_name| tensor_name.starts_with("vision_tower."))
    } else {
        false
    };

    let context_window = config_value
        .get("text_config")
        .and_then(|text_config| text_config.get("max_position_embeddings"))
        .or_else(|| config_value.get("max_position_embeddings"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;

    Some(Qwen3_5DiscoveredModelMetadata {
        context_window,
        max_input_tokens: context_window.saturating_sub(max_output_tokens),
        max_output_tokens,
        has_vision,
        // The Qwen text processor owns both structured output contracts.
        supports_reasoning: true,
        supports_tool_calls: true,
        model_size_bytes: measure_model_safetensors_bytes(model_directory)?,
    })
}

fn contains_mtp_component(tensor_name: &str) -> bool {
    tensor_name
        .split('.')
        .any(|tensor_name_component| tensor_name_component == "mtp")
}

fn measure_model_safetensors_bytes(model_directory: &Path) -> Option<u64> {
    let mut pending_directories = vec![model_directory.to_path_buf()];
    let mut measured_safetensors_paths = HashSet::new();
    let mut model_size_bytes = 0_u64;
    while let Some(pending_directory) = pending_directories.pop() {
        for directory_entry in fs::read_dir(pending_directory).ok()? {
            let directory_entry = directory_entry.ok()?;
            let entry_path = directory_entry.path();
            let entry_file_type = directory_entry.file_type().ok()?;
            if entry_file_type.is_dir() {
                pending_directories.push(entry_path);
                continue;
            }
            if entry_path
                .extension()
                .and_then(|extension| extension.to_str())
                != Some("safetensors")
                || !entry_path.is_file()
            {
                continue;
            }
            let canonical_safetensors_path = fs::canonicalize(&entry_path).ok()?;
            if measured_safetensors_paths.insert(canonical_safetensors_path) {
                model_size_bytes =
                    model_size_bytes.checked_add(fs::metadata(entry_path).ok()?.len())?;
            }
        }
    }
    Some(model_size_bytes)
}
