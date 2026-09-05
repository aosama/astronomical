//! Shallow discovery rules for executable ModernBERT embedding artifacts.
//!
//! Geometry and completeness prove an executable text-embedding artifact.
//! Embedding inference is a single encoder forward pass, so discovery derives
//! a vector width and prompt budget instead of autoregressive token limits.

use std::fs;
use std::path::Path;

/// Family-derived metadata returned to neutral discovery orchestration.
pub(super) struct ModernBertDiscoveredModelMetadata {
    pub vector_width: u32,
    pub max_input_tokens: u32,
    pub model_size_bytes: u64,
}

/// Recognizes the MLX-converted ModernBERT embedding model type.
pub(super) fn recognizes_model_type(model_type: Option<&str>) -> bool {
    matches!(model_type, Some("modernbert"))
}

/// Validates shallow ModernBERT completeness and derives public discovery metadata.
pub(super) fn discover_model_metadata(
    model_directory: &Path,
    config_value: &serde_json::Value,
) -> Option<ModernBertDiscoveredModelMetadata> {
    if !model_directory.join("model.safetensors").is_file()
        || !model_directory.join("tokenizer.json").is_file()
    {
        return None;
    }
    let vector_width = config_value.get("hidden_size")?.as_u64()? as u32;
    if vector_width == 0 {
        return None;
    }
    let max_input_tokens = config_value.get("max_position_embeddings")?.as_u64()? as u32;
    if max_input_tokens < 2 {
        return None;
    }
    let quantization = config_value.get("quantization")?;
    let bits = quantization.get("bits")?.as_u64()?;
    if bits != 8 {
        // Only the reviewed affine 8-bit profile is executable; other widths
        // stay undiscoverable until an engine validates them.
        return None;
    }
    let model_size_bytes = measure_model_safetensors_bytes(model_directory)?;
    Some(ModernBertDiscoveredModelMetadata {
        vector_width,
        max_input_tokens,
        model_size_bytes,
    })
}

fn measure_model_safetensors_bytes(model_directory: &Path) -> Option<u64> {
    fs::metadata(model_directory.join("model.safetensors"))
        .ok()?
        .len()
        .into()
}
