//! Family-owned shallow discovery rules for executable folded ternary packs.

use std::path::Path;

/// Lowest context window disk discovery accepts: one token for the prompt and
/// one for the completion. Anything smaller cannot serve a single user turn.
pub(super) const MINIMUM_SERVABLE_CONTEXT_WINDOW_TOKENS: u64 = 2;

/// Family-derived metadata returned to neutral discovery orchestration.
pub(super) struct PrismHadamardDiscoveredModelMetadata {
    pub context_window: u32,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub supports_reasoning: bool,
    pub supports_tool_calls: bool,
    pub model_size_bytes: u64,
}

/// Recognizes the folded ternary pack model type.
pub(super) fn recognizes_model_type(model_type: Option<&str>) -> bool {
    matches!(model_type, Some("prism_hadamard_qwen35"))
}

/// Validates shallow folded-pack completeness and derives public discovery
/// metadata. The pack is a four-file artifact — config, Hadamard contract,
/// tokenizer, and single-file weights — and anything less is not executable.
pub(super) fn discover_model_metadata(
    model_directory: &Path,
    config_value: &serde_json::Value,
) -> Option<PrismHadamardDiscoveredModelMetadata> {
    if !model_directory.join("hadamard.json").is_file()
        || !model_directory.join("tokenizer.json").is_file()
        || !model_directory.join("model.safetensors").is_file()
    {
        return None;
    }

    // The folded document carries the text window under text_config, matching
    // the shipped pack layout.
    let context_window_tokens = config_value
        .get("text_config")
        .and_then(|text_config| text_config.get("max_position_embeddings"))
        .and_then(serde_json::Value::as_u64)?;
    let context_window = u32::try_from(context_window_tokens).ok()?;
    if context_window_tokens < MINIMUM_SERVABLE_CONTEXT_WINDOW_TOKENS {
        return None;
    }

    let model_size_bytes = std::fs::metadata(model_directory.join("model.safetensors"))
        .ok()?
        .len();

    Some(PrismHadamardDiscoveredModelMetadata {
        context_window,
        max_input_tokens: context_window - 1,
        max_output_tokens: u32::from(u16::MAX).min(context_window - 1),
        // The Qwen text dialect (thinking markers + tool markers) is shared
        // with the qwen3_5 text processor the family reuses.
        supports_reasoning: true,
        supports_tool_calls: true,
        model_size_bytes,
    })
}
