//! Family-owned shallow discovery rules for `qwen4_exp` artifacts.
//!
//! The family is recognized so download preflight and diagnostics can name it,
//! but it is deliberately not executable and never advertised: no engine
//! exists yet. Reads stay bounded to the configuration document, and no rule
//! here may assume one artifact's packaging, because published checkpoints
//! differ on expert count, quantization profile, and lookup-table form.

/// Recognizes the conditional-generation wrapper and a text-only
/// distribution of the same family.
pub(super) fn recognizes_model_type(model_type: Option<&str>) -> bool {
    matches!(model_type, Some("qwen4_exp") | Some("qwen4_exp_text"))
}

/// Bounded summary of the text configuration.
///
/// This exists so diagnostics, download preflight, and tests can state what a
/// recognized artifact declares without reading weight payloads. Every field
/// comes from the nested text configuration; a document without one is not a
/// summary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpConfigurationSummary {
    pub decoder_layers: u32,
    pub context_window_tokens: u64,
    pub routed_experts: u32,
}

/// Reads the family's text configuration summary from a parsed config document.
///
/// Returns `None` when the document lacks the nested text configuration or any
/// required field, which keeps a malformed artifact from producing a partial
/// summary that diagnostics could present as authoritative.
#[must_use]
pub fn describe_configuration(
    config_value: &serde_json::Value,
) -> Option<Qwen4ExpConfigurationSummary> {
    let text_config = config_value.get("text_config")?;
    let decoder_layers = bounded_u32(text_config.get("num_hidden_layers")?)?;
    let context_window_tokens = text_config.get("max_position_embeddings")?.as_u64()?;
    let routed_experts = bounded_u32(text_config.get("num_experts")?)?;
    Some(Qwen4ExpConfigurationSummary {
        decoder_layers,
        context_window_tokens,
        routed_experts,
    })
}

fn bounded_u32(value: &serde_json::Value) -> Option<u32> {
    u32::try_from(value.as_u64()?).ok()
}
