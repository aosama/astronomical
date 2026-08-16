//! Family-owned shallow classification rules for DeepSeek-V4 artifacts.

/// Recognizes the DeepSeek-V4 family marker without claiming execution support.
pub(super) fn recognizes_model_type(model_type: Option<&str>) -> bool {
    model_type == Some("deepseek_v4")
}
