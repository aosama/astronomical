//! Family-owned shallow classification rules for Laguna artifacts.

/// Recognizes the Laguna family marker without claiming execution support.
pub(super) fn recognizes_model_type(model_type: Option<&str>) -> bool {
    model_type == Some("laguna")
}
