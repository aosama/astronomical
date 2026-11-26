//! Startup placement for dense Qwen 3.5 models.
//!
//! A dense model has no experts to page, so the startup question is whether
//! the text weights it binds fit under the effective MLX limit. That limit is
//! what macOS granted through Metal, or what the user set, and the decision
//! adds no headroom of its own. Either way the one text engine serves
//! the model: it keeps the weights in memory when they fit and reads decoder
//! layers from the SSD when they do not (issue #1985). The placement decides
//! the prompt chunk size and what the worker reports.
//!
//! The arithmetic is `StartupResidencyFacts`, the same
//! complete-residency decision a sparse model's geometry goes through.

use crate::{
    EnginePlacement, ExpertMemoryAdmissionError, StartupResidencyFacts, ValidatedQwen3_5Artifact,
};

use super::qwen3_5_model_startup_error::Qwen3_5ModelStartupError;

/// Places a dense artifact by comparing the text weights it binds with the
/// effective limit.
///
/// # Errors
///
/// Fails when the artifact's text weight sizes cannot be read, rather than
/// placing the model on an undercount.
pub(super) fn place_dense_artifact(
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_directory_path: &std::path::Path,
    active_memory_ceiling_bytes: usize,
) -> Result<EnginePlacement, Qwen3_5ModelStartupError> {
    let dense_text_weight_payload_bytes = match validated_artifact.text_weight_payload_bytes() {
        Some(text_weight_payload_bytes) if text_weight_payload_bytes > 0 => {
            text_weight_payload_bytes
        }
        _ => {
            return Err(Qwen3_5ModelStartupError::TextWeightPayloadUnreadable {
                model_directory: model_directory_path.to_path_buf(),
            });
        }
    };
    let engine_placement = place_dense_weights_under_ceiling(
        dense_text_weight_payload_bytes,
        u64::try_from(active_memory_ceiling_bytes).unwrap_or(u64::MAX),
    )
    .map_err(
        |source| Qwen3_5ModelStartupError::CompleteResidencyDecision {
            model_directory: model_directory_path.to_path_buf(),
            source,
        },
    )?;
    tracing::info!(
        dense_text_weight_payload_bytes,
        active_memory_ceiling_bytes,
        engine_placement = engine_placement.log_label(),
        "Qwen3.5 dense startup placement"
    );
    Ok(engine_placement)
}

/// Places dense text weights of `dense_text_weight_payload_bytes` under
/// `active_memory_ceiling_bytes`.
///
/// # Errors
///
/// Returns the admission error an invalid observation would carry; dense
/// inputs never produce one, but the shared decision keeps the contract.
pub(super) fn place_dense_weights_under_ceiling(
    dense_text_weight_payload_bytes: u64,
    active_memory_ceiling_bytes: u64,
) -> Result<EnginePlacement, ExpertMemoryAdmissionError> {
    let dense_startup_facts =
        StartupResidencyFacts::for_dense_text_weights(dense_text_weight_payload_bytes);
    EnginePlacement::from_startup_complete_residency_decision(
        dense_startup_facts.startup_complete_residency_decision(active_memory_ceiling_bytes),
    )
}
