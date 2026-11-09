//! Startup placement for dense Qwen 3.5 models.
//!
//! A dense model has no experts to page, so the startup question is whether
//! the text weights it binds fit under the effective MLX limit. That limit is
//! what macOS granted through Metal, or what the user set, and the decision
//! adds no headroom of its own. A model whose weights may fit keeps the fully
//! resident engine it always ran on; one that cannot fit goes to the
//! streaming engine, which reads weights from the SSD once dense layer
//! streaming lands there (issue #1985).
//!
//! The arithmetic lives with the streaming engine's raise check in
//! `StreamingStartupResidencyFacts`, so a live raise demands the resident
//! engine exactly when a restart would choose it.

use astronomical_model_serving::{
    EnginePlacement, ExpertMemoryAdmissionError, StreamingStartupResidencyFacts,
    ValidatedQwen3_5Artifact,
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
    // The sequence-state rate does not enter the startup decision.
    let dense_startup_facts =
        StreamingStartupResidencyFacts::for_dense_text_weights(dense_text_weight_payload_bytes, 0);
    EnginePlacement::from_startup_complete_residency_decision(
        dense_startup_facts.startup_complete_residency_decision(active_memory_ceiling_bytes),
    )
}
