//! Startup placement for dense Qwen 3.5 models.
//!
//! A dense model has no experts to page, so the only startup question is
//! whether its text weights fit under the effective MLX limit at all. That
//! limit is what macOS granted through Metal, or what the user set; this
//! module adds no headroom of its own. The answer is therefore "does not fit"
//! or "may fit": a model that may fit keeps the fully resident engine it ran
//! on before, and one that does not fit goes to the streaming engine, which
//! reads weights from the SSD.

use astronomical_model_serving::{
    CompleteResidencyRequirements, EnginePlacement, ExpertMemoryAdmissionError,
    ValidatedQwen3_5Artifact,
};

use super::qwen3_5_model_startup_error::Qwen3_5ModelStartupError;

/// Places a dense artifact by comparing the text weights it binds with the
/// effective limit.
pub(super) fn place_dense_artifact(
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_directory_path: &std::path::Path,
    active_memory_limit_bytes: usize,
) -> Result<EnginePlacement, Qwen3_5ModelStartupError> {
    let dense_weight_payload_bytes = validated_artifact.target_weight_payload_bytes();
    let engine_placement = dense_engine_placement(
        dense_weight_payload_bytes,
        u64::try_from(active_memory_limit_bytes).unwrap_or(u64::MAX),
    )
    .map_err(
        |source| Qwen3_5ModelStartupError::CompleteResidencyDecision {
            model_directory: model_directory_path.to_path_buf(),
            source,
        },
    )?;
    tracing::info!(
        dense_weight_payload_bytes,
        active_memory_limit_bytes,
        engine_placement = engine_placement.log_label(),
        "Qwen3.5 dense startup placement"
    );
    Ok(engine_placement)
}

/// Places a dense model whose bound text weights total
/// `dense_weight_payload_bytes` under `active_memory_ceiling_bytes`.
///
/// # Errors
///
/// Returns the admission error an invalid observation would carry; dense
/// inputs never produce one, but the shared decision keeps the contract.
pub(super) fn dense_engine_placement(
    dense_weight_payload_bytes: u64,
    active_memory_ceiling_bytes: u64,
) -> Result<EnginePlacement, ExpertMemoryAdmissionError> {
    let complete_residency_decision = CompleteResidencyRequirements {
        current_active_memory_bytes: dense_weight_payload_bytes,
        retained_paged_expert_payload_bytes: 0,
        complete_expert_payload_bytes: 0,
        required_headroom_bytes: 0,
        active_memory_ceiling_bytes,
    }
    .decide();
    EnginePlacement::from_startup_complete_residency_decision(complete_residency_decision)
}
