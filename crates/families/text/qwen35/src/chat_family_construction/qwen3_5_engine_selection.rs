//! Which engine a Qwen3.5 text model starts on, and the prompt chunk it
//! reads in (issue #2585).
//!
//! Every text model starts on the streaming engine. The artifact's memory
//! geometry is still worked out, because it says whether the whole model
//! fits, which the placement reported to the worker and the prompt chunk
//! size depend on.

use ipc_protocol::WorkerChunkingConfiguration;

use crate::{
    CompleteResidencyRequirements, EnginePlacement, Qwen3_5MemoryGeometryError,
    model_memory_geometry_from_validated_artifact,
};

use super::qwen3_5_model_startup_error::Qwen3_5ModelStartupError;

/// How the streaming engine, the one engine every text model is served by,
/// starts for a model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StreamingStart {
    /// The startup arithmetic found room for the whole model, so prompts
    /// are read in the in-memory chunk size.
    pub(super) holds_whole_model: bool,
}

/// Every text model is served by the streaming engine, which keeps in
/// memory what fits and streams the rest (issue #2585). The artifact's
/// memory geometry still says whether the whole model fits, which decides
/// the prompt chunk size and nothing else.
pub(super) fn select_qwen3_5_engine_from_artifact(
    validated_artifact: &crate::ValidatedQwen3_5Artifact,
    model_directory_path: &std::path::Path,
    active_memory_limit_bytes: usize,
) -> Result<(StreamingStart, EnginePlacement), Qwen3_5ModelStartupError> {
    let geometry_placement = select_qwen3_5_engine_from_artifact_geometry(
        validated_artifact,
        model_directory_path,
        active_memory_limit_bytes,
    )?;
    Ok(streaming_engine_choice(
        geometry_placement,
        validated_artifact
            .config()
            .qwen4_exp_architecture()
            .is_some(),
    ))
}

/// How the engine starts for a model the geometry placed.
///
/// A model the limit holds whole reads prompts in the in-memory chunk size.
/// Qwen3.8-Flash-Next keeps the streaming chunk size even when its experts
/// would fit, as it always has (issue #2533).
fn streaming_engine_choice(
    geometry_placement: EnginePlacement,
    keeps_the_streaming_chunk_size: bool,
) -> (StreamingStart, EnginePlacement) {
    (
        StreamingStart {
            holds_whole_model: geometry_placement.holds_whole_model()
                && !keeps_the_streaming_chunk_size,
        },
        geometry_placement,
    )
}

/// The prompt chunk a streaming engine reads in: the in-memory size when it
/// holds the whole model, which is the size the Neural Engine programs are
/// compiled for and the one every in-memory comparison was measured at;
/// otherwise the streaming size, which bounds a streamed layer's reads.
pub(super) const fn prompt_chunk_tokens_for(
    chunking: &WorkerChunkingConfiguration,
    holds_whole_model: bool,
) -> u32 {
    if holds_whole_model {
        chunking.fixed_prompt_processing_chunk_size_tokens
    } else {
        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens
    }
}

fn select_qwen3_5_engine_from_artifact_geometry(
    validated_artifact: &crate::ValidatedQwen3_5Artifact,
    model_directory_path: &std::path::Path,
    active_memory_limit_bytes: usize,
) -> Result<EnginePlacement, Qwen3_5ModelStartupError> {
    let (model_geometry, required_headroom_bytes) =
        match model_memory_geometry_from_validated_artifact(
            validated_artifact,
            model_directory_path,
        ) {
            Ok(geometry) => geometry,
            Err(Qwen3_5MemoryGeometryError::NotSparseMixtureOfExperts) => {
                let engine_placement = super::qwen3_5_dense_engine_placement::place_dense_artifact(
                    validated_artifact,
                    model_directory_path,
                    active_memory_limit_bytes,
                )?;
                return Ok(engine_placement);
            }
            Err(source) => {
                return Err(Qwen3_5ModelStartupError::ArtifactRamGeometry {
                    model_directory: model_directory_path.to_path_buf(),
                    source,
                });
            }
        };
    let complete_residency_decision = CompleteResidencyRequirements {
        current_active_memory_bytes: model_geometry.model_core_payload_bytes,
        retained_paged_expert_payload_bytes: 0,
        complete_expert_payload_bytes: model_geometry.complete_expert_payload_bytes,
        required_headroom_bytes,
        active_memory_ceiling_bytes: u64::try_from(active_memory_limit_bytes).unwrap_or(u64::MAX),
    }
    .decide();
    tracing::info!(
        model_core_payload_bytes = model_geometry.model_core_payload_bytes,
        complete_expert_payload_bytes = model_geometry.complete_expert_payload_bytes,
        largest_complete_expert_layer_bytes = model_geometry.largest_complete_expert_layer_bytes,
        required_headroom_bytes,
        active_memory_limit_bytes,
        complete_residency_decision = ?complete_residency_decision,
        "Qwen3.5 startup complete-residency decision"
    );
    let engine_placement =
        EnginePlacement::from_startup_complete_residency_decision(complete_residency_decision)
            .map_err(
                |source| Qwen3_5ModelStartupError::CompleteResidencyDecision {
                    model_directory: model_directory_path.to_path_buf(),
                    source,
                },
            )?;
    Ok(engine_placement)
}

#[cfg(test)]
#[path = "qwen3_5_engine_selection_tests.rs"]
mod qwen3_5_engine_selection_tests;
