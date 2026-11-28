//! Artifact-side RAM-budget measurements for a validated Qwen3.5 artifact.
//!
//! This adapter reads SafeTensors headers through the existing expert-layer
//! planner and maps the plans into the measured layer facts that
//! `memory/model_memory_geometry.rs` consumes. The byte arithmetic itself is
//! owned by `memory`; this file only measures. It does not load weights onto
//! the GPU and does not invent gigabyte souvenirs.

use std::path::Path;

use thiserror::Error;

use crate::ModelMemoryGeometry;
use crate::memory::{
    MeasuredExpertLayerPayload, ModelMemoryGeometryError,
    model_memory_geometry_from_measured_layer_facts,
};
use crate::qwen3_5_core::artifacts::{Qwen3_5FeedForwardArchitecture, ValidatedQwen3_5Artifact};
use engine_core::expert_paging::QuantizedExpertLayerPlan;

/// Why disk-only RAM measurements could not be composed for this artifact.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Qwen3_5MemoryGeometryError {
    #[error("this Qwen3.5 artifact has no sparse expert payload")]
    NotSparseMixtureOfExperts,
    #[error("expert layer plans could not be read from the validated artifact")]
    ExpertLayerPlan,
    #[error("expert payload accounting overflowed")]
    ExpertPayloadOverflow,
}

/// Returns payload geometry and startup complete-residency headroom for one artifact.
pub fn model_memory_geometry_from_validated_artifact(
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_directory: &Path,
) -> Result<(ModelMemoryGeometry, u64), Qwen3_5MemoryGeometryError> {
    if validated_artifact.config().feed_forward_architecture()
        != Qwen3_5FeedForwardArchitecture::MixtureOfExperts
    {
        return Err(Qwen3_5MemoryGeometryError::NotSparseMixtureOfExperts);
    }
    let layer_plans = super::atlas_expert_layer_plans::expert_layer_plans_for_validated_artifact(
        validated_artifact,
        model_directory,
    )
    .map_err(|_| Qwen3_5MemoryGeometryError::ExpertLayerPlan)?;
    let measured_layer_payloads = measured_layer_payloads(&layer_plans)?;
    let complete_residency_transient_bytes = complete_residency_transient_bytes(&layer_plans)?;
    model_memory_geometry_from_measured_layer_facts(
        &measured_layer_payloads,
        validated_artifact.total_payload_bytes(),
        usize::try_from(validated_artifact.config().experts_per_token()).unwrap_or(usize::MAX),
        complete_residency_transient_bytes,
    )
    .map_err(map_composer_error)
}

fn measured_layer_payloads(
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<Vec<MeasuredExpertLayerPayload>, Qwen3_5MemoryGeometryError> {
    layer_plans
        .iter()
        .map(|layer_plan| {
            Ok(MeasuredExpertLayerPayload::new(
                layer_plan
                    .complete_expert_payload_byte_count()
                    .map_err(|_| Qwen3_5MemoryGeometryError::ExpertPayloadOverflow)?,
                layer_plan.expert_capacity,
            ))
        })
        .collect()
}

/// Working memory the startup placement decision keeps free beside a model
/// held whole.
///
/// With MLX linked this is the size of the largest layer's gate and up
/// projections where they share one encoding, and zero where they do not.
/// It was sized for a fusion of those two projections that the engine no
/// longer performs; the value is kept because it was measured adequate:
/// a 35B-A3B at 4 bits held whole under a 36.5 GB limit needed 0.385 GB
/// beyond its held weights in the largest step of an 11,000-token prompt,
/// against 0.531 GB reserved here (2026-11-26). Without MLX the largest
/// complete layer stands in so unit tests still reserve a meaningful floor.
///
/// This reserve is small beside what the memory governor sets aside once
/// the model serves (about 1.5 GB of margin, working buffers and buffer room
/// in that run, before the room for macOS's wired limit), so near the limit
/// the governor, not this figure, decides whether every layer stays held.
/// That measurement read 2,048-token chunks; on an M5 or newer the engine's
/// Automatic chunk for this model is 4,096, whose step peaks about 0.35 GB
/// higher, and under a 25 GB limit every expert still stayed held
/// (2026-11-27).
fn complete_residency_transient_bytes(
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<u64, Qwen3_5MemoryGeometryError> {
    #[cfg(feature = "direct-mlx")]
    {
        super::expert_gate_up_fusion_plan::maximum_expert_gate_up_fusion_transient_payload_bytes(
            layer_plans,
        )
        .map_err(|_| Qwen3_5MemoryGeometryError::ExpertLayerPlan)
    }
    #[cfg(not(feature = "direct-mlx"))]
    {
        layer_plans
            .iter()
            .try_fold(0_u64, |largest_layer_bytes, layer_plan| {
                Ok(largest_layer_bytes.max(
                    layer_plan
                        .complete_expert_payload_byte_count()
                        .map_err(|_| Qwen3_5MemoryGeometryError::ExpertPayloadOverflow)?,
                ))
            })
    }
}

fn map_composer_error(composer_error: ModelMemoryGeometryError) -> Qwen3_5MemoryGeometryError {
    match composer_error {
        ModelMemoryGeometryError::NotSparseMixtureOfExperts => {
            Qwen3_5MemoryGeometryError::NotSparseMixtureOfExperts
        }
        ModelMemoryGeometryError::ExpertPayloadOverflow => {
            Qwen3_5MemoryGeometryError::ExpertPayloadOverflow
        }
    }
}
