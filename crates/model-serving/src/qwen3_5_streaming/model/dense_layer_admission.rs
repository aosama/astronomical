//! Making room for a request by giving kept dense layers back (issue #1985).
//!
//! A dense model keeps as many leading layers resident as its plan allowed.
//! When a request needs more room than is left, admission gives the last kept
//! layers back so they stream, then checks the request again, before it would
//! ever refuse it: a slower answer is acceptable, a refused one is not (#1902).
//! Admission only gets here when its first decision was to reclaim, which it
//! makes only when the kept layers cover the shortfall.

use crate::{ContextAdmissionRequirements, InferenceEngineError, MemoryAdmissionDecision};

use super::super::inference_execution::qwen3_5_runtime_error;
use super::Qwen3_5Model;

/// The reservations a request still needs after kept layers are released.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseReleaseReservations {
    pub(crate) context_reservation_bytes: usize,
    pub(crate) maximum_expert_page_reservation_bytes: usize,
    pub(crate) temporary_workspace_reservation_bytes: usize,
    pub(crate) active_memory_ceiling_bytes: usize,
}

/// What giving kept dense layers back achieved for one request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DenseReleaseOutcome {
    /// A sparse model: its retained experts are reclaimed elsewhere.
    NotDense,
    /// The request fits after the release.
    Admitted,
    /// Every kept layer was given back and the request still does not fit.
    StillShort,
}

/// Gives back kept layers worth `required_bytes`, then decides again; while
/// the decision still asks for room and layers remain kept, gives back more.
pub(crate) fn admit_by_releasing_kept_dense_layers(
    model: &Qwen3_5Model,
    reservations: DenseReleaseReservations,
    required_bytes: usize,
) -> Result<DenseReleaseOutcome, InferenceEngineError> {
    if model.dense_layer_streamer.is_none() {
        return Ok(DenseReleaseOutcome::NotDense);
    }
    let mut bytes_to_release = u64::try_from(required_bytes).unwrap_or(u64::MAX);
    loop {
        model.release_kept_dense_layers(bytes_to_release);
        match decide_after_release(model, reservations)? {
            MemoryAdmissionDecision::Admit => return Ok(DenseReleaseOutcome::Admitted),
            MemoryAdmissionDecision::Reclaim { required_bytes }
                if model.dense_kept_layer_payload_bytes() > 0 =>
            {
                bytes_to_release = required_bytes;
            }
            _ => return Ok(DenseReleaseOutcome::StillShort),
        }
    }
}

/// Measures the memory left after a release and decides the request again.
fn decide_after_release(
    model: &Qwen3_5Model,
    reservations: DenseReleaseReservations,
) -> Result<MemoryAdmissionDecision, InferenceEngineError> {
    model
        .runtime()
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .map_err(qwen3_5_runtime_error)?;
    let memory_snapshot_after_release = model
        .runtime()
        .memory_snapshot()
        .map_err(qwen3_5_runtime_error)?;
    Ok(ContextAdmissionRequirements {
        current_active_memory_bytes: memory_snapshot_after_release.active_memory_bytes(),
        context_growth_bytes: reservations.context_reservation_bytes,
        expert_page_reservation_bytes: reservations.maximum_expert_page_reservation_bytes,
        temporary_workspace_bytes: reservations.temporary_workspace_reservation_bytes,
        retained_expert_payload_bytes: usize::try_from(model.dense_kept_layer_payload_bytes())
            .unwrap_or(usize::MAX),
        active_memory_ceiling_bytes: reservations.active_memory_ceiling_bytes,
        complete_experts_are_resident: false,
    }
    .decide())
}
