//! Making room for a request by giving kept dense layers back (issue #1985).
//!
//! A dense model keeps as many leading layers resident as its plan allowed.
//! When a request needs more room than is left, admission gives the last kept
//! layers back so they stream, then checks the request again, before it would
//! ever refuse it: a slower answer is acceptable, a refused one is not (#1902).

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

/// For a dense model, releases kept layers worth `required_bytes` and decides
/// again: `Ok(())` admits, `Err(())` means the request still does not fit
/// with every layer streaming. `None` for a sparse model, whose retained
/// experts are reclaimed elsewhere.
pub(crate) fn admit_by_releasing_kept_dense_layers(
    model: &Qwen3_5Model,
    reservations: DenseReleaseReservations,
    required_bytes: usize,
) -> Result<Option<Result<(), ()>>, InferenceEngineError> {
    if model.dense_layer_streamer.is_none() {
        return Ok(None);
    }
    model.release_kept_dense_layers(u64::try_from(required_bytes).unwrap_or(u64::MAX));
    model
        .runtime()
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .map_err(qwen3_5_runtime_error)?;
    let memory_snapshot_after_release = model
        .runtime()
        .memory_snapshot()
        .map_err(qwen3_5_runtime_error)?;
    let admission_after_release = ContextAdmissionRequirements {
        current_active_memory_bytes: memory_snapshot_after_release.active_memory_bytes(),
        context_growth_bytes: reservations.context_reservation_bytes,
        expert_page_reservation_bytes: reservations.maximum_expert_page_reservation_bytes,
        temporary_workspace_bytes: reservations.temporary_workspace_reservation_bytes,
        retained_expert_payload_bytes: usize::try_from(model.dense_kept_layer_payload_bytes())
            .unwrap_or(usize::MAX),
        active_memory_ceiling_bytes: reservations.active_memory_ceiling_bytes,
        complete_experts_are_resident: false,
    }
    .decide();
    Ok(Some(match admission_after_release {
        MemoryAdmissionDecision::Admit => Ok(()),
        _ => Err(()),
    }))
}
