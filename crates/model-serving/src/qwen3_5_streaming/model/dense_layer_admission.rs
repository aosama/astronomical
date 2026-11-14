//! What a dense model's kept layers mean to admission (issue #1985).
//!
//! A dense model keeps as many leading layers resident as its plan allowed.
//! Kept layers are reclaimable payload: when a request needs more room than
//! is left, admission gives the last kept layers back so they stream, then
//! checks the request again, before it would ever refuse it: a slower answer
//! is acceptable, a refused one is not (#1902). Streamed layers need their
//! slots, which admission charges only while some layer streams.

use crate::{ContextAdmissionRequirements, InferenceEngineError, MemoryAdmissionDecision};

use super::super::inference_execution::qwen3_5_runtime_error;
use super::Qwen3_5Model;
use super::dense_layer_streamer::Qwen3_5DenseLayerStreamer;

/// The reservations a request still needs after kept layers are released;
/// the layer slots are charged afresh, since releasing a layer can start
/// streaming.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseReleaseReservations {
    pub(crate) context_reservation_bytes: usize,
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
    /// Every kept layer was given back and the request still does not fit;
    /// the planned layers are kept again before the refusal.
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
            _ => {
                // The request is refused either way, so streaming the
                // released layers would only slow the next one.
                model.replan_kept_dense_layers_under_installed_limit()?;
                return Ok(DenseReleaseOutcome::StillShort);
            }
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
        expert_page_reservation_bytes: usize::try_from(
            model.dense_streaming_slot_reservation_bytes(),
        )
        .unwrap_or(usize::MAX),
        temporary_workspace_bytes: reservations.temporary_workspace_reservation_bytes,
        // A dense model has no retained experts; its kept layers are the
        // reclaimable payload this field carries.
        retained_expert_payload_bytes: usize::try_from(model.dense_kept_layer_payload_bytes())
            .unwrap_or(usize::MAX),
        active_memory_ceiling_bytes: reservations.active_memory_ceiling_bytes,
        complete_experts_are_resident: false,
    }
    .decide())
}

impl Qwen3_5Model {
    /// The text weight bytes in memory: the bound tensors less any dense
    /// decoder layers that are read per forward rather than kept.
    pub(crate) fn resident_model_payload_byte_count(&self) -> u64 {
        let bound_payload_bytes = self.base.resident_model_payload_byte_count();
        match &self.dense_layer_streamer {
            Some(dense_layer_streamer) => bound_payload_bytes
                .saturating_sub(dense_layer_streamer.streamed_layer_payload_bytes()),
            None => bound_payload_bytes,
        }
    }

    /// Bytes a forward's streamed layers occupy with the current kept count:
    /// zero for a sparse model and while every dense layer is kept.
    pub(crate) fn dense_streaming_slot_reservation_bytes(&self) -> u64 {
        self.dense_layer_streamer
            .as_ref()
            .map_or(0, Qwen3_5DenseLayerStreamer::streaming_slot_bytes_in_use)
    }

    /// The bytes one forward's admission reserves for weights it streams: a
    /// sparse model's routed-page reservation, or the slots a dense model's
    /// streamed layers use. Post-forward learning gives the peak back at most
    /// this much, because this is what the next admission adds on top of the
    /// learned workspace (issue #2218).
    pub(crate) fn streamed_weight_reservation_bytes_for_forward(
        &self,
        forward_token_count: usize,
    ) -> Result<u64, crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError> {
        if self.expert_pager.is_some() {
            return self.expert_page_reservation_bytes_for_forward(forward_token_count);
        }
        Ok(self.dense_streaming_slot_reservation_bytes())
    }

    /// Payload admission may reclaim: retained experts, plus a dense model's
    /// kept layers, which stream once given back.
    pub(crate) fn reclaimable_retained_payload_bytes(&self) -> u64 {
        self.expert_weight_memory_cache_statistics()
            .resident_payload_byte_count
            .saturating_add(self.dense_kept_layer_payload_bytes())
    }

    /// Where this model's decoder layers live, for status; `None` for a
    /// sparse model.
    pub(crate) fn dense_layer_residency_report(
        &self,
    ) -> Option<astronomical_ipc_protocol::WorkerDenseLayerResidency> {
        let dense_layer_streamer = self.dense_layer_streamer.as_ref()?;
        Some(astronomical_ipc_protocol::WorkerDenseLayerResidency {
            decoder_layer_count: u32::try_from(dense_layer_streamer.layer_payload_bytes().len())
                .unwrap_or(u32::MAX),
            kept_layer_count: u32::try_from(dense_layer_streamer.kept_layer_count())
                .unwrap_or(u32::MAX),
            streamed_layer_payload_bytes: dense_layer_streamer.streamed_layer_payload_bytes(),
        })
    }

    /// Payload of the dense layers kept resident, which admission may give back.
    pub(crate) fn dense_kept_layer_payload_bytes(&self) -> u64 {
        self.dense_layer_streamer
            .as_ref()
            .map_or(0, Qwen3_5DenseLayerStreamer::kept_layer_payload_bytes)
    }

    /// Gives back kept dense layers until `required_bytes` are freed,
    /// returning the bytes freed; those layers stream until a re-plan.
    pub(crate) fn release_kept_dense_layers(&self, required_bytes: u64) -> u64 {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return 0;
        };
        let released_bytes = dense_layer_streamer.release_trailing_kept_layers(required_bytes);
        if released_bytes > 0 {
            tracing::info!(
                released_bytes,
                kept_layer_count = dense_layer_streamer.kept_layer_count(),
                "gave kept dense layers back to make room"
            );
        }
        released_bytes
    }
}
