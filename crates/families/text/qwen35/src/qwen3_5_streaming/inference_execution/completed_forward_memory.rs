//! Post-forward MLX sampling: what a forward left in memory and how high it
//! peaked.
//!
//! The memory governor is told about every completed forward and answers
//! with how much the kept weights may hold (issue #2221). Nothing else is
//! learned here: the MLX working buffers a step needs is what the governor
//! measured, not an estimate fitted to earlier forwards.
//!
//! The sequence:
//!
//! 1. Sample active and peak MLX bytes under attribution.
//! 2. Tell the governor, which applies the capacity it answers with.
//! 3. In the reply phase, resize the hot-expert tables to that capacity.

use runtime_integration::MlxMemorySnapshot;

use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::qwen3_5_streaming::model::memory_governance::CompletedForwardMemoryReadings;
use crate::{InferenceEngineError, MemoryPhase, PerformanceAttribution, PerformanceOperation};

use super::fatal_runtime_error;

/// A memory observation whose raw MLX snapshot is captured at one owner-thread
/// instant. The residency claim paired with this snapshot is derived from the
/// reconciled breakdown of this exact snapshot at emission time, so a claim
/// can never disagree with its own measurement (issue #337).
#[derive(Debug)]
pub(in crate::qwen3_5_streaming) struct CompletedForwardMemoryObservation {
    pub(in crate::qwen3_5_streaming) mlx_memory_snapshot: MlxMemorySnapshot,
}

/// Which forward was measured: its phase and how many tokens it ran.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::qwen3_5_streaming) struct MeasuredForward {
    pub(in crate::qwen3_5_streaming) memory_phase: MemoryPhase,
    pub(in crate::qwen3_5_streaming) forward_token_count: usize,
}

impl MeasuredForward {
    /// A prompt forward over `forward_token_count` tokens.
    pub(in crate::qwen3_5_streaming) const fn prompt(forward_token_count: usize) -> Self {
        Self {
            memory_phase: MemoryPhase::Prefill,
            forward_token_count,
        }
    }

    /// One reply token through every layer.
    pub(in crate::qwen3_5_streaming) const fn reply_token() -> Self {
        Self {
            memory_phase: MemoryPhase::Decode,
            forward_token_count: 1,
        }
    }
}

/// Captures the raw MLX snapshot at one instant.
pub(in crate::qwen3_5_streaming) fn capture_completed_forward_memory_observation(
    model: &Qwen3_5Model,
) -> Result<CompletedForwardMemoryObservation, InferenceEngineError> {
    let mlx_memory_snapshot = model
        .memory_snapshot_with_forward_peak()
        .map_err(fatal_runtime_error)?;
    Ok(CompletedForwardMemoryObservation {
        mlx_memory_snapshot,
    })
}

/// Measures a completed forward that wrote reply tokens, against the
/// active memory before it and the request's stored state.
pub(in crate::qwen3_5_streaming) fn measure_completed_reply_forward(
    model: &Qwen3_5Model,
    active_request: &mut super::engine_request::StreamingEngineRequest,
    active_memory_bytes_before_forward: usize,
) -> Result<CompletedForwardMemoryObservation, InferenceEngineError> {
    measure_completed_forward(
        model,
        MeasuredForward::reply_token(),
        request_stored_state_bytes(active_request),
        active_memory_bytes_before_forward,
        &mut active_request.performance_attribution,
    )
}

/// Bytes the request keeps from one forward to the next: the target's
/// stored state and, while a hosted drafter drafts, the rows kept for it.
/// The drafter's rows are about 47 MB on a dense 27B after a 2,600-token
/// prompt and grow with the context up to the drafter's row limit, so they
/// are counted, not left to the ledger's margin (issue #2572).
pub(in crate::qwen3_5_streaming) fn request_stored_state_bytes(
    active_request: &super::engine_request::StreamingEngineRequest,
) -> u64 {
    let drafter_kept_row_bytes = active_request
        .engine_extension
        .draft_tree_drafting
        .as_ref()
        .map_or(
            0,
            super::draft_tree_reply_drafting::DraftTreeReplyDrafting::kept_row_byte_count,
        );
    active_request
        .request_decoder_state
        .payload_byte_count()
        .saturating_add(drafter_kept_row_bytes)
}

/// Samples MLX after a forward, tells the memory governor what the forward
/// measured, and returns the sample.
///
/// The returned snapshot is also used for user-visible memory telemetry.
/// Returning the exact sample the governor was told about prevents status
/// from presenting a different instant than the one that moved the kept
/// weights' capacity.
pub(in crate::qwen3_5_streaming) fn measure_completed_forward(
    model: &Qwen3_5Model,
    measured_forward: MeasuredForward,
    live_context_state_payload_bytes: u64,
    active_memory_bytes_before_forward: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<CompletedForwardMemoryObservation, InferenceEngineError> {
    let observation = performance_attribution.measure_operation(
        PerformanceOperation::CompletedForwardMemorySnapshot,
        |_performance_attribution| capture_completed_forward_memory_observation(model),
    )?;
    let memory_snapshot_after_forward = &observation.mlx_memory_snapshot;
    model.govern_completed_forward(CompletedForwardMemoryReadings {
        memory_phase: measured_forward.memory_phase,
        forward_token_count: measured_forward.forward_token_count,
        live_context_state_payload_bytes,
        active_bytes_before_forward: u64::try_from(active_memory_bytes_before_forward)
            .unwrap_or(u64::MAX),
        active_bytes_after_forward: u64::try_from(
            memory_snapshot_after_forward.active_memory_bytes(),
        )
        .unwrap_or(u64::MAX),
        peak_bytes_during_forward: u64::try_from(memory_snapshot_after_forward.peak_memory_bytes())
            .unwrap_or(u64::MAX),
    });
    model.govern_system_wired_memory();
    if measured_forward.memory_phase == MemoryPhase::Decode {
        resize_hot_expert_tables_to_the_capacity(model);
    }
    Ok(observation)
}

/// Issue #512: the hot-expert tables are sized at the reply's start, before
/// any reply step has been measured. The governor's reply-step measurement
/// settles within a few tokens and the capacity rises with it; a slot count
/// frozen at the first size kept the tables churning while the capacity
/// admitted gigabytes more. Every completed reply step re-sizes here.
fn resize_hot_expert_tables_to_the_capacity(model: &Qwen3_5Model) {
    let hot_expert_warm_slot_count =
        super::decode_expert_memory_handoff::decode_warm_slot_count(model);
    if hot_expert_warm_slot_count != model.hot_expert_warm_slot_count() {
        model.set_hot_expert_warm_slot_count(hot_expert_warm_slot_count);
    }
}
