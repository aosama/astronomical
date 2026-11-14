//! Feeds the memory ledger what this engine does (issue #2221, phase 1).
//!
//! The ledger only watches here. The engine keeps deciding the way it does
//! today and reports each completed forward; the ledger answers with what it
//! expected in memory, what MLX reported, and how the weights the engine
//! kept compare with the capacity the ledger would have allowed. The reading
//! travels out with the engine's memory telemetry.

use astronomical_memory_policy::ledger::{
    HeldMemory, LedgerObserver, LedgerPhase, LedgerReading, ModelMemoryShape, ObservedStep,
};

use super::Qwen3_5Model;
use crate::{MemoryPhase, MlxRamBudgetModelGeometry};

/// Layers in flight while a sparse model's prompt is read: the one being
/// multiplied and the one read ahead of it.
const SPARSE_READING_STREAM_SLOT_COUNT: u64 = 2;

/// One completed forward's memory readings, in bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CompletedForwardMemoryReadings {
    pub(crate) memory_phase: MemoryPhase,
    pub(crate) forward_token_count: usize,
    pub(crate) context_token_count: u64,
    pub(crate) live_context_state_payload_bytes: u64,
    pub(crate) active_bytes_before_forward: u64,
    pub(crate) active_bytes_after_forward: u64,
    pub(crate) peak_bytes_during_forward: u64,
}

impl Qwen3_5Model {
    /// What this model occupies and could keep, from the geometry measured
    /// at load.
    fn memory_ledger_shape(&self) -> ModelMemoryShape {
        let model_geometry = self.mlx_ram_budget().model_geometry();
        match self.dense_layer_streamer.as_ref() {
            Some(dense_layer_streamer) => dense_model_memory_shape(
                model_geometry.model_core_payload_bytes,
                dense_layer_streamer.layer_payload_bytes(),
                dense_layer_streamer.streaming_slot_bytes(),
            ),
            None => sparse_model_memory_shape(model_geometry),
        }
    }

    /// Opens the observer with what was measured at load.
    fn open_memory_ledger_observer(&self, limit_bytes: u64) -> Option<LedgerObserver> {
        let mut ledger_observer =
            LedgerObserver::new(limit_bytes, self.memory_ledger_shape()).ok()?;
        if let Some(probed_working_memory) = self.probed_prompt_layer_working_memory.get() {
            ledger_observer.record_probed_working_memory(
                LedgerPhase::ReadingPrompt {
                    span_token_count: probed_working_memory.span_token_count,
                },
                probed_working_memory.working_memory_bytes,
            );
        }
        Some(ledger_observer)
    }

    /// Reports one completed forward to the ledger.
    pub(crate) fn observe_completed_forward_in_memory_ledger(
        &self,
        forward_readings: CompletedForwardMemoryReadings,
    ) {
        let Some(ledger_phase) = ledger_phase_for(
            forward_readings.memory_phase,
            forward_readings.forward_token_count,
        ) else {
            return;
        };
        let (limit_bytes, sequence_state_bytes_per_token) = {
            let mlx_ram_budget = self.mlx_ram_budget();
            (
                mlx_ram_budget.mlx_active_memory_ceiling_bytes(),
                mlx_ram_budget
                    .model_geometry()
                    .sequence_state_bytes_per_token,
            )
        };
        let projected_context_bytes = forward_readings
            .context_token_count
            .saturating_mul(sequence_state_bytes_per_token);
        let observed_step = ObservedStep {
            phase: ledger_phase,
            limit_bytes,
            request_context_bytes: projected_context_bytes
                .max(forward_readings.live_context_state_payload_bytes),
            held_memory: HeldMemory {
                context_bytes: forward_readings.live_context_state_payload_bytes,
                kept_weight_bytes: self.reclaimable_retained_payload_bytes(),
                // A forward's streamed weights are dropped when it completes.
                stream_slot_bytes: 0,
            },
            active_bytes_before_step: forward_readings.active_bytes_before_forward,
            active_bytes_after_step: forward_readings.active_bytes_after_forward,
            peak_bytes_during_step: forward_readings.peak_bytes_during_forward,
        };
        let mut memory_ledger_observer = self.memory_ledger_observer.borrow_mut();
        if memory_ledger_observer.is_none() {
            *memory_ledger_observer = self.open_memory_ledger_observer(limit_bytes);
        }
        if let Some(ledger_observer) = memory_ledger_observer.as_mut() {
            let ledger_reading = ledger_observer.observe_step(observed_step);
            log_ledger_reading(ledger_phase, &ledger_reading);
        }
    }

    /// The request is over; the next observed forward opens a new one. The
    /// streaming engine keeps no stored state between requests.
    pub(crate) fn finish_request_in_memory_ledger(&self, request_id: u64) {
        let mut memory_ledger_observer = self.memory_ledger_observer.borrow_mut();
        let Some(ledger_observer) = memory_ledger_observer.as_mut() else {
            return;
        };
        if let Some(ledger_reading) = ledger_observer.latest_reading() {
            tracing::info!(
                request_id,
                observed_step_count = ledger_reading.observed_step_count,
                largest_unaccounted_bytes = ledger_reading.largest_unaccounted_bytes,
                largest_overcounted_bytes = ledger_reading.largest_overcounted_bytes,
                largest_kept_over_capacity_bytes = ledger_reading.largest_kept_over_capacity_bytes,
                lowest_kept_share_while_reading_permille =
                    ?ledger_reading.lowest_kept_share_while_reading_permille,
                "memory ledger watched one request"
            );
        }
        ledger_observer.finish_request(0);
    }

    /// The ledger's reading of the most recent completed forward.
    pub(crate) fn latest_memory_ledger_reading(&self) -> Option<LedgerReading> {
        self.memory_ledger_observer
            .borrow()
            .as_ref()
            .and_then(LedgerObserver::latest_reading)
    }
}

/// A sparse model's shape: its experts are what may be kept or streamed.
/// Reading a prompt holds whole layers in flight; writing a reply holds only
/// the experts one token chose.
fn sparse_model_memory_shape(model_geometry: MlxRamBudgetModelGeometry) -> ModelMemoryShape {
    ModelMemoryShape {
        core_weight_bytes: model_geometry.model_core_payload_bytes,
        streamable_weight_bytes: model_geometry.complete_expert_payload_bytes,
        reading_prompt_stream_slot_bytes: model_geometry
            .largest_complete_expert_layer_bytes
            .saturating_mul(SPARSE_READING_STREAM_SLOT_COUNT),
        writing_reply_stream_slot_bytes: model_geometry.largest_routed_expert_page_bytes,
    }
}

/// A dense model's shape: the decoder layers its streamer owns are what may
/// be kept or streamed, through the same layer slots in both phases.
fn dense_model_memory_shape(
    core_weight_bytes: u64,
    decoder_layer_payload_bytes: &[u64],
    streaming_slot_bytes: u64,
) -> ModelMemoryShape {
    let mut streamable_weight_bytes = 0_u64;
    for layer_payload_bytes in decoder_layer_payload_bytes {
        streamable_weight_bytes = streamable_weight_bytes.saturating_add(*layer_payload_bytes);
    }
    ModelMemoryShape {
        core_weight_bytes,
        streamable_weight_bytes,
        reading_prompt_stream_slot_bytes: streaming_slot_bytes,
        writing_reply_stream_slot_bytes: streaming_slot_bytes,
    }
}

/// Maps the engine's phase onto the ledger's. Generation preparation runs
/// between the two and is not a forward the ledger measures.
fn ledger_phase_for(memory_phase: MemoryPhase, forward_token_count: usize) -> Option<LedgerPhase> {
    match memory_phase {
        MemoryPhase::Prefill => Some(LedgerPhase::ReadingPrompt {
            span_token_count: u32::try_from(forward_token_count).unwrap_or(u32::MAX),
        }),
        MemoryPhase::Decode => Some(LedgerPhase::WritingReply),
        MemoryPhase::GenerationPreparation | MemoryPhase::Idle => None,
    }
}

fn log_ledger_reading(ledger_phase: LedgerPhase, ledger_reading: &LedgerReading) {
    tracing::debug!(
        ?ledger_phase,
        limit_bytes = ledger_reading.accounts.limit_bytes,
        context_bytes = ledger_reading.accounts.context_bytes,
        working_memory_bytes = ledger_reading.accounts.working_memory_bytes,
        stream_slot_bytes = ledger_reading.accounts.stream_slot_bytes,
        margin_bytes = ledger_reading.accounts.margin_bytes,
        kept_weight_capacity_bytes = ledger_reading.accounts.kept_weight_capacity_bytes,
        held_kept_weight_bytes = ledger_reading.held_memory.kept_weight_bytes,
        held_context_bytes = ledger_reading.held_memory.context_bytes,
        expected_active_bytes = ledger_reading.reconciliation.expected_active_bytes,
        actual_active_bytes = ledger_reading.reconciliation.actual_active_bytes,
        unaccounted_bytes = ledger_reading.reconciliation.unaccounted_bytes,
        overcounted_bytes = ledger_reading.reconciliation.overcounted_bytes,
        kept_over_capacity_bytes = ledger_reading.reconciliation.kept_over_capacity_bytes,
        step_peak_beyond_held_bytes = ledger_reading.step_peak_beyond_held_bytes,
        "memory ledger watched one forward"
    );
}

#[cfg(test)]
#[path = "memory_ledger_observation_tests.rs"]
mod tests;
