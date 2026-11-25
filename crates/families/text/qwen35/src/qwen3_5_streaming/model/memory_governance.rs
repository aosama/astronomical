//! This engine's side of the memory governor (issue #2221).
//!
//! The governor in the policy crate owns the ledger. This file tells it what
//! the engine does (a request starts, the reply begins, a forward was
//! measured, a forward hit the limit, the limit moved, the request ended)
//! and applies the capacity each answer carries. `apply_kept_weight_capacity`
//! is the only place the astro expert cache is told how much it may keep.

use memory_policy::ledger::{
    HeldMemory, KeptWeightCapacity, LedgerPhase, LedgerReading, MeasuredStep, MemoryGovernor,
    ModelMemoryShape,
};

use super::Qwen3_5Model;
use crate::{MemoryPhase, ModelMemoryGeometry};

/// Layers in flight while a sparse model's prompt is read: the one being
/// multiplied and the one read ahead of it.
const SPARSE_READING_STREAM_SLOT_COUNT: u64 = 2;

/// One completed forward's memory readings, in bytes.
#[derive(Clone, Copy, Debug)]
pub struct CompletedForwardMemoryReadings {
    pub memory_phase: MemoryPhase,
    pub forward_token_count: usize,
    pub live_context_state_payload_bytes: u64,
    pub active_bytes_before_forward: u64,
    pub active_bytes_after_forward: u64,
    pub peak_bytes_during_forward: u64,
}

/// The memory governor and what this engine holds from it between events.
#[derive(Debug, Default)]
pub struct GovernedMemory {
    /// The owner of this model's memory ledger, opened once loading has
    /// measured what it can.
    pub(super) governor: std::cell::RefCell<Option<MemoryGovernor>>,
    /// The redo a forward that hit the limit is being retried under, until
    /// the retried forward is measured or the request ends.
    pub(super) pending_redo_attempt: std::cell::RefCell<Option<memory_policy::ledger::RedoAttempt>>,
    /// Room leased for publishing astro SSD prompt cache blocks, until the next
    /// forward completes.
    pub(super) publication_workspace_lease: std::cell::Cell<Option<memory_policy::ledger::LeaseId>>,
    /// Room leased for blocks read back from the `astro_ssd_prompt_cache`, until they
    /// have joined the request's stored state.
    pub(super) restore_workspace_lease: std::cell::Cell<Option<memory_policy::ledger::LeaseId>>,
    /// What MLX held at the last layer boundary of a prompt that is being
    /// measured one layer at a time; `None` outside such a prompt.
    pub(super) prompt_layer_boundary_active_bytes: std::cell::Cell<Option<u64>>,
    /// Whether the forward in flight was measured one layer at a time, so
    /// it is not measured again as one large step when it completes.
    pub(super) measured_forward_layer_by_layer: std::cell::Cell<bool>,
    /// The largest peak of the prompt layers measured since the forward in
    /// flight began. Measuring each layer resets MLX's peak counter, so the
    /// forward's own peak has to be kept here.
    pub(super) largest_prompt_layer_peak_bytes: std::cell::Cell<usize>,
}

impl Qwen3_5Model {
    /// What this model occupies and could keep, from the geometry measured
    /// at load.
    fn memory_ledger_shape(&self) -> ModelMemoryShape {
        let model_geometry = self.memory_facts().model_geometry();
        match self.dense_layer_streamer.as_ref() {
            Some(dense_layer_streamer) => dense_model_memory_shape(
                model_geometry.model_core_payload_bytes,
                dense_layer_streamer.layer_payload_bytes(),
                dense_layer_streamer.streaming_slot_bytes(),
            ),
            None => sparse_model_memory_shape(self.geometry_with_drafting_head(model_geometry)),
        }
    }

    /// Opens the governor once loading has measured what it can, and gives
    /// the astro expert cache its first capacity.
    pub fn open_memory_governor(&self) {
        let limit_bytes = self.memory_facts().mlx_active_memory_ceiling_bytes();
        let Ok(mut memory_governor) = MemoryGovernor::new(limit_bytes, self.memory_ledger_shape())
        else {
            tracing::warn!("the memory governor could not open under a zero limit");
            return;
        };
        if let Some(probed_mlx_working_buffers) = self.probed_prompt_layer_mlx_working_buffers.get()
        {
            let _ = memory_governor.record_probe(
                LedgerPhase::ReadingPrompt {
                    span_token_count: probed_mlx_working_buffers.span_token_count,
                },
                probed_mlx_working_buffers.mlx_working_buffer_bytes,
            );
            let _ = memory_governor.record_probe(
                LedgerPhase::WritingReply,
                probed_mlx_working_buffers.reply_token_mlx_working_buffer_bytes,
            );
        }
        let idle_capacity = memory_governor.kept_weight_capacity();
        *self.governed_memory.governor.borrow_mut() = Some(memory_governor);
        self.apply_kept_weight_capacity(idle_capacity);
        if let Err(keep_error) = self.keep_dense_layers_the_governor_has_room_for() {
            // The layers stream instead, which is still correct.
            tracing::warn!(error = %keep_error, "could not keep dense layers at load");
        }
    }

    /// The room a holder that pays to fill itself may fill while no request
    /// runs: the smaller of the capacity now and the capacity a prompt with
    /// no stored state would leave. `None` during a request, before the
    /// governor opens, and while no prompt step has been measured, since
    /// room nobody measured is not room.
    pub(super) fn idle_kept_weight_room_bytes(&self) -> Option<u64> {
        let memory_governor = self.governed_memory.governor.borrow();
        let memory_governor = memory_governor.as_ref()?;
        if !matches!(memory_governor.phase(), LedgerPhase::Idle) {
            return None;
        }
        let probed_mlx_working_buffers = self.probed_prompt_layer_mlx_working_buffers.get()?;
        let capacity_while_reading = memory_governor.capacity_while(LedgerPhase::ReadingPrompt {
            span_token_count: probed_mlx_working_buffers.span_token_count,
        });
        Some(
            memory_governor
                .kept_weight_capacity()
                .bytes()
                .min(capacity_while_reading.bytes()),
        )
    }

    /// The one place the astro expert cache is told how much it may keep. The
    /// astro expert cache gives back, in its own order, whatever no longer fits.
    ///
    /// A dense model's kept layers are its kept weights: trailing layers
    /// are given back until what is kept fits.
    pub(super) fn apply_kept_weight_capacity(&self, capacity: KeptWeightCapacity) {
        self.give_back_kept_dense_layers_beyond(capacity);
        let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() else {
            return;
        };
        let released_expert_bytes = astro_expert_cache
            .borrow_mut()
            .set_capacity(capacity.bytes())
            .released_payload_bytes();
        if released_expert_bytes > 0 {
            tracing::info!(
                kept_weight_capacity_bytes = capacity.bytes(),
                released_expert_bytes,
                "the astro expert cache gave back what its new capacity no longer holds"
            );
        }
    }

    /// The capacity the astro expert cache has now.
    pub fn kept_expert_capacity_bytes(&self) -> u64 {
        self.astro_expert_cache
            .as_ref()
            .map_or(0, |astro_expert_cache| {
                astro_expert_cache.borrow().capacity_bytes()
            })
    }

    /// A request starts reading a prompt of `prompt_token_count` tokens in
    /// spans of at most `span_token_count`. The stored state the whole
    /// prompt will occupy is set aside now, so the capacity does not move
    /// again while the prompt is read.
    pub fn govern_request_start(&self, prompt_token_count: u64, span_token_count: u32) {
        // The peak kept for the last request's prompt would otherwise be
        // reported as this request's until its first forward begins.
        self.begin_measuring_a_new_forward();
        let prompt_context_bytes = prompt_token_count.saturating_mul(
            self.memory_facts()
                .model_geometry()
                .sequence_state_bytes_per_token,
        );
        let capacity = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            self.follow_installed_limit(memory_governor);
            memory_governor.start_request(
                LedgerPhase::ReadingPrompt { span_token_count },
                prompt_context_bytes,
            )
        };
        self.apply_kept_weight_capacity(capacity);
    }

    /// The prompt is read and the reply begins: one token per step needs far
    /// fewer MLX working buffers, so the capacity rises.
    pub fn govern_reply_start(&self) {
        let capacity = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            memory_governor.enter_phase(LedgerPhase::WritingReply)
        };
        self.apply_kept_weight_capacity(capacity);
    }

    /// A prompt begins to be read one layer at a time: samples what is held
    /// so each layer can be measured as its own step. Only an expert model
    /// is measured this way; it waits for each layer anyway.
    pub fn begin_prompt_layer_measurement(&self) {
        let governed_memory = &self.governed_memory;
        governed_memory.prompt_layer_boundary_active_bytes.set(None);
        if self.astro_expert_cache.is_none() || governed_memory.governor.borrow().is_none() {
            return;
        }
        let Ok(memory_snapshot) = self.runtime.memory_snapshot() else {
            return;
        };
        if self.runtime.reset_peak_memory().is_err() {
            return;
        }
        governed_memory
            .prompt_layer_boundary_active_bytes
            .set(Some(memory_snapshot.active_memory_bytes() as u64));
    }

    /// What MLX holds now, with the peak of the forward in flight.
    ///
    /// Measuring a prompt one layer at a time resets MLX's peak counter at
    /// every layer, so a plain snapshot afterwards shows only the last
    /// layer's peak. Every sample the engine reports goes through here, so
    /// what it reports is the forward's peak.
    pub fn memory_snapshot_with_forward_peak(
        &self,
    ) -> Result<runtime_integration::MlxMemorySnapshot, runtime_integration::MlxRuntimeError> {
        Ok(self
            .runtime
            .memory_snapshot()?
            .with_peak_at_least(self.governed_memory.largest_prompt_layer_peak_bytes.get()))
    }

    /// A forward begins: the peak kept from the last one's layers is spent,
    /// and so is the mark that it was measured layer by layer. A prompt that
    /// failed after its first layer otherwise leaves the mark set, and the
    /// next forward, measured as one step, would be skipped.
    pub fn begin_measuring_a_new_forward(&self) {
        self.governed_memory.largest_prompt_layer_peak_bytes.set(0);
        self.governed_memory
            .measured_forward_layer_by_layer
            .set(false);
    }

    /// The prompt has stopped being read, finished or not.
    pub fn end_prompt_layer_measurement(&self) {
        self.governed_memory
            .prompt_layer_boundary_active_bytes
            .set(None);
    }

    /// One layer has finished every span of the prompt: measures it as one
    /// step, compares the ledger with MLX at this boundary, and applies the
    /// capacity the governor answers with. `pages_in_hand` is what the layer
    /// had in flight and what is still being read ahead.
    pub fn govern_completed_prompt_layer(
        &self,
        pages_in_hand: super::held_prompt_expert_layer::ExpertPagesInHandBytes,
        context_bytes: u64,
    ) {
        let governed_memory = &self.governed_memory;
        let Some(active_bytes_before_layer) =
            governed_memory.prompt_layer_boundary_active_bytes.get()
        else {
            return;
        };
        let Ok(memory_snapshot) = self.runtime.memory_snapshot() else {
            return;
        };
        let active_bytes_after_layer = memory_snapshot.active_memory_bytes() as u64;
        let step_verdict = {
            let mut memory_governor = governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            let ledger_phase = memory_governor.phase();
            if !matches!(ledger_phase, LedgerPhase::ReadingPrompt { .. }) {
                return;
            }
            // The prompt's stored state was set aside from a per-token
            // figure when the request began. Attention state is allocated a
            // growth step at a time, so what is really held can run a
            // little ahead of that; the account follows what is held.
            let context_growth_bytes =
                context_bytes.saturating_sub(memory_governor.accounts().context_bytes);
            if context_growth_bytes > 0 {
                let _ = memory_governor.grow_context(context_growth_bytes);
            }
            let measured_step = MeasuredStep {
                phase: ledger_phase,
                held_memory: HeldMemory {
                    context_bytes,
                    kept_weight_bytes: self.reclaimable_retained_payload_bytes(),
                    stream_slot_bytes: pages_in_hand.read_ahead_bytes,
                },
                in_flight_weight_bytes: pages_in_hand
                    .held_bytes
                    .saturating_add(pages_in_hand.read_ahead_bytes),
                active_bytes_before_step: active_bytes_before_layer,
                active_bytes_after_step: active_bytes_after_layer,
                peak_bytes_during_step: memory_snapshot.peak_memory_bytes() as u64,
            };
            // Taken only now: a layer that is not measured must leave the
            // redo waiting for the step that settles it.
            match governed_memory.pending_redo_attempt.borrow_mut().take() {
                Some(redo_attempt) => {
                    memory_governor.complete_redone_step(redo_attempt, measured_step)
                }
                None => memory_governor.complete_step(measured_step),
            }
        };
        governed_memory
            .prompt_layer_boundary_active_bytes
            .set(Some(active_bytes_after_layer));
        governed_memory.measured_forward_layer_by_layer.set(true);
        governed_memory.largest_prompt_layer_peak_bytes.set(
            governed_memory
                .largest_prompt_layer_peak_bytes
                .get()
                .max(memory_snapshot.peak_memory_bytes()),
        );
        // A failed reset only makes the next layer's peak include this one's.
        let _ = self.runtime.reset_peak_memory();
        self.apply_kept_weight_capacity(step_verdict.capacity);
    }

    /// One forward completed and was measured. A forward that follows a
    /// redo settles it. A prompt already measured one layer at a time is
    /// not measured again as one large step.
    pub fn govern_completed_forward(&self, forward_readings: CompletedForwardMemoryReadings) {
        let Some(ledger_phase) = ledger_phase_for(
            forward_readings.memory_phase,
            forward_readings.forward_token_count,
        ) else {
            return;
        };
        if self
            .governed_memory
            .measured_forward_layer_by_layer
            .replace(false)
        {
            self.give_back_publication_workspace_after_forward();
            return;
        }
        let measured_step = MeasuredStep {
            phase: ledger_phase,
            held_memory: HeldMemory {
                context_bytes: forward_readings.live_context_state_payload_bytes,
                kept_weight_bytes: self.reclaimable_retained_payload_bytes(),
                // A forward's streamed weights are dropped when it completes.
                stream_slot_bytes: 0,
            },
            in_flight_weight_bytes: self
                .memory_ledger_shape()
                .stream_slot_bytes_for(ledger_phase),
            active_bytes_before_step: forward_readings.active_bytes_before_forward,
            active_bytes_after_step: forward_readings.active_bytes_after_forward,
            peak_bytes_during_step: forward_readings.peak_bytes_during_forward,
        };
        let pending_redo_attempt = self
            .governed_memory
            .pending_redo_attempt
            .borrow_mut()
            .take();
        let step_verdict = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            self.give_back_publication_workspace(memory_governor);
            let context_growth_bytes = forward_readings
                .live_context_state_payload_bytes
                .saturating_sub(memory_governor.accounts().context_bytes);
            if context_growth_bytes > 0 {
                let _ = memory_governor.grow_context(context_growth_bytes);
            }
            match pending_redo_attempt {
                Some(redo_attempt) => {
                    memory_governor.complete_redone_step(redo_attempt, measured_step)
                }
                None => memory_governor.complete_step(measured_step),
            }
        };
        log_ledger_reading(ledger_phase, &step_verdict.reading);
        self.apply_kept_weight_capacity(step_verdict.capacity);
    }

    /// A forward hit the limit. Leases room for another attempt, at least
    /// `known_shortfall_bytes` when the failure said how far over it went,
    /// and has the astro expert cache give back just the experts that room displaces.
    /// `false` once the attempts are spent: the limit cannot hold the
    /// forward.
    pub fn govern_redo_after_memory_failure(
        &self,
        memory_phase: MemoryPhase,
        known_shortfall_bytes: u64,
    ) -> bool {
        let Some(ledger_phase) = self.current_ledger_phase(memory_phase) else {
            return false;
        };
        let failed_attempt = self
            .governed_memory
            .pending_redo_attempt
            .borrow_mut()
            .take();
        let redo = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return false;
            };
            memory_governor.begin_redo(ledger_phase, failed_attempt, known_shortfall_bytes)
        };
        let Some((redo_attempt, capacity)) = redo else {
            return false;
        };
        tracing::warn!(
            attempt_number = redo_attempt.attempt_number(),
            leased_room_bytes = redo_attempt.headroom_bytes(),
            kept_weight_capacity_bytes = capacity.bytes(),
            "a forward hit the memory limit; redoing it with more room"
        );
        *self.governed_memory.pending_redo_attempt.borrow_mut() = Some(redo_attempt);
        self.apply_kept_weight_capacity(capacity);
        true
    }

    /// The phase the governor is in, when it matches what the engine says
    /// it was doing.
    fn current_ledger_phase(&self, memory_phase: MemoryPhase) -> Option<LedgerPhase> {
        let memory_governor = self.governed_memory.governor.borrow();
        let ledger_phase = memory_governor.as_ref()?.phase();
        match (memory_phase, ledger_phase) {
            (MemoryPhase::Prefill, LedgerPhase::ReadingPrompt { .. })
            | (MemoryPhase::Decode, LedgerPhase::WritingReply) => Some(ledger_phase),
            _ => None,
        }
    }

    /// The request is over. A redo left unsettled gives its room back; the
    /// streaming engine keeps no stored state between requests.
    pub fn govern_request_end(&self, request_id: u64) {
        let abandoned_redo_attempt = self
            .governed_memory
            .pending_redo_attempt
            .borrow_mut()
            .take();
        let capacity = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            if let Some(redo_attempt) = abandoned_redo_attempt {
                let _ = memory_governor.abandon_redo(redo_attempt);
            }
            self.give_back_publication_workspace(memory_governor);
            self.give_back_restore_workspace_lease(memory_governor);
            if let Some(ledger_reading) = memory_governor.latest_reading() {
                tracing::info!(
                    request_id,
                    observed_step_count = ledger_reading.observed_step_count,
                    largest_unaccounted_bytes = ledger_reading.largest_unaccounted_bytes,
                    largest_overcounted_bytes = ledger_reading.largest_overcounted_bytes,
                    lowest_kept_share_while_reading_permille =
                        ?ledger_reading.lowest_kept_share_while_reading_permille,
                    overflow_count = ledger_reading.overflow_count,
                    redo_attempt_count = ledger_reading.redo_attempt_count,
                    "the memory governor closed one request"
                );
            }
            memory_governor.finish_request(0)
        };
        self.apply_kept_weight_capacity(capacity);
    }

    /// The limit moved while the model is loaded.
    pub fn govern_limit_change(&self) {
        let capacity = {
            let mut memory_governor = self.governed_memory.governor.borrow_mut();
            let Some(memory_governor) = memory_governor.as_mut() else {
                return;
            };
            self.follow_installed_limit(memory_governor);
            memory_governor.kept_weight_capacity()
        };
        self.apply_kept_weight_capacity(capacity);
    }

    /// Tells the governor the limit MLX has installed. A zero limit is
    /// refused and the governor keeps the one it had.
    fn follow_installed_limit(&self, memory_governor: &mut MemoryGovernor) {
        let installed_limit_bytes = self.memory_facts().mlx_active_memory_ceiling_bytes();
        if installed_limit_bytes != memory_governor.accounts().limit_bytes {
            let _ = memory_governor.change_limit(installed_limit_bytes);
        }
    }

    /// The governor's reading of the most recent completed forward.
    /// The governor's accounts in the shape the status breakdown composes
    /// against, so what the status page shows as reserved is what is really
    /// set aside.
    ///
    /// A dense model's kept layers count as
    /// model core in the breakdown, so they move from the kept-weight
    /// budget to the core here.
    pub fn memory_reserve_snapshot_from_governor(
        &self,
        held_expert_payload_bytes: u64,
    ) -> Option<crate::memory::MemoryReserveSnapshot> {
        let memory_governor = self.governed_memory.governor.borrow();
        let accounts = memory_governor.as_ref()?.accounts();
        let kept_core_layer_bytes = self
            .reclaimable_retained_payload_bytes()
            .saturating_sub(held_expert_payload_bytes);
        let mut budget_snapshot =
            crate::memory::MemoryReserveSnapshot::from_ledger_accounts(&accounts);
        budget_snapshot.model_core_payload_bytes = budget_snapshot
            .model_core_payload_bytes
            .saturating_add(kept_core_layer_bytes);
        budget_snapshot.retained_expert_budget_bytes = budget_snapshot
            .retained_expert_budget_bytes
            .saturating_sub(kept_core_layer_bytes);
        Some(budget_snapshot)
    }

    /// The MLX working buffers one prompt step needs, as far as it has been
    /// measured: the larger of the probe at load and the largest prompt step
    /// of the latest request. `None` until either exists.
    pub fn measured_prompt_mlx_working_buffer_bytes(&self) -> Option<u64> {
        let probed_mlx_working_buffer_bytes = self
            .probed_prompt_layer_mlx_working_buffers
            .get()
            .map(|probed_mlx_working_buffers| probed_mlx_working_buffers.mlx_working_buffer_bytes);
        let largest_measured_step_bytes = self
            .latest_memory_ledger_reading()
            .map(|ledger_reading| ledger_reading.largest_reading_step_peak_beyond_held_bytes)
            .filter(|largest_step_bytes| *largest_step_bytes > 0);
        larger_of_what_was_measured(probed_mlx_working_buffer_bytes, largest_measured_step_bytes)
    }

    /// Steps that hit the limit and redo attempts made since the model
    /// loaded, in that order.
    pub fn memory_redo_counts(&self) -> (u64, u64) {
        self.governed_memory
            .governor
            .borrow()
            .as_ref()
            .map_or((0, 0), |memory_governor| {
                (
                    memory_governor.overflow_count(),
                    memory_governor.redo_attempt_count(),
                )
            })
    }

    pub fn latest_memory_ledger_reading(&self) -> Option<LedgerReading> {
        self.governed_memory
            .governor
            .borrow()
            .as_ref()
            .and_then(MemoryGovernor::latest_reading)
    }
}

/// The larger of two measurements of the same thing, either of which may
/// not exist yet.
const fn larger_of_what_was_measured(
    first_measurement_bytes: Option<u64>,
    second_measurement_bytes: Option<u64>,
) -> Option<u64> {
    match (first_measurement_bytes, second_measurement_bytes) {
        (Some(first_bytes), Some(second_bytes)) => Some(if first_bytes > second_bytes {
            first_bytes
        } else {
            second_bytes
        }),
        (Some(only_bytes), None) | (None, Some(only_bytes)) => Some(only_bytes),
        (None, None) => None,
    }
}

/// A sparse model's shape: its experts are what may be kept or streamed.
/// Reading a prompt holds whole layers in flight; writing a reply holds only
/// the experts one token chose.
fn sparse_model_memory_shape(model_geometry: ModelMemoryGeometry) -> ModelMemoryShape {
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
/// between the two and is not a forward the governor measures.
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
        mlx_working_buffer_bytes = ledger_reading.accounts.mlx_working_buffer_bytes,
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
        "the memory governor measured one forward"
    );
}

#[cfg(test)]
#[path = "memory_governance_tests.rs"]
mod tests;
