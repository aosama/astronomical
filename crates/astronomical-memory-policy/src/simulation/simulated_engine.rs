//! Reads prompts one layer at a time and writes replies, telling the
//! governor what happens and applying the capacity it answers with.
//!
//! This is the engine the redesign describes, reduced to its memory
//! decisions. It keeps whole layers in a fixed order of preference and
//! gives back the least preferred first; the real kept-weight policy
//! replaces that order without changing what the ledger is asked.

use super::simulated_model::SimulatedModel;
use crate::ledger::{
    HeldMemory, KeptWeightCapacity, LeaseId, LeasePurpose, LedgerAccounts, LedgerPhase,
    MeasuredStep, MemoryGovernor, RedoAttempt,
};

/// Reply tokens after which the stored state grows by one block.
const REPLY_CONTEXT_BLOCK_TOKEN_COUNT: u32 = 256;

/// What happened, for a scenario to assert on.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct EngineJournal {
    pub(crate) overflow_count: u64,
    pub(crate) redone_layer_step_count: u64,
    pub(crate) eviction_event_count: u64,
    pub(crate) evicted_weight_bytes: u64,
    pub(crate) largest_single_eviction_bytes: u64,
    pub(crate) storage_read_bytes: u64,
    pub(crate) largest_unaccounted_bytes: u64,
    pub(crate) peak_active_bytes: u64,
    /// What was kept when the most recent overflow happened, and how much
    /// of it redoing that step gave back.
    pub(crate) kept_weight_bytes_at_last_overflow: u64,
    pub(crate) evicted_for_redo_bytes: u64,
    /// A step could not fit even with nothing kept. The ledger never
    /// refuses; this is MLX running out for real.
    pub(crate) ran_out_of_memory: bool,
}

/// Which layers the engine prefers to keep. The ledger does not care; the
/// order decides where in a prompt the tightest step falls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KeptLayerOrder {
    /// Keeps the first layers, so a prompt meets its streamed layers last.
    LowestLayersFirst,
    /// Keeps the last layers, so a prompt's very first step streams.
    HighestLayersFirst,
}

pub(crate) struct SimulatedEngine {
    model: SimulatedModel,
    governor: MemoryGovernor,
    kept_layer_order: KeptLayerOrder,
    layer_is_kept: Vec<bool>,
    held_context_bytes: u64,
    reserved_context_bytes: u64,
    /// Memory behind leases that are real allocations, such as a vision
    /// tower. A redo's headroom is room, not an allocation, and is absent.
    leased_allocation_bytes: u64,
    /// Memory a scenario allocates without telling the ledger.
    undeclared_bytes: u64,
    journal: EngineJournal,
}

impl SimulatedEngine {
    pub(crate) fn load(model: SimulatedModel, limit_bytes: u64) -> Self {
        Self::load_keeping(model, limit_bytes, KeptLayerOrder::LowestLayersFirst)
    }

    pub(crate) fn load_keeping(
        model: SimulatedModel,
        limit_bytes: u64,
        kept_layer_order: KeptLayerOrder,
    ) -> Self {
        let governor = MemoryGovernor::new(limit_bytes, model.memory_shape())
            .expect("a scenario's limit is never zero");
        let layer_count = model.layer_weight_bytes.len();
        Self {
            model,
            governor,
            kept_layer_order,
            layer_is_kept: vec![false; layer_count],
            held_context_bytes: 0,
            reserved_context_bytes: 0,
            leased_allocation_bytes: 0,
            undeclared_bytes: 0,
            journal: EngineJournal::default(),
        }
    }

    /// The probe a real engine runs at load: one layer step of each kind,
    /// measured. `reported_percent` below 100 misjudges the measurement on
    /// purpose, the way a test hook would.
    pub(crate) fn probe_working_memory(&mut self, span_token_count: u32, reported_percent: u64) {
        for phase in [
            LedgerPhase::ReadingPrompt { span_token_count },
            LedgerPhase::WritingReply,
        ] {
            let reported_bytes =
                self.model.true_working_memory_bytes(phase) * reported_percent / 100;
            let capacity = self.governor.record_probe(phase, reported_bytes);
            self.trim_kept_layers_to(capacity);
        }
    }

    pub(crate) fn accounts(&self) -> LedgerAccounts {
        self.governor.accounts()
    }

    pub(crate) const fn journal(&self) -> EngineJournal {
        self.journal
    }

    pub(crate) fn kept_weight_bytes(&self) -> u64 {
        let mut kept_weight_bytes = 0_u64;
        for (layer_index, layer_weight_bytes) in self.model.layer_weight_bytes.iter().enumerate() {
            if self.layer_is_kept[layer_index] {
                kept_weight_bytes += layer_weight_bytes;
            }
        }
        kept_weight_bytes
    }

    pub(crate) fn hold_undeclared_bytes(&mut self, undeclared_bytes: u64) {
        self.undeclared_bytes = undeclared_bytes;
    }

    pub(crate) fn change_limit(&mut self, limit_bytes: u64) {
        let capacity = self
            .governor
            .change_limit(limit_bytes)
            .expect("a scenario's limit is never zero");
        self.trim_kept_layers_to(capacity);
    }

    /// Declares a real allocation outside the normal pattern.
    pub(crate) fn grant_lease(&mut self, purpose: LeasePurpose, lease_bytes: u64) -> LeaseId {
        let (lease_id, capacity) = self.governor.grant_lease(purpose, lease_bytes);
        self.trim_kept_layers_to(capacity);
        self.leased_allocation_bytes += lease_bytes;
        lease_id
    }

    pub(crate) fn give_back_lease(&mut self, lease_id: LeaseId, lease_bytes: u64) {
        let _ = self
            .governor
            .give_back_lease(lease_id)
            .expect("a scenario gives back only leases it holds");
        self.leased_allocation_bytes -= lease_bytes;
    }

    /// One whole request: prompt, reply, end.
    pub(crate) fn serve(
        &mut self,
        prompt_token_count: u32,
        reply_token_count: u32,
        span_token_count: u32,
    ) {
        self.start_request(prompt_token_count);
        self.read_prompt(prompt_token_count, span_token_count);
        self.write_reply(reply_token_count);
        self.finish_request();
    }

    pub(crate) fn start_request(&mut self, prompt_token_count: u32) {
        self.reserved_context_bytes =
            u64::from(prompt_token_count) * self.model.context_bytes_per_token;
        self.held_context_bytes = 0;
        // The phase follows in `read_prompt`, once the span size is known.
        let capacity = self
            .governor
            .start_request(LedgerPhase::Idle, self.reserved_context_bytes);
        self.trim_kept_layers_to(capacity);
    }

    pub(crate) fn read_prompt(&mut self, prompt_token_count: u32, span_token_count: u32) {
        let phase = LedgerPhase::ReadingPrompt {
            span_token_count: span_token_count.min(prompt_token_count),
        };
        let capacity = self.governor.enter_phase(phase);
        self.trim_kept_layers_to(capacity);
        let span_count = prompt_token_count.div_ceil(span_token_count);
        let layer_count = self.model.layer_weight_bytes.len();
        for layer_index in 0..layer_count {
            self.read_prompt_through_layer(layer_index, phase, span_count);
            if self.journal.ran_out_of_memory {
                return;
            }
            self.held_context_bytes =
                self.reserved_context_bytes * (layer_index as u64 + 1) / layer_count as u64;
        }
    }

    pub(crate) fn write_reply(&mut self, reply_token_count: u32) {
        if self.journal.ran_out_of_memory {
            return;
        }
        let capacity = self.governor.enter_phase(LedgerPhase::WritingReply);
        self.trim_kept_layers_to(capacity);
        self.fill_kept_layers();
        let mut written_token_count = 0;
        while written_token_count < reply_token_count {
            let block_token_count =
                REPLY_CONTEXT_BLOCK_TOKEN_COUNT.min(reply_token_count - written_token_count);
            let block_context_bytes =
                u64::from(block_token_count) * self.model.context_bytes_per_token;
            self.reserved_context_bytes += block_context_bytes;
            self.held_context_bytes += block_context_bytes;
            let capacity = self.governor.grow_context(block_context_bytes);
            self.trim_kept_layers_to(capacity);
            let in_flight_bytes = self.writing_reply_in_flight_bytes();
            if !self.run_layer_step(LedgerPhase::WritingReply, in_flight_bytes) {
                return;
            }
            written_token_count += block_token_count;
        }
    }

    pub(crate) fn finish_request(&mut self) {
        self.reserved_context_bytes = 0;
        self.held_context_bytes = 0;
        let capacity = self.governor.finish_request(0);
        self.trim_kept_layers_to(capacity);
    }

    /// Runs every span of the prompt through one layer. A layer that is not
    /// kept is read once for all of them, and the next unkept layer is read
    /// ahead beside it; afterwards it is kept if there is capacity.
    fn read_prompt_through_layer(
        &mut self,
        layer_index: usize,
        phase: LedgerPhase,
        span_count: u32,
    ) {
        let layer_weight_bytes = self.model.layer_weight_bytes[layer_index];
        let layer_was_kept = self.layer_is_kept[layer_index];
        let mut in_flight_bytes = self.read_ahead_bytes_after(layer_index);
        if !layer_was_kept {
            self.journal.storage_read_bytes += layer_weight_bytes;
            in_flight_bytes += layer_weight_bytes;
        }
        for _ in 0..span_count {
            if !self.run_layer_step(phase, in_flight_bytes) {
                return;
            }
        }
        let capacity_bytes = self.governor.accounts().kept_weight_capacity_bytes;
        if !layer_was_kept && self.kept_weight_bytes() + layer_weight_bytes <= capacity_bytes {
            self.layer_is_kept[layer_index] = true;
        }
    }

    /// The next layer's bytes when it has to be read, since its read
    /// overlaps this layer's work.
    fn read_ahead_bytes_after(&self, layer_index: usize) -> u64 {
        match self.layer_is_kept.get(layer_index + 1) {
            Some(false) => self.model.layer_weight_bytes[layer_index + 1],
            _ => 0,
        }
    }

    fn writing_reply_in_flight_bytes(&self) -> u64 {
        if self.layer_is_kept.contains(&false) {
            return self.model.writing_reply_stream_slot_bytes;
        }
        0
    }

    /// One layer step. Returns `false` only when the step cannot fit even
    /// with nothing kept.
    fn run_layer_step(&mut self, phase: LedgerPhase, in_flight_bytes: u64) -> bool {
        if self.step_fits(phase, in_flight_bytes) {
            let measured_step = self.measured_step(phase, in_flight_bytes);
            let verdict = self.governor.complete_step(measured_step);
            self.note_verdict(
                measured_step,
                verdict.reading.reconciliation.unaccounted_bytes,
            );
            self.trim_kept_layers_to(verdict.capacity);
            return true;
        }
        self.redo_layer_step_with_more_room(phase, in_flight_bytes)
    }

    fn step_fits(&self, phase: LedgerPhase, in_flight_bytes: u64) -> bool {
        self.active_bytes(in_flight_bytes) + self.model.true_working_memory_bytes(phase)
            <= self.allowed_active_bytes()
    }

    /// What a real engine would measure around this step.
    fn measured_step(&self, phase: LedgerPhase, in_flight_bytes: u64) -> MeasuredStep {
        let held_bytes = self.active_bytes(0);
        MeasuredStep {
            phase,
            held_memory: HeldMemory {
                context_bytes: self.held_context_bytes,
                kept_weight_bytes: self.kept_weight_bytes(),
                stream_slot_bytes: 0,
            },
            in_flight_weight_bytes: in_flight_bytes,
            active_bytes_before_step: held_bytes,
            active_bytes_after_step: held_bytes,
            peak_bytes_during_step: held_bytes
                + in_flight_bytes
                + self.model.true_working_memory_bytes(phase),
        }
    }

    fn note_verdict(&mut self, measured_step: MeasuredStep, unaccounted_bytes: u64) {
        self.journal.peak_active_bytes = self
            .journal
            .peak_active_bytes
            .max(measured_step.peak_bytes_during_step);
        self.journal.largest_unaccounted_bytes = self
            .journal
            .largest_unaccounted_bytes
            .max(unaccounted_bytes);
    }

    /// The step hit the limit. The governor leases room for another
    /// attempt, the engine gives back just the kept weight that room
    /// displaces and redoes the one step, and a redone step that fits is
    /// measured, so the true number replaces the guess.
    fn redo_layer_step_with_more_room(&mut self, phase: LedgerPhase, in_flight_bytes: u64) -> bool {
        self.journal.overflow_count += 1;
        self.journal.kept_weight_bytes_at_last_overflow = self.kept_weight_bytes();
        let evicted_before_redo_bytes = self.journal.evicted_weight_bytes;
        let mut failed_attempt: Option<RedoAttempt> = None;
        let mut redone_step_fits = false;
        while let Some((redo_attempt, capacity)) =
            self.governor.begin_redo(phase, failed_attempt.take(), 0)
        {
            self.trim_kept_layers_to(capacity);
            self.journal.redone_layer_step_count += 1;
            if !self.step_fits(phase, in_flight_bytes) {
                failed_attempt = Some(redo_attempt);
                continue;
            }
            let measured_step = self.measured_step(phase, in_flight_bytes);
            let verdict = self
                .governor
                .complete_redone_step(redo_attempt, measured_step);
            self.note_verdict(
                measured_step,
                verdict.reading.reconciliation.unaccounted_bytes,
            );
            self.trim_kept_layers_to(verdict.capacity);
            redone_step_fits = true;
            break;
        }
        self.journal.evicted_for_redo_bytes +=
            self.journal.evicted_weight_bytes - evicted_before_redo_bytes;
        self.journal.ran_out_of_memory = !redone_step_fits;
        redone_step_fits
    }

    /// Layer indexes from the most preferred to keep to the least.
    fn layer_indexes_by_keeping_preference(&self) -> Vec<usize> {
        let mut layer_indexes: Vec<usize> = (0..self.layer_is_kept.len()).collect();
        if self.kept_layer_order == KeptLayerOrder::HighestLayersFirst {
            layer_indexes.reverse();
        }
        layer_indexes
    }

    /// Keeps more layers, most preferred first, while there is capacity.
    /// This is what entering the reply phase does with the room it gains.
    fn fill_kept_layers(&mut self) {
        let capacity_bytes = self.governor.accounts().kept_weight_capacity_bytes;
        let mut kept_weight_bytes = self.kept_weight_bytes();
        for layer_index in self.layer_indexes_by_keeping_preference() {
            let layer_weight_bytes = self.model.layer_weight_bytes[layer_index];
            if self.layer_is_kept[layer_index]
                || kept_weight_bytes + layer_weight_bytes > capacity_bytes
            {
                continue;
            }
            self.layer_is_kept[layer_index] = true;
            kept_weight_bytes += layer_weight_bytes;
            self.journal.storage_read_bytes += layer_weight_bytes;
        }
    }

    /// Gives back kept layers, least preferred first, until what is kept
    /// fits the capacity. The only place anything is evicted.
    fn trim_kept_layers_to(&mut self, capacity: KeptWeightCapacity) {
        let mut kept_weight_bytes = self.kept_weight_bytes();
        let mut evicted_now_bytes = 0;
        let mut layer_indexes_least_preferred_first = self.layer_indexes_by_keeping_preference();
        layer_indexes_least_preferred_first.reverse();
        for layer_index in layer_indexes_least_preferred_first {
            if kept_weight_bytes <= capacity.bytes() {
                break;
            }
            if self.layer_is_kept[layer_index] {
                self.layer_is_kept[layer_index] = false;
                kept_weight_bytes -= self.model.layer_weight_bytes[layer_index];
                evicted_now_bytes += self.model.layer_weight_bytes[layer_index];
            }
        }
        if evicted_now_bytes > 0 {
            self.journal.eviction_event_count += 1;
            self.journal.evicted_weight_bytes += evicted_now_bytes;
            self.journal.largest_single_eviction_bytes = self
                .journal
                .largest_single_eviction_bytes
                .max(evicted_now_bytes);
        }
    }

    /// What MLX would report: everything really in memory right now.
    fn active_bytes(&self, in_flight_bytes: u64) -> u64 {
        self.model.core_weight_bytes
            + self.held_context_bytes
            + self.kept_weight_bytes()
            + in_flight_bytes
            + self.leased_allocation_bytes
            + self.undeclared_bytes
    }

    /// MLX enforces the limit plus one percent.
    fn allowed_active_bytes(&self) -> u64 {
        let limit_bytes = self.governor.accounts().limit_bytes;
        limit_bytes + limit_bytes / 100
    }
}
