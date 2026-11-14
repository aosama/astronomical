//! The ledger fed with what an engine does, while it decides nothing.
//!
//! Before the ledger takes control of anything it watches: the engine keeps
//! deciding the way it always has, and reports each completed step here. The
//! ledger then says what it expected to be in memory, what MLX reported, and
//! how the weights the engine kept compare with the capacity the ledger
//! would have allowed. Whatever no account can explain is the list of code
//! paths that allocate without declaring it (issue #2221, phase 1).

use super::accounts::LedgerAccounts;
use super::memory_ledger::{MemoryLedger, MemoryLedgerError};
use super::model_shape::ModelMemoryShape;
use super::phase::LedgerPhase;
use super::reconciliation::{HeldMemory, LedgerReconciliation};

/// One completed step, as the engine measured it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservedStep {
    pub phase: LedgerPhase,
    /// The MLX limit in force during the step.
    pub limit_bytes: u64,
    /// The stored state the request has to hold: what it holds already, or
    /// what its whole prompt will hold, whichever is larger.
    pub request_context_bytes: u64,
    /// What the engine's own owners say is in memory after the step.
    pub held_memory: HeldMemory,
    pub active_bytes_before_step: u64,
    pub active_bytes_after_step: u64,
    pub peak_bytes_during_step: u64,
}

/// The ledger's view after one observed step, with the worst readings of
/// the request so far.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LedgerReading {
    pub accounts: LedgerAccounts,
    pub held_memory: HeldMemory,
    pub reconciliation: LedgerReconciliation,
    /// How far the step's peak rose above what was held around it.
    pub step_peak_beyond_held_bytes: u64,
    pub observed_step_count: u64,
    pub largest_unaccounted_bytes: u64,
    pub largest_overcounted_bytes: u64,
    pub largest_kept_over_capacity_bytes: u64,
    /// The largest rise of a prompt step's peak above what was held around
    /// it, and the same for a reply step: what one step needs for a moment.
    pub largest_reading_step_peak_beyond_held_bytes: u64,
    pub largest_writing_step_peak_beyond_held_bytes: u64,
    /// The emptiest the kept weights were against the ledger's capacity
    /// while a prompt was read, in thousandths; `None` until a prompt step
    /// with nonzero capacity has been observed.
    pub lowest_kept_share_while_reading_permille: Option<u32>,
}

/// The worst readings since the request began.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct WorstReadings {
    observed_step_count: u64,
    largest_unaccounted_bytes: u64,
    largest_overcounted_bytes: u64,
    largest_kept_over_capacity_bytes: u64,
    largest_reading_step_peak_beyond_held_bytes: u64,
    largest_writing_step_peak_beyond_held_bytes: u64,
    lowest_kept_share_while_reading_permille: Option<u32>,
}

/// A ledger that watches one engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerObserver {
    ledger: MemoryLedger,
    limit_bytes: u64,
    request_is_open: bool,
    request_context_bytes: u64,
    worst_readings: WorstReadings,
    latest_reading: Option<LedgerReading>,
}

impl LedgerObserver {
    pub fn new(limit_bytes: u64, model_shape: ModelMemoryShape) -> Result<Self, MemoryLedgerError> {
        Ok(Self {
            ledger: MemoryLedger::new(limit_bytes, model_shape)?,
            limit_bytes,
            request_is_open: false,
            request_context_bytes: 0,
            worst_readings: WorstReadings::default(),
            latest_reading: None,
        })
    }

    /// The reading of the most recent observed step.
    #[must_use]
    pub const fn latest_reading(&self) -> Option<LedgerReading> {
        self.latest_reading
    }

    /// Feeds one completed step to the ledger and returns its view.
    pub fn observe_step(&mut self, observed_step: ObservedStep) -> LedgerReading {
        self.follow_limit(observed_step.limit_bytes);
        self.follow_request_context(observed_step.request_context_bytes);
        self.ledger.enter_phase(observed_step.phase);
        let step_peak_beyond_held_bytes = observed_step.peak_bytes_during_step.saturating_sub(
            observed_step
                .active_bytes_before_step
                .max(observed_step.active_bytes_after_step),
        );
        // The engine being watched holds the weights it is reading inside
        // its peak, and the ledger names those as slots, so they come off
        // before the rest is recorded as one step's working memory.
        let stream_slot_bytes = self
            .ledger
            .model_shape()
            .stream_slot_bytes_for(observed_step.phase);
        let accounts = self.ledger.record_working_memory(
            observed_step.phase,
            step_peak_beyond_held_bytes.saturating_sub(stream_slot_bytes),
        );
        let reconciliation = self.ledger.reconcile(
            observed_step.held_memory,
            observed_step.active_bytes_after_step,
        );
        self.remember_worst(observed_step, accounts, reconciliation);
        self.remember_step_peak(observed_step.phase, step_peak_beyond_held_bytes);
        let reading = LedgerReading {
            accounts,
            held_memory: observed_step.held_memory,
            reconciliation,
            step_peak_beyond_held_bytes,
            observed_step_count: self.worst_readings.observed_step_count,
            largest_unaccounted_bytes: self.worst_readings.largest_unaccounted_bytes,
            largest_overcounted_bytes: self.worst_readings.largest_overcounted_bytes,
            largest_kept_over_capacity_bytes: self.worst_readings.largest_kept_over_capacity_bytes,
            largest_reading_step_peak_beyond_held_bytes: self
                .worst_readings
                .largest_reading_step_peak_beyond_held_bytes,
            largest_writing_step_peak_beyond_held_bytes: self
                .worst_readings
                .largest_writing_step_peak_beyond_held_bytes,
            lowest_kept_share_while_reading_permille: self
                .worst_readings
                .lowest_kept_share_while_reading_permille,
        };
        self.latest_reading = Some(reading);
        reading
    }

    /// The request is over; the next observed step opens a new one.
    pub fn finish_request(&mut self, retained_context_bytes: u64) {
        self.ledger.finish_request(retained_context_bytes);
        self.request_is_open = false;
        self.request_context_bytes = retained_context_bytes;
    }

    fn follow_limit(&mut self, limit_bytes: u64) {
        if limit_bytes == self.limit_bytes {
            return;
        }
        // A zero limit is refused and the ledger keeps the one it had.
        if self.ledger.change_limit(limit_bytes).is_ok() {
            self.limit_bytes = limit_bytes;
        }
    }

    fn follow_request_context(&mut self, request_context_bytes: u64) {
        if !self.request_is_open {
            self.ledger.start_request(request_context_bytes);
            self.request_is_open = true;
            self.request_context_bytes = request_context_bytes;
            self.worst_readings = WorstReadings::default();
            return;
        }
        let context_growth_bytes = request_context_bytes.saturating_sub(self.request_context_bytes);
        if context_growth_bytes > 0 {
            self.ledger.grow_context(context_growth_bytes);
            self.request_context_bytes = request_context_bytes;
        }
    }

    fn remember_step_peak(&mut self, phase: LedgerPhase, step_peak_beyond_held_bytes: u64) {
        let largest_peak_of_phase = match phase {
            LedgerPhase::Idle => return,
            LedgerPhase::ReadingPrompt { .. } => {
                &mut self
                    .worst_readings
                    .largest_reading_step_peak_beyond_held_bytes
            }
            LedgerPhase::WritingReply => {
                &mut self
                    .worst_readings
                    .largest_writing_step_peak_beyond_held_bytes
            }
        };
        *largest_peak_of_phase = (*largest_peak_of_phase).max(step_peak_beyond_held_bytes);
    }

    fn remember_worst(
        &mut self,
        observed_step: ObservedStep,
        accounts: LedgerAccounts,
        reconciliation: LedgerReconciliation,
    ) {
        let worst_readings = &mut self.worst_readings;
        worst_readings.observed_step_count += 1;
        worst_readings.largest_unaccounted_bytes = worst_readings
            .largest_unaccounted_bytes
            .max(reconciliation.unaccounted_bytes);
        worst_readings.largest_overcounted_bytes = worst_readings
            .largest_overcounted_bytes
            .max(reconciliation.overcounted_bytes);
        worst_readings.largest_kept_over_capacity_bytes = worst_readings
            .largest_kept_over_capacity_bytes
            .max(reconciliation.kept_over_capacity_bytes);
        if !matches!(observed_step.phase, LedgerPhase::ReadingPrompt { .. }) {
            return;
        }
        let Some(kept_share_permille) = kept_share_permille(
            observed_step.held_memory.kept_weight_bytes,
            accounts.kept_weight_capacity_bytes,
        ) else {
            return;
        };
        worst_readings.lowest_kept_share_while_reading_permille = Some(
            match worst_readings.lowest_kept_share_while_reading_permille {
                Some(lowest_so_far_permille) => lowest_so_far_permille.min(kept_share_permille),
                None => kept_share_permille,
            },
        );
    }
}

/// Kept weights as thousandths of the capacity, capped at a thousand; `None`
/// when there is no capacity to fill.
fn kept_share_permille(kept_weight_bytes: u64, kept_weight_capacity_bytes: u64) -> Option<u32> {
    if kept_weight_capacity_bytes == 0 {
        return None;
    }
    let share_permille =
        u128::from(kept_weight_bytes) * 1_000 / u128::from(kept_weight_capacity_bytes);
    Some(u32::try_from(share_permille.min(1_000)).unwrap_or(1_000))
}

#[cfg(test)]
#[path = "observation_tests.rs"]
mod tests;
