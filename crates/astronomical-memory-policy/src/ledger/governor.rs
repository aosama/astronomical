//! The one owner of a loaded model's memory decisions.
//!
//! The governor holds the ledger and is the only thing that changes it. An
//! engine tells it what happens: a request starts, the phase changes, the
//! stored state grows, a step was measured, a step hit the limit, a lease is
//! needed, the limit moves. Every answer carries the kept-weight capacity
//! now in force, and the engine applies that capacity to whatever holds its
//! kept weights. Nothing else may tell those holders how much to keep, and
//! the capacity type cannot be dropped unused without a compiler warning
//! (issue #2221).

use super::accounts::LedgerAccounts;
use super::lease::{LeaseId, LeasePurpose};
use super::memory_ledger::{MemoryLedger, MemoryLedgerError};
use super::model_shape::ModelMemoryShape;
use super::phase::LedgerPhase;
use super::reconciliation::{HeldMemory, LedgerReconciliation};

/// A redone step gets its room doubled each attempt. After this many the
/// limit cannot hold the step and the request ends with MLX's own error.
pub const MAXIMUM_REDO_ATTEMPT_COUNT: u32 = 4;

/// How many streamable weight bytes may be kept from now on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "a capacity that is not applied leaves kept weights the ledger no longer has room for"]
pub struct KeptWeightCapacity {
    bytes: u64,
}

impl KeptWeightCapacity {
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// One completed step, as the engine measured it: one layer over one span
/// while a prompt is read, one token through every layer while a reply is
/// written.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasuredStep {
    pub phase: LedgerPhase,
    /// What the engine's own owners say is in memory after the step.
    pub held_memory: HeldMemory,
    /// Weights the step read from storage that were alive during it. They
    /// are the ledger's slots, not working memory.
    pub in_flight_weight_bytes: u64,
    pub active_bytes_before_step: u64,
    pub active_bytes_after_step: u64,
    pub peak_bytes_during_step: u64,
}

impl MeasuredStep {
    /// How far the step's peak rose above what was held around it.
    #[must_use]
    pub const fn peak_beyond_held_bytes(&self) -> u64 {
        let held_around_step_bytes = if self.active_bytes_before_step > self.active_bytes_after_step
        {
            self.active_bytes_before_step
        } else {
            self.active_bytes_after_step
        };
        self.peak_bytes_during_step
            .saturating_sub(held_around_step_bytes)
    }

    /// The step's working memory: its rise less the weights it had in
    /// flight.
    #[must_use]
    pub const fn working_memory_bytes(&self) -> u64 {
        self.peak_beyond_held_bytes()
            .saturating_sub(self.in_flight_weight_bytes)
    }
}

/// The governor's view after one measured step, with the worst readings of
/// the request so far. It travels out with the engine's memory telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LedgerReading {
    pub accounts: LedgerAccounts,
    pub held_memory: HeldMemory,
    /// The ledger compared with what MLX reported after the step.
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
    /// The largest working memory measured outside a request, such as one
    /// layer step probed at load; `None` when nothing was probed.
    pub probed_working_memory_bytes: Option<u64>,
    /// The emptiest the kept weights were against the capacity while a
    /// prompt was read, in thousandths; `None` until a prompt step with
    /// nonzero capacity has been measured.
    pub lowest_kept_share_while_reading_permille: Option<u32>,
    /// Steps that hit the limit since the model loaded.
    pub overflow_count: u64,
    /// Redo attempts made since the model loaded.
    pub redo_attempt_count: u64,
}

/// What the governor says after one measured step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepVerdict {
    pub capacity: KeptWeightCapacity,
    pub reading: LedgerReading,
}

/// One attempt at redoing a step that hit the limit. It carries the room
/// leased for the attempt, which must be settled or abandoned.
#[derive(Debug, Eq, PartialEq)]
#[must_use = "a redo attempt holds leased room until it is settled or abandoned"]
pub struct RedoAttempt {
    headroom_lease: LeaseId,
    headroom_bytes: u64,
    attempt_number: u32,
    phase: LedgerPhase,
}

impl RedoAttempt {
    #[must_use]
    pub const fn attempt_number(&self) -> u32 {
        self.attempt_number
    }

    #[must_use]
    pub const fn headroom_bytes(&self) -> u64 {
        self.headroom_bytes
    }
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

/// The owner of one loaded model's ledger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryGovernor {
    ledger: MemoryLedger,
    worst_readings: WorstReadings,
    probed_working_memory_bytes: Option<u64>,
    overflow_count: u64,
    redo_attempt_count: u64,
    latest_reading: Option<LedgerReading>,
}

impl MemoryGovernor {
    /// Opens the governor for a model that has just loaded: idle, no
    /// request, nothing measured.
    pub fn new(limit_bytes: u64, model_shape: ModelMemoryShape) -> Result<Self, MemoryLedgerError> {
        Ok(Self {
            ledger: MemoryLedger::new(limit_bytes, model_shape)?,
            worst_readings: WorstReadings::default(),
            probed_working_memory_bytes: None,
            overflow_count: 0,
            redo_attempt_count: 0,
            latest_reading: None,
        })
    }

    /// Every account as it stands now.
    #[must_use]
    pub fn accounts(&self) -> LedgerAccounts {
        self.ledger.accounts()
    }

    /// What the model is doing, as far as memory is concerned.
    #[must_use]
    pub const fn phase(&self) -> LedgerPhase {
        self.ledger.phase()
    }

    /// The capacity in force, for a holder that has just been created and
    /// has not been told yet.
    pub fn kept_weight_capacity(&self) -> KeptWeightCapacity {
        capacity_of(self.ledger.accounts())
    }

    /// The reading of the most recent measured step.
    #[must_use]
    pub const fn latest_reading(&self) -> Option<LedgerReading> {
        self.latest_reading
    }

    /// Steps that hit the limit since the model loaded.
    #[must_use]
    pub const fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    /// Redo attempts made since the model loaded.
    #[must_use]
    pub const fn redo_attempt_count(&self) -> u64 {
        self.redo_attempt_count
    }

    /// A working memory measured outside a request, such as one layer step
    /// probed at load.
    pub fn record_probe(
        &mut self,
        phase: LedgerPhase,
        working_memory_bytes: u64,
    ) -> KeptWeightCapacity {
        self.probed_working_memory_bytes = Some(
            self.probed_working_memory_bytes
                .unwrap_or(0)
                .max(working_memory_bytes),
        );
        capacity_of(
            self.ledger
                .record_working_memory(phase, working_memory_bytes),
        )
    }

    pub fn change_limit(
        &mut self,
        limit_bytes: u64,
    ) -> Result<KeptWeightCapacity, MemoryLedgerError> {
        self.ledger.change_limit(limit_bytes).map(capacity_of)
    }

    /// A request starts in `phase` with this much stored state to hold: for
    /// a prompt, what the whole prompt will occupy once it is read.
    pub fn start_request(&mut self, phase: LedgerPhase, context_bytes: u64) -> KeptWeightCapacity {
        self.worst_readings = WorstReadings::default();
        self.ledger.start_request(context_bytes);
        capacity_of(self.ledger.enter_phase(phase))
    }

    pub fn enter_phase(&mut self, phase: LedgerPhase) -> KeptWeightCapacity {
        capacity_of(self.ledger.enter_phase(phase))
    }

    pub fn grow_context(&mut self, additional_context_bytes: u64) -> KeptWeightCapacity {
        capacity_of(self.ledger.grow_context(additional_context_bytes))
    }

    /// The request is over; whatever stored state outlives it stays on the
    /// context account.
    pub fn finish_request(&mut self, retained_context_bytes: u64) -> KeptWeightCapacity {
        capacity_of(self.ledger.finish_request(retained_context_bytes))
    }

    pub fn grant_lease(
        &mut self,
        purpose: LeasePurpose,
        lease_bytes: u64,
    ) -> (LeaseId, KeptWeightCapacity) {
        let (lease_id, accounts) = self.ledger.grant_lease(purpose, lease_bytes);
        (lease_id, capacity_of(accounts))
    }

    pub fn give_back_lease(
        &mut self,
        lease_id: LeaseId,
    ) -> Result<KeptWeightCapacity, MemoryLedgerError> {
        self.ledger.give_back_lease(lease_id).map(capacity_of)
    }

    /// One step completed and was measured.
    pub fn complete_step(&mut self, measured_step: MeasuredStep) -> StepVerdict {
        let accounts = self
            .ledger
            .record_working_memory(measured_step.phase, measured_step.working_memory_bytes());
        self.verdict_after(measured_step, accounts)
    }

    /// A step hit the limit. What was set aside was too little, so the
    /// first attempt leases as much again and each later one doubles it.
    /// When the failure itself says how far over the limit the step went
    /// (`known_shortfall_bytes`, zero when it does not), the lease is at
    /// least that, so a reserve that was badly wrong is not approached in
    /// doublings. `None` once the attempts are spent: the limit cannot hold
    /// the step.
    ///
    /// The engine applies the returned capacity, which gives back just the
    /// kept weights the leased room displaces, then runs the step again.
    pub fn begin_redo(
        &mut self,
        phase: LedgerPhase,
        failed_attempt: Option<RedoAttempt>,
        known_shortfall_bytes: u64,
    ) -> Option<(RedoAttempt, KeptWeightCapacity)> {
        let (attempt_number, doubled_headroom_bytes) = match failed_attempt {
            None => {
                self.overflow_count += 1;
                (1, self.ledger.accounts().working_memory_bytes.max(1))
            }
            Some(failed_attempt) => {
                // The lease was granted by this governor, so giving it back
                // cannot fail; a lease already gone changes nothing.
                let _ = self.ledger.give_back_lease(failed_attempt.headroom_lease);
                (
                    failed_attempt.attempt_number + 1,
                    failed_attempt.headroom_bytes.saturating_mul(2),
                )
            }
        };
        let headroom_bytes = doubled_headroom_bytes.max(known_shortfall_bytes);
        if attempt_number > MAXIMUM_REDO_ATTEMPT_COUNT {
            return None;
        }
        self.redo_attempt_count += 1;
        let (headroom_lease, accounts) = self
            .ledger
            .grant_lease(LeasePurpose::RedoHeadroom, headroom_bytes);
        Some((
            RedoAttempt {
                headroom_lease,
                headroom_bytes,
                attempt_number,
                phase,
            },
            capacity_of(accounts),
        ))
    }

    /// The redone step fitted and was measured: its room goes back and the
    /// measurement replaces the guess, as one ledger event.
    pub fn complete_redone_step(
        &mut self,
        redo_attempt: RedoAttempt,
        measured_step: MeasuredStep,
    ) -> StepVerdict {
        let settled_accounts = self.ledger.settle_redone_step(
            redo_attempt.headroom_lease,
            redo_attempt.phase,
            measured_step.working_memory_bytes(),
        );
        // A lease already gone leaves only the measurement to record.
        let accounts = match settled_accounts {
            Ok(accounts) => accounts,
            Err(_) => self
                .ledger
                .record_working_memory(redo_attempt.phase, measured_step.working_memory_bytes()),
        };
        self.verdict_after(measured_step, accounts)
    }

    /// The redo will not be tried again, for a reason other than memory:
    /// its room goes back.
    pub fn abandon_redo(&mut self, redo_attempt: RedoAttempt) -> KeptWeightCapacity {
        let _ = self.ledger.give_back_lease(redo_attempt.headroom_lease);
        capacity_of(self.ledger.accounts())
    }

    fn verdict_after(
        &mut self,
        measured_step: MeasuredStep,
        accounts: LedgerAccounts,
    ) -> StepVerdict {
        let reconciliation = self.ledger.reconcile(
            measured_step.held_memory,
            measured_step.active_bytes_after_step,
        );
        self.remember_worst(measured_step, accounts, reconciliation);
        let worst_readings = self.worst_readings;
        let reading = LedgerReading {
            accounts,
            held_memory: measured_step.held_memory,
            reconciliation,
            step_peak_beyond_held_bytes: measured_step.peak_beyond_held_bytes(),
            observed_step_count: worst_readings.observed_step_count,
            largest_unaccounted_bytes: worst_readings.largest_unaccounted_bytes,
            largest_overcounted_bytes: worst_readings.largest_overcounted_bytes,
            largest_kept_over_capacity_bytes: worst_readings.largest_kept_over_capacity_bytes,
            largest_reading_step_peak_beyond_held_bytes: worst_readings
                .largest_reading_step_peak_beyond_held_bytes,
            largest_writing_step_peak_beyond_held_bytes: worst_readings
                .largest_writing_step_peak_beyond_held_bytes,
            probed_working_memory_bytes: self.probed_working_memory_bytes,
            lowest_kept_share_while_reading_permille: worst_readings
                .lowest_kept_share_while_reading_permille,
            overflow_count: self.overflow_count,
            redo_attempt_count: self.redo_attempt_count,
        };
        self.latest_reading = Some(reading);
        StepVerdict {
            capacity: capacity_of(accounts),
            reading,
        }
    }

    fn remember_worst(
        &mut self,
        measured_step: MeasuredStep,
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
        let step_peak_beyond_held_bytes = measured_step.peak_beyond_held_bytes();
        match measured_step.phase {
            LedgerPhase::Idle => {}
            LedgerPhase::WritingReply => {
                worst_readings.largest_writing_step_peak_beyond_held_bytes = worst_readings
                    .largest_writing_step_peak_beyond_held_bytes
                    .max(step_peak_beyond_held_bytes);
            }
            LedgerPhase::ReadingPrompt { .. } => {
                worst_readings.largest_reading_step_peak_beyond_held_bytes = worst_readings
                    .largest_reading_step_peak_beyond_held_bytes
                    .max(step_peak_beyond_held_bytes);
                worst_readings.lowest_kept_share_while_reading_permille = lower_kept_share(
                    worst_readings.lowest_kept_share_while_reading_permille,
                    measured_step.held_memory.kept_weight_bytes,
                    accounts.kept_weight_capacity_bytes,
                );
            }
        }
    }
}

const fn capacity_of(accounts: LedgerAccounts) -> KeptWeightCapacity {
    KeptWeightCapacity {
        bytes: accounts.kept_weight_capacity_bytes,
    }
}

/// The lower of the share so far and this step's: kept weights as
/// thousandths of the capacity, capped at a thousand. A step with no
/// capacity to fill leaves the share as it was.
fn lower_kept_share(
    lowest_so_far_permille: Option<u32>,
    kept_weight_bytes: u64,
    kept_weight_capacity_bytes: u64,
) -> Option<u32> {
    if kept_weight_capacity_bytes == 0 {
        return lowest_so_far_permille;
    }
    let share_permille =
        u128::from(kept_weight_bytes) * 1_000 / u128::from(kept_weight_capacity_bytes);
    let share_permille = u32::try_from(share_permille.min(1_000)).unwrap_or(1_000);
    Some(match lowest_so_far_permille {
        Some(lowest_so_far_permille) => lowest_so_far_permille.min(share_permille),
        None => share_permille,
    })
}

#[cfg(test)]
#[path = "governor_tests.rs"]
mod tests;
