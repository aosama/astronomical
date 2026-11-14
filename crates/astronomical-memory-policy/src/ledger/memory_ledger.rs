//! The owner of the limit: the events that change the accounts.
//!
//! Kept-weight capacity is never stored. It is composed from the accounts in
//! front of it each time it is asked for, so it changes only when one of
//! them does: a request starts or ends, the phase changes, the context
//! grows, the limit moves, a layer step measures more working memory than
//! was known, or a lease is granted or given back. Between those events it
//! does not move, which is what keeps a request from trimming and refilling
//! the same weights.

use thiserror::Error;

use super::accounts::{AccountInputs, LedgerAccounts, compose_accounts};
use super::lease::{LeaseBook, LeaseId, LeasePurpose};
use super::model_shape::ModelMemoryShape;
use super::phase::LedgerPhase;
use super::reconciliation::{HeldMemory, LedgerReconciliation};
use super::working_memory::WorkingMemoryMeasurements;

/// A request the ledger cannot carry out.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MemoryLedgerError {
    #[error("the memory limit must be at least one byte")]
    ZeroLimit,
    #[error("no such lease is held; it was given back already or never granted")]
    UnknownLease,
}

/// One loaded model's memory, by account.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryLedger {
    limit_bytes: u64,
    model_shape: ModelMemoryShape,
    context_bytes: u64,
    phase: LedgerPhase,
    working_memory: WorkingMemoryMeasurements,
    leases: LeaseBook,
}

impl MemoryLedger {
    /// Opens the ledger for a model that has just loaded: idle, no context,
    /// nothing measured.
    pub fn new(limit_bytes: u64, model_shape: ModelMemoryShape) -> Result<Self, MemoryLedgerError> {
        if limit_bytes == 0 {
            return Err(MemoryLedgerError::ZeroLimit);
        }
        Ok(Self {
            limit_bytes,
            model_shape,
            context_bytes: 0,
            phase: LedgerPhase::Idle,
            working_memory: WorkingMemoryMeasurements::default(),
            leases: LeaseBook::default(),
        })
    }

    /// Every account as it stands now.
    #[must_use]
    pub fn accounts(&self) -> LedgerAccounts {
        compose_accounts(AccountInputs {
            limit_bytes: self.limit_bytes,
            model_shape: self.model_shape,
            context_bytes: self.context_bytes,
            phase: self.phase,
            working_memory_reserve: self.working_memory.reserve_for(self.phase),
            lease_bytes: self.leases.held_bytes(),
        })
    }

    #[must_use]
    pub const fn phase(&self) -> LedgerPhase {
        self.phase
    }

    #[must_use]
    pub const fn model_shape(&self) -> ModelMemoryShape {
        self.model_shape
    }

    /// Moves the limit. Measurements are kept: one layer's working memory
    /// does not depend on how much else is in memory.
    pub fn change_limit(&mut self, limit_bytes: u64) -> Result<LedgerAccounts, MemoryLedgerError> {
        if limit_bytes == 0 {
            return Err(MemoryLedgerError::ZeroLimit);
        }
        self.limit_bytes = limit_bytes;
        Ok(self.accounts())
    }

    /// A request starts with this much stored state to hold.
    pub fn start_request(&mut self, context_bytes: u64) -> LedgerAccounts {
        self.context_bytes = context_bytes;
        self.accounts()
    }

    /// The stored state grows, as it does while a reply is written.
    pub fn grow_context(&mut self, additional_context_bytes: u64) -> LedgerAccounts {
        self.context_bytes = self.context_bytes.saturating_add(additional_context_bytes);
        self.accounts()
    }

    /// The request is over. Whatever stored state outlives it, such as a
    /// prefix kept for the next turn, stays on the context account.
    pub fn finish_request(&mut self, retained_context_bytes: u64) -> LedgerAccounts {
        self.context_bytes = retained_context_bytes;
        self.phase = LedgerPhase::Idle;
        self.accounts()
    }

    /// The model moves between idle, reading a prompt and writing a reply.
    pub fn enter_phase(&mut self, phase: LedgerPhase) -> LedgerAccounts {
        self.phase = phase;
        self.accounts()
    }

    /// One layer step of `phase` needed this much working memory. A step
    /// that hit the limit reports at least what it reached, so the next
    /// attempt is planned around the larger number.
    pub fn record_working_memory(
        &mut self,
        phase: LedgerPhase,
        observed_working_memory_bytes: u64,
    ) -> LedgerAccounts {
        self.working_memory
            .record(phase, observed_working_memory_bytes);
        self.accounts()
    }

    /// Sets memory aside for a need outside the normal pattern.
    pub fn grant_lease(
        &mut self,
        purpose: LeasePurpose,
        lease_bytes: u64,
    ) -> (LeaseId, LedgerAccounts) {
        let lease_id = self.leases.grant(purpose, lease_bytes);
        (lease_id, self.accounts())
    }

    /// The need is over; its bytes return to the remainder.
    pub fn give_back_lease(
        &mut self,
        lease_id: LeaseId,
    ) -> Result<LedgerAccounts, MemoryLedgerError> {
        if self.leases.give_back(lease_id).is_none() {
            return Err(MemoryLedgerError::UnknownLease);
        }
        Ok(self.accounts())
    }

    /// A redone layer step fitted and was measured: its headroom lease goes
    /// back and the measurement is recorded, as one event.
    ///
    /// They are one event because the order matters. With the measurement
    /// recorded while the headroom was still held, both counted against the
    /// limit for a moment, and the first simulated redo gave back twice the
    /// weights it needed to.
    pub fn settle_redone_step(
        &mut self,
        headroom_lease: LeaseId,
        phase: LedgerPhase,
        observed_working_memory_bytes: u64,
    ) -> Result<LedgerAccounts, MemoryLedgerError> {
        if self.leases.give_back(headroom_lease).is_none() {
            return Err(MemoryLedgerError::UnknownLease);
        }
        Ok(self.record_working_memory(phase, observed_working_memory_bytes))
    }

    /// Compares what the ledger expects in memory with one MLX reading.
    /// Working memory is absent on purpose: at a layer boundary the step is
    /// over and its working memory has been released. A redo's headroom is
    /// absent too: it is room set aside, not memory anything holds.
    #[must_use]
    pub fn reconcile(
        &self,
        held_memory: HeldMemory,
        actual_active_bytes: u64,
    ) -> LedgerReconciliation {
        let expected_active_bytes = self
            .model_shape
            .core_weight_bytes
            .saturating_add(held_memory.context_bytes)
            .saturating_add(held_memory.kept_weight_bytes)
            .saturating_add(held_memory.stream_slot_bytes)
            .saturating_add(self.leases.allocated_bytes());
        LedgerReconciliation::new(
            expected_active_bytes,
            actual_active_bytes,
            held_memory.kept_weight_bytes,
            self.accounts().kept_weight_capacity_bytes,
        )
    }
}

#[cfg(test)]
#[path = "memory_ledger_tests.rs"]
mod tests;
