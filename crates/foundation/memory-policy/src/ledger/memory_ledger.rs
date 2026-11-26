//! The owner of the limit: the events that change the accounts.
//!
//! Kept-weight capacity is never stored. It is composed from the accounts in
//! front of it each time it is asked for, so it changes only when one of
//! them does: a request starts or ends, the phase changes, the context
//! grows, the limit moves, a layer step measures more MLX working buffers than
//! was known, or a lease is granted or given back. Between those events it
//! does not move, which is what keeps a request from trimming and refilling
//! the same weights.

use thiserror::Error;

use super::accounts::{AccountInputs, LedgerAccounts, compose_accounts};
use super::lease::{LeaseBook, LeaseId, LeasePurpose};
use super::mlx_working_buffers::MlxWorkingBufferMeasurements;
use super::model_shape::ModelMemoryShape;
use super::phase::LedgerPhase;
use super::reconciliation::{HeldMemory, LedgerReconciliation};
use super::system_wired_ceiling::{
    SystemWiredReading, limit_past_system_wired_ceiling_bytes, next_system_wired_reserve_bytes,
    steady_wired_outside_mlx_bytes, wired_outside_mlx_bytes, with_wired_outside_mlx_bytes,
};

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
    mlx_working_buffers: MlxWorkingBufferMeasurements,
    leases: LeaseBook,
    system_wired_reserve_bytes: u64,
    /// The last reading of the system's wired total and the kernel's wire
    /// limit; `None` before any reading.
    latest_system_wired_reading: Option<(u64, u64)>,
    /// The memory wired outside MLX at the last reading, so a rise that one
    /// reading alone shows does not move the reserve.
    previous_wired_outside_mlx_bytes: Option<u64>,
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
            mlx_working_buffers: MlxWorkingBufferMeasurements::default(),
            leases: LeaseBook::default(),
            system_wired_reserve_bytes: 0,
            latest_system_wired_reading: None,
            previous_wired_outside_mlx_bytes: None,
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
            mlx_working_buffer_reserve: self.mlx_working_buffers.reserve_for(self.phase),
            lease_bytes: self.leases.held_bytes(),
            system_wired_reserve_bytes: self.system_wired_reserve_bytes,
        })
    }

    /// The accounts as they would stand in `phase` with `context_bytes` of
    /// stored state and everything else as it is. A question only: nothing
    /// changes.
    #[must_use]
    pub fn accounts_while(&self, phase: LedgerPhase, context_bytes: u64) -> LedgerAccounts {
        compose_accounts(AccountInputs {
            limit_bytes: self.limit_bytes,
            model_shape: self.model_shape,
            context_bytes,
            phase,
            mlx_working_buffer_reserve: self.mlx_working_buffers.reserve_for(phase),
            lease_bytes: self.leases.held_bytes(),
            system_wired_reserve_bytes: self.system_wired_reserve_bytes,
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

    /// Moves the limit. Measurements are kept: one layer's MLX working buffers
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

    /// The stored state is now exactly this, by the engine's own count.
    pub fn settle_context(&mut self, context_bytes: u64) -> LedgerAccounts {
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

    /// One layer step of `phase` needed this many MLX working buffers. A step
    /// that hit the limit reports at least what it reached, so the next
    /// attempt is planned around the larger number.
    pub fn record_mlx_working_buffers(
        &mut self,
        phase: LedgerPhase,
        observed_mlx_working_buffer_bytes: u64,
    ) -> LedgerAccounts {
        self.mlx_working_buffers
            .record(phase, observed_mlx_working_buffer_bytes);
        self.accounts()
    }

    /// Moves the system wired reserve after a reading of the memory macOS
    /// has wired across the whole system.
    pub fn observe_system_wired(&mut self, reading: SystemWiredReading) -> LedgerAccounts {
        let current_wired_outside_mlx_bytes = wired_outside_mlx_bytes(reading);
        let steady_wired_outside_mlx_bytes = steady_wired_outside_mlx_bytes(
            self.previous_wired_outside_mlx_bytes,
            current_wired_outside_mlx_bytes,
        );
        // What the limit cannot give is not carried forward, or a reserve
        // run up on a small limit would take many readings to come down.
        self.system_wired_reserve_bytes = next_system_wired_reserve_bytes(
            self.system_wired_reserve_bytes,
            reading.system_wired_bytes,
            reading.system_wire_limit_bytes,
        )
        .max(limit_past_system_wired_ceiling_bytes(
            self.limit_bytes,
            with_wired_outside_mlx_bytes(reading, steady_wired_outside_mlx_bytes),
        ))
        .min(self.limit_bytes);
        self.previous_wired_outside_mlx_bytes = Some(current_wired_outside_mlx_bytes);
        self.latest_system_wired_reading =
            Some((reading.system_wired_bytes, reading.system_wire_limit_bytes));
        self.accounts()
    }

    /// The last reading of the system's wired total and the kernel's wire
    /// limit, in that order.
    #[must_use]
    pub const fn latest_system_wired_reading(&self) -> Option<(u64, u64)> {
        self.latest_system_wired_reading
    }

    /// The memory wired outside MLX at the last reading, as read (before
    /// the two-reading rule); `None` before any reading.
    #[must_use]
    pub const fn latest_wired_outside_mlx_bytes(&self) -> Option<u64> {
        self.previous_wired_outside_mlx_bytes
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
        observed_mlx_working_buffer_bytes: u64,
    ) -> Result<LedgerAccounts, MemoryLedgerError> {
        if self.leases.give_back(headroom_lease).is_none() {
            return Err(MemoryLedgerError::UnknownLease);
        }
        Ok(self.record_mlx_working_buffers(phase, observed_mlx_working_buffer_bytes))
    }

    /// Compares what the ledger expects in memory with one MLX reading.
    /// MLX working buffers are absent on purpose: at a layer boundary the step is
    /// over and its MLX working buffers have been released. A redo's headroom is
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
