//! The ledger compared with what MLX reports.
//!
//! The ledger says what should be in memory; MLX says what is. A difference
//! is a code path that allocates without telling the ledger, or an account
//! that believes it holds memory it does not. Reporting that difference at
//! every layer boundary, and failing a test on it, is what turns an unknown
//! allocation into a defect someone can see.

/// What the engine actually holds at one boundary, from its own owners.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HeldMemory {
    /// The conversation's stored state as it is now.
    pub context_bytes: u64,
    /// Streamable weights currently kept.
    pub kept_weight_bytes: u64,
    /// Weights read from storage and not yet dropped or kept.
    pub stream_slot_bytes: u64,
}

/// The outcome of comparing the ledger with one MLX reading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LedgerReconciliation {
    /// What the ledger says is in memory.
    pub expected_active_bytes: u64,
    /// What MLX reports as in memory.
    pub actual_active_bytes: u64,
    /// Memory in use that no account names.
    pub unaccounted_bytes: u64,
    /// Memory an account names that is not in use.
    pub overcounted_bytes: u64,
    /// Kept weights beyond the capacity the ledger allows.
    pub kept_over_capacity_bytes: u64,
}

impl LedgerReconciliation {
    /// Compares what the ledger expects in memory with one MLX reading.
    #[must_use]
    pub const fn new(
        expected_active_bytes: u64,
        actual_active_bytes: u64,
        kept_weight_bytes: u64,
        kept_weight_capacity_bytes: u64,
    ) -> Self {
        Self {
            expected_active_bytes,
            actual_active_bytes,
            unaccounted_bytes: actual_active_bytes.saturating_sub(expected_active_bytes),
            overcounted_bytes: expected_active_bytes.saturating_sub(actual_active_bytes),
            kept_over_capacity_bytes: kept_weight_bytes.saturating_sub(kept_weight_capacity_bytes),
        }
    }

    /// The size of the disagreement in either direction.
    #[must_use]
    pub const fn disagreement_bytes(&self) -> u64 {
        if self.unaccounted_bytes > self.overcounted_bytes {
            self.unaccounted_bytes
        } else {
            self.overcounted_bytes
        }
    }
}

#[cfg(test)]
#[path = "reconciliation_tests.rs"]
mod tests;
