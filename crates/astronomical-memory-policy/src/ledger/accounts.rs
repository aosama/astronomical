//! The accounts at one instant, and the one rule that composes them.

use super::model_shape::ModelMemoryShape;
use super::phase::LedgerPhase;
use super::working_memory::WorkingMemoryReserve;

/// The safety margin as a divisor of the limit: one fiftieth, two percent.
///
/// The margin is the room for what no account names: allocator rounding,
/// buffers MLX has freed but not yet returned. MLX itself tolerates one
/// percent above the limit; planning to within that percent is what turned
/// a 0.3 GB misjudgment into a discarded cache (issue #2221). Two percent is
/// the starting value and is to be settled by measurement.
const SAFETY_MARGIN_DIVISOR: u64 = 50;

/// What composes the accounts: everything the ledger knows at one instant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AccountInputs {
    pub(super) limit_bytes: u64,
    pub(super) model_shape: ModelMemoryShape,
    pub(super) context_bytes: u64,
    pub(super) phase: LedgerPhase,
    pub(super) working_memory_reserve: WorkingMemoryReserve,
    pub(super) lease_bytes: u64,
}

/// Every account under the limit at one instant.
///
/// The needs come first. `kept_weight_capacity_bytes` is what is left for
/// kept weights and `unassigned_bytes` is what is left when the whole model
/// is already kept. Their sum with the needs is the limit, unless the needs
/// alone exceed it, in which case `shortfall_bytes` says by how much.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LedgerAccounts {
    pub limit_bytes: u64,
    pub core_weight_bytes: u64,
    pub context_bytes: u64,
    pub working_memory_bytes: u64,
    pub stream_slot_bytes: u64,
    pub lease_bytes: u64,
    pub margin_bytes: u64,
    /// How many streamable weight bytes may be kept. Always the remainder.
    pub kept_weight_capacity_bytes: u64,
    /// Room left over once every streamable weight is kept.
    pub unassigned_bytes: u64,
    /// How far the needs alone exceed the limit. Nonzero means the limit is
    /// below what streaming everything requires; the model still runs, and
    /// MLX and macOS decide the rest (issue #1902).
    pub shortfall_bytes: u64,
    /// Whether a measurement covers the working memory of this phase.
    pub working_memory_is_measured: bool,
}

impl LedgerAccounts {
    /// Bytes the phase needs before any weight is kept.
    #[must_use]
    pub const fn needed_bytes(&self) -> u64 {
        self.core_weight_bytes
            .saturating_add(self.context_bytes)
            .saturating_add(self.working_memory_bytes)
            .saturating_add(self.stream_slot_bytes)
            .saturating_add(self.lease_bytes)
            .saturating_add(self.margin_bytes)
    }

    /// Whether every streamable weight fits beside the needs, so nothing has
    /// to be read from storage.
    #[must_use]
    pub const fn keeps_every_weight(&self, model_shape: &ModelMemoryShape) -> bool {
        self.kept_weight_capacity_bytes >= model_shape.streamable_weight_bytes
    }
}

/// Composes the accounts.
///
/// Reading weights from storage costs its slots first, so the slots are
/// charged only when the model does not fit without them: a limit just short
/// of the whole model keeps noticeably less than the whole model, not one
/// layer less.
pub(super) const fn compose_accounts(account_inputs: AccountInputs) -> LedgerAccounts {
    let model_shape = account_inputs.model_shape;
    let margin_bytes = account_inputs.limit_bytes / SAFETY_MARGIN_DIVISOR;
    let fixed_bytes = model_shape
        .core_weight_bytes
        .saturating_add(account_inputs.context_bytes)
        .saturating_add(account_inputs.lease_bytes)
        .saturating_add(margin_bytes);
    let stream_slot_bytes = model_shape.stream_slot_bytes_for(account_inputs.phase);
    let mut accounts = LedgerAccounts {
        limit_bytes: account_inputs.limit_bytes,
        core_weight_bytes: model_shape.core_weight_bytes,
        context_bytes: account_inputs.context_bytes,
        working_memory_bytes: 0,
        stream_slot_bytes: 0,
        lease_bytes: account_inputs.lease_bytes,
        margin_bytes,
        kept_weight_capacity_bytes: 0,
        unassigned_bytes: 0,
        shortfall_bytes: 0,
        working_memory_is_measured: true,
    };
    let WorkingMemoryReserve::Measured(working_memory_bytes) =
        account_inputs.working_memory_reserve
    else {
        return with_unmeasured_working_memory(accounts, fixed_bytes, stream_slot_bytes);
    };
    accounts.working_memory_bytes = working_memory_bytes;
    let needed_without_slots_bytes = fixed_bytes.saturating_add(working_memory_bytes);
    let room_without_slots_bytes = accounts
        .limit_bytes
        .saturating_sub(needed_without_slots_bytes);
    if room_without_slots_bytes >= model_shape.streamable_weight_bytes {
        accounts.kept_weight_capacity_bytes = model_shape.streamable_weight_bytes;
        accounts.unassigned_bytes = room_without_slots_bytes - model_shape.streamable_weight_bytes;
        return accounts;
    }
    accounts.stream_slot_bytes = stream_slot_bytes;
    let needed_bytes = needed_without_slots_bytes.saturating_add(stream_slot_bytes);
    accounts.kept_weight_capacity_bytes = accounts.limit_bytes.saturating_sub(needed_bytes);
    accounts.shortfall_bytes = needed_bytes.saturating_sub(accounts.limit_bytes);
    accounts
}

/// With no measurement, nothing is kept and everything streams: the whole
/// remainder is held for the unknown working memory until a layer step has
/// been measured.
const fn with_unmeasured_working_memory(
    mut accounts: LedgerAccounts,
    fixed_bytes: u64,
    stream_slot_bytes: u64,
) -> LedgerAccounts {
    let needed_before_working_memory_bytes = fixed_bytes.saturating_add(stream_slot_bytes);
    accounts.working_memory_is_measured = false;
    accounts.stream_slot_bytes = stream_slot_bytes;
    accounts.working_memory_bytes = accounts
        .limit_bytes
        .saturating_sub(needed_before_working_memory_bytes);
    accounts.shortfall_bytes =
        needed_before_working_memory_bytes.saturating_sub(accounts.limit_bytes);
    accounts
}

#[cfg(test)]
#[path = "accounts_tests.rs"]
mod tests;
