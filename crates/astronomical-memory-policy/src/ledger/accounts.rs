//! The accounts at one instant, and the one rule that composes them.

use super::mlx_working_buffers::MlxWorkingBufferReserve;
use super::model_shape::ModelMemoryShape;
use super::phase::LedgerPhase;

/// The safety margin as a divisor of the limit: one fiftieth, two percent.
///
/// The margin is the room for what no account names: allocator rounding,
/// buffers MLX has freed but not yet returned. MLX itself tolerates one
/// percent above the limit; planning to within that percent is what turned
/// a 0.3 GB misjudgment into a discarded astro expert cache (issue #2221). Two percent is
/// the starting value and is to be settled by measurement.
const SAFETY_MARGIN_DIVISOR: u64 = 50;

/// What composes the accounts: everything the ledger knows at one instant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AccountInputs {
    pub(super) limit_bytes: u64,
    pub(super) model_shape: ModelMemoryShape,
    pub(super) context_bytes: u64,
    pub(super) phase: LedgerPhase,
    pub(super) mlx_working_buffer_reserve: MlxWorkingBufferReserve,
    pub(super) lease_bytes: u64,
    /// What the system's wired ceiling asks to be left unused; see
    /// `system_wired_ceiling`.
    pub(super) system_wired_reserve_bytes: u64,
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
    pub mlx_working_buffer_bytes: u64,
    /// Room kept free for the `mlx_buffer_cache`, so MLX can hold one
    /// step's buffers for the next step to reuse. See `mlx_buffer_cache_bytes_for`.
    pub mlx_buffer_cache_bytes: u64,
    /// Part of the limit left unused so the memory macOS has wired across
    /// the whole system stays under the line where it stops keeping MLX's
    /// memory wired. See `system_wired_ceiling`.
    pub system_wired_reserve_bytes: u64,
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
    /// Whether a measurement covers the MLX working buffers of this phase.
    pub mlx_working_buffers_are_measured: bool,
}

impl LedgerAccounts {
    /// Bytes the phase needs before any weight is kept.
    #[must_use]
    pub const fn needed_bytes(&self) -> u64 {
        self.core_weight_bytes
            .saturating_add(self.context_bytes)
            .saturating_add(self.mlx_working_buffer_bytes)
            .saturating_add(self.mlx_buffer_cache_bytes)
            .saturating_add(self.system_wired_reserve_bytes)
            .saturating_add(self.stream_slot_bytes)
            .saturating_add(self.lease_bytes)
            .saturating_add(self.margin_bytes)
    }

    /// How far what stays in memory after a step exceeds the limit: the
    /// fixed weights, the stored state, leases and the margin, without the
    /// step's MLX working buffers. Above zero, no smaller step would fit either.
    #[must_use]
    pub const fn stable_shortfall_bytes(&self) -> u64 {
        self.core_weight_bytes
            .saturating_add(self.context_bytes)
            .saturating_add(self.lease_bytes)
            .saturating_add(self.margin_bytes)
            .saturating_sub(self.limit_bytes)
    }

    /// Everything the accounts set aside apart from kept weights: the
    /// fixed weights, the stored state, one step's MLX working buffers, the
    /// room for the MLX buffer cache, the slots, leases and the margin.
    #[must_use]
    pub const fn set_aside_bytes(&self) -> u64 {
        self.core_weight_bytes
            .saturating_add(self.context_bytes)
            .saturating_add(self.mlx_working_buffer_bytes)
            .saturating_add(self.mlx_buffer_cache_bytes)
            .saturating_add(self.system_wired_reserve_bytes)
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
        mlx_working_buffer_bytes: 0,
        mlx_buffer_cache_bytes: 0,
        system_wired_reserve_bytes: 0,
        stream_slot_bytes: 0,
        lease_bytes: account_inputs.lease_bytes,
        margin_bytes,
        kept_weight_capacity_bytes: 0,
        unassigned_bytes: 0,
        shortfall_bytes: 0,
        mlx_working_buffers_are_measured: true,
    };
    let MlxWorkingBufferReserve::Measured(mlx_working_buffer_bytes) =
        account_inputs.mlx_working_buffer_reserve
    else {
        return with_unmeasured_mlx_working_buffers(accounts, fixed_bytes, stream_slot_bytes);
    };
    accounts.mlx_working_buffer_bytes = mlx_working_buffer_bytes;
    let needed_before_optional_room_bytes = fixed_bytes.saturating_add(mlx_working_buffer_bytes);
    // The reserve comes first: room that is not kept wired is no use to the
    // buffer cache either.
    accounts.system_wired_reserve_bytes = fitting_optional_room_bytes(
        account_inputs.system_wired_reserve_bytes,
        accounts.limit_bytes,
        needed_before_optional_room_bytes.saturating_add(stream_slot_bytes),
    );
    let needed_with_reserve_bytes =
        needed_before_optional_room_bytes.saturating_add(accounts.system_wired_reserve_bytes);
    accounts.mlx_buffer_cache_bytes = fitting_optional_room_bytes(
        mlx_buffer_cache_bytes_for(account_inputs.phase, mlx_working_buffer_bytes),
        accounts.limit_bytes,
        needed_with_reserve_bytes.saturating_add(stream_slot_bytes),
    );
    let needed_without_slots_bytes =
        needed_with_reserve_bytes.saturating_add(accounts.mlx_buffer_cache_bytes);
    let room_without_slots_bytes = accounts
        .limit_bytes
        .saturating_sub(needed_without_slots_bytes);
    if room_without_slots_bytes >= model_shape.streamable_weight_bytes {
        accounts.kept_weight_capacity_bytes = model_shape.streamable_weight_bytes;
        accounts.unassigned_bytes = room_without_slots_bytes - model_shape.streamable_weight_bytes;
        // A model with nothing to stream reaches here even when its needs
        // are over the limit, and that is the one thing its engine asks.
        accounts.shortfall_bytes = needed_without_slots_bytes.saturating_sub(accounts.limit_bytes);
        return accounts;
    }
    accounts.stream_slot_bytes = stream_slot_bytes;
    let needed_bytes = needed_without_slots_bytes.saturating_add(stream_slot_bytes);
    accounts.kept_weight_capacity_bytes = accounts.limit_bytes.saturating_sub(needed_bytes);
    accounts.shortfall_bytes = needed_bytes.saturating_sub(accounts.limit_bytes);
    accounts
}

/// Room the `mlx_buffer_cache` needs beside the working buffers of `phase`.
///
/// When a reply step finishes, MLX keeps its buffers in the
/// `mlx_buffer_cache` and hands them to the next layer, which asks for the
/// same sizes. MLX releases cached buffers once memory in use plus the cache
/// reaches its collection line, so a plan that fills the limit with kept
/// experts leaves the cache empty and every token allocates its buffers
/// afresh. Measured on a 48 GB Mac: a 122B model replied at 28 to 30 tokens
/// per second with room for the cache and at 10 to 11 without it. One reply
/// step's working buffers is what one layer hands to the next, so that is
/// the room set aside.
///
/// Reading a prompt sets nothing aside yet: its working buffers are many
/// times larger, and what they would cost the prompt's kept layers has not
/// been measured.
const fn mlx_buffer_cache_bytes_for(phase: LedgerPhase, mlx_working_buffer_bytes: u64) -> u64 {
    match phase {
        LedgerPhase::WritingReply => mlx_working_buffer_bytes,
        LedgerPhase::Idle | LedgerPhase::ReadingPrompt { .. } => 0,
    }
}

/// The part of an optional room that still fits beside every need.
///
/// The room for the `mlx_buffer_cache` and the system wired reserve only
/// make replies faster; a model runs without them. Each is
/// therefore taken from what would otherwise be kept, never added to a
/// shortfall: on a limit with nothing to spare it shrinks to zero, and
/// whether a model can be served is decided exactly as before it existed.
const fn fitting_optional_room_bytes(
    wanted_room_bytes: u64,
    limit_bytes: u64,
    needed_with_slots_bytes: u64,
) -> u64 {
    let room_bytes = limit_bytes.saturating_sub(needed_with_slots_bytes);
    if wanted_room_bytes < room_bytes {
        wanted_room_bytes
    } else {
        room_bytes
    }
}

/// With no measurement, nothing is kept and everything streams: the whole
/// remainder is held for the unknown MLX working buffers until a layer step has
/// been measured.
const fn with_unmeasured_mlx_working_buffers(
    mut accounts: LedgerAccounts,
    fixed_bytes: u64,
    stream_slot_bytes: u64,
) -> LedgerAccounts {
    let needed_before_mlx_working_buffer_bytes = fixed_bytes.saturating_add(stream_slot_bytes);
    accounts.mlx_working_buffers_are_measured = false;
    accounts.stream_slot_bytes = stream_slot_bytes;
    accounts.mlx_working_buffer_bytes = accounts
        .limit_bytes
        .saturating_sub(needed_before_mlx_working_buffer_bytes);
    accounts.shortfall_bytes =
        needed_before_mlx_working_buffer_bytes.saturating_sub(accounts.limit_bytes);
    accounts
}

#[cfg(test)]
#[path = "accounts_tests.rs"]
mod tests;
