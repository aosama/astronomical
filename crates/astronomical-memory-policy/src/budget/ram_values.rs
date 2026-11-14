//! Value types for a model's memory facts: the geometry measured once at
//! load, the snapshot the status breakdown is composed against, and the
//! construction error.

use thiserror::Error;

/// Immutable inputs known once a model is loaded against one ceiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxRamBudgetModelGeometry {
    /// Non-expert resident model payload (language core, optional vision).
    pub model_core_payload_bytes: u64,
    /// Bytes required if every sparse expert is fully resident.
    pub complete_expert_payload_bytes: u64,
    /// One complete sparse layer; reserved as the streaming workspace.
    pub largest_complete_expert_layer_bytes: u64,
    /// Largest top-K page used by one-token decode: the sound ceiling-rounded
    /// lower bound derived from aggregate layer facts (see `budget/ram_geometry`).
    pub largest_routed_expert_page_bytes: u64,
    /// Persistent decoder-state bytes added by one more prompt token.
    pub sequence_state_bytes_per_token: u64,
}

/// One composed RAM split for a planned operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxRamBudgetSnapshot {
    /// Total MLX active-memory ceiling for this plan.
    pub mlx_active_memory_ceiling_bytes: u64,
    /// Non-expert model core already charged against the ceiling.
    pub model_core_payload_bytes: u64,
    /// Reserved bytes for context-window growth at the planned token count.
    pub context_window_reserve_bytes: u64,
    /// Reserved bytes for temporary activations / transient workspace.
    pub activation_headroom_bytes: u64,
    /// Reserved bytes for one complete-layer stream workspace.
    pub complete_layer_stream_slot_bytes: u64,
    /// Any additional fixed non-expert owners (draft model, publication workspace, …).
    pub other_fixed_bytes: u64,
    /// Leftover budget that may pin retained expert layers in MLX.
    pub retained_expert_budget_bytes: u64,
}

/// Live measurement that refines context-window reserve and activation headroom.
impl MlxRamBudgetSnapshot {
    /// The memory ledger's accounts in this shape.
    ///
    /// Working memory, the margin and any lease are one reserve to a
    /// reader: room held for work in progress. Room left once every
    /// streamable weight is kept has no owner and is reported as a fixed
    /// reserve instead of as unexplained.
    #[must_use]
    pub const fn from_ledger_accounts(accounts: &crate::ledger::LedgerAccounts) -> Self {
        Self {
            mlx_active_memory_ceiling_bytes: accounts.limit_bytes,
            model_core_payload_bytes: accounts.core_weight_bytes,
            context_window_reserve_bytes: accounts.context_bytes,
            activation_headroom_bytes: accounts
                .working_memory_bytes
                .saturating_add(accounts.margin_bytes)
                .saturating_add(accounts.lease_bytes),
            complete_layer_stream_slot_bytes: accounts.stream_slot_bytes,
            other_fixed_bytes: accounts.unassigned_bytes,
            retained_expert_budget_bytes: accounts.kept_weight_capacity_bytes,
        }
    }

    /// Before anything has been measured: the weights that are always in
    /// memory, and the rest of the limit with no owner yet.
    #[must_use]
    pub const fn with_only_weights_set_aside(
        mlx_active_memory_ceiling_bytes: u64,
        model_core_payload_bytes: u64,
    ) -> Self {
        Self {
            mlx_active_memory_ceiling_bytes,
            model_core_payload_bytes,
            context_window_reserve_bytes: 0,
            activation_headroom_bytes: 0,
            complete_layer_stream_slot_bytes: 0,
            other_fixed_bytes: mlx_active_memory_ceiling_bytes
                .saturating_sub(model_core_payload_bytes),
            retained_expert_budget_bytes: 0,
        }
    }
}

/// Invalid configuration for the RAM budget owner.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MlxRamBudgetError {
    #[error("MLX RAM budget requires a positive active-memory ceiling")]
    InvalidCeiling,
}
