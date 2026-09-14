//! Value vocabulary for the MLX RAM budget owner.
//!
//! These types are the immutable inputs and composed outputs of the budget
//! composition in [`ram.rs`](./ram.rs): model geometry measured once at load,
//! the per-plan snapshot, one live measurement sample, and the construction
//! error. Keeping them beside the owner separates *what flows through the
//! budget* from *how the budget learns and composes*.

use crate::memory::MemoryPhase;
use thiserror::Error;

/// Immutable inputs known once a model is loaded against one ceiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxRamBudgetModelGeometry {
    /// Non-expert resident model payload (language core, optional vision/MTP).
    pub model_core_payload_bytes: u64,
    /// Bytes required if every sparse expert is fully resident.
    pub complete_expert_payload_bytes: u64,
    /// One complete sparse layer; reserved as the streaming workspace.
    pub largest_complete_expert_layer_bytes: u64,
    /// Largest exact top-K page used by one-token decode.
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxRamBudgetMeasurement {
    /// Execution class whose activation high-water this sample may raise.
    pub phase: MemoryPhase,
    /// Context size used to choose a monotonic coarse learning bucket.
    pub context_token_count: u64,
    /// Measured request-owned persistent and transient bytes above model core.
    pub measured_context_and_activation_bytes: u64,
    /// Transient-only high-water independently learned by forward admission.
    pub observed_activation_headroom_bytes: u64,
    /// Explicit operation workspace already reserved by forward admission.
    pub exact_temporary_workspace_bytes: u64,
}

/// Invalid configuration for the RAM budget owner.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MlxRamBudgetError {
    #[error("MLX RAM budget requires a positive active-memory ceiling")]
    InvalidCeiling,
}
