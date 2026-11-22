//! The memory ledger's reading of one completed step, as the worker sends it.

use serde::{Deserialize, Serialize};

/// The memory ledger's reading of one completed step: the accounts the
/// engine's memory governor divides the limit into, what the engine held,
/// and how far the two were from what MLX reported.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerMemoryLedgerSnapshot {
    pub limit_bytes: u64,
    pub core_weight_bytes: u64,
    pub context_bytes: u64,
    pub mlx_working_buffer_bytes: u64,
    /// Room kept free for the `mlx_buffer_cache` while a reply is written.
    pub mlx_buffer_cache_bytes: u64,
    /// Part of the limit left unused so the memory macOS has wired across
    /// the whole system stays under its wired ceiling.
    pub system_wired_reserve_bytes: u64,
    /// The memory macOS had wired across the whole system at the last
    /// reading, and the kernel's wire limit; zero before any reading.
    #[serde(default)]
    pub system_wired_bytes: u64,
    #[serde(default)]
    pub system_wire_limit_bytes: u64,
    pub stream_slot_bytes: u64,
    pub lease_bytes: u64,
    pub margin_bytes: u64,
    /// What the ledger leaves for kept weights: always the remainder.
    pub kept_weight_capacity_bytes: u64,
    pub unassigned_bytes: u64,
    pub shortfall_bytes: u64,
    pub mlx_working_buffers_are_measured: bool,
    pub held_context_bytes: u64,
    pub held_kept_weight_bytes: u64,
    pub held_stream_slot_bytes: u64,
    /// What the ledger expected MLX to report as in use.
    pub expected_active_bytes: u64,
    /// Memory in use that no account names.
    pub unaccounted_bytes: u64,
    /// Memory an account names that is not in use.
    pub overcounted_bytes: u64,
    pub kept_over_capacity_bytes: u64,
    pub step_peak_beyond_held_bytes: u64,
    pub observed_step_count: u64,
    pub largest_unaccounted_bytes: u64,
    pub largest_overcounted_bytes: u64,
    pub largest_kept_over_capacity_bytes: u64,
    /// The largest rise of one prompt step's peak above what was held
    /// around it, and the same for one reply step.
    pub largest_reading_step_peak_beyond_held_bytes: u64,
    pub largest_writing_step_peak_beyond_held_bytes: u64,
    /// The MLX working buffers of one layer step measured at load; `None` when
    /// nothing was probed.
    #[serde(default)]
    pub probed_mlx_working_buffer_bytes: Option<u64>,
    /// The emptiest the kept weights were against the capacity while a
    /// prompt was read, in thousandths.
    pub lowest_kept_share_while_reading_permille: Option<u32>,
    /// Whether MLX itself refuses an allocation past `limit_bytes`.
    pub active_memory_ceiling_enforcement: ActiveMemoryCeilingEnforcement,
}

/// Whether MLX refuses allocations past the active memory ceiling.
///
/// The ledger plans every build against the ceiling, but only the Metal build
/// links Astronomical's patched allocator, which refuses an allocation that
/// would pass it with a typed, recoverable error. MLX's other allocators only
/// use the limit to decide when to release the `mlx_buffer_cache`, so past
/// the ceiling the operating system intervenes instead (issue #2425).
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveMemoryCeilingEnforcement {
    /// MLX refuses an allocation that would pass the ceiling.
    EnforcedByMlx,
    /// Nothing in MLX enforces the ceiling on this build.
    Unavailable,
}

impl ActiveMemoryCeilingEnforcement {
    /// The enforcement of the build this process runs: macOS builds the
    /// Metal backend with the patched allocator, every other platform builds
    /// one of MLX's stock allocators.
    #[must_use]
    pub const fn for_this_build() -> Self {
        if cfg!(target_os = "macos") {
            Self::EnforcedByMlx
        } else {
            Self::Unavailable
        }
    }
}
