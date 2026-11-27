//! Why an engine refuses to move its memory ceiling, and the floor it holds
//! a change to.
//!
//! Every engine takes a new ceiling live, between requests and between the
//! steps of a running one (issue #2681). This module is the shared vocabulary
//! of their refusals and the pure rules for the floor, so the engines cannot
//! disagree about what "too low" means.

use crate::telemetry::MlxMemoryLimitAdjustment;

/// Why a live ceiling change was not applied. The engine keeps serving at its
/// previous ceiling after any of these.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveMemoryLimitRefusal {
    /// The requested ceiling is below what the loaded model holds.
    BelowModelFloor {
        /// The smallest ceiling the loaded model can run under.
        minimum_mlx_memory_ceiling_bytes: u64,
    },
    /// A request or training run is in progress; limits never move under it.
    EngineBusy,
    /// This engine cannot change its ceiling while loaded.
    Unsupported,
    /// The runtime refused the change for a reason outside the engine's rules.
    Failed {
        /// The runtime's own description of the failure.
        reason: String,
    },
}

/// An engine that can move its MLX memory ceiling while its model stays
/// loaded. The default refuses, which the worker answers as a clean rejection,
/// so an engine that cannot move its ceiling implements the trait empty.
///
/// It is a supertrait of the embedding, decision and speech engine traits, and
/// a boxed engine forwards it here once instead of in each of them.
pub trait LiveMemoryLimitEngine {
    /// Moves the ceiling to `requested_mlx_memory_ceiling_bytes`, or says why
    /// the engine keeps serving at the one it has.
    ///
    /// # Errors
    ///
    /// The refusal names whether the ceiling is below the model's floor, the
    /// engine is busy, the engine cannot move it, or the runtime failed.
    fn update_mlx_memory_limit(
        &mut self,
        _requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, LiveMemoryLimitRefusal> {
        Err(LiveMemoryLimitRefusal::Unsupported)
    }
}

impl<Engine: LiveMemoryLimitEngine + ?Sized> LiveMemoryLimitEngine for Box<Engine> {
    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, LiveMemoryLimitRefusal> {
        self.as_mut()
            .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
    }
}

/// Refuses a ceiling below the memory the loaded model already holds.
///
/// A ceiling equal to the held memory is accepted: the model fits exactly, and
/// refusing it would make the floor itself unreachable.
#[must_use]
pub fn live_ceiling_refusal(
    requested_mlx_memory_ceiling_bytes: u64,
    held_mlx_memory_bytes: u64,
) -> Option<LiveMemoryLimitRefusal> {
    (requested_mlx_memory_ceiling_bytes < held_mlx_memory_bytes).then_some(
        LiveMemoryLimitRefusal::BelowModelFloor {
            minimum_mlx_memory_ceiling_bytes: held_mlx_memory_bytes,
        },
    )
}

/// What a running request holds that a lower ceiling cannot take back, read
/// between two of its steps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InFlightRequestMemory {
    /// Memory active now that giving back kept weights would not free: the
    /// model's core, the request's context so far, its drafter.
    pub non_evictable_active_bytes: u64,
    /// The MLX working buffers the request's next step allocates on top.
    pub next_step_working_buffer_bytes: u64,
}

/// The smallest ceiling a model can move to now: its idle floor, raised while
/// a request runs to what that request cannot give back plus room for its
/// next step. Between requests the idle floor alone applies, as it always has.
#[must_use]
pub fn request_aware_ceiling_floor_bytes(
    idle_floor_bytes: u64,
    in_flight_request_memory: Option<InFlightRequestMemory>,
) -> u64 {
    let Some(in_flight_request_memory) = in_flight_request_memory else {
        return idle_floor_bytes;
    };
    let running_request_floor_bytes = in_flight_request_memory
        .non_evictable_active_bytes
        .saturating_add(in_flight_request_memory.next_step_working_buffer_bytes);
    idle_floor_bytes.max(running_request_floor_bytes)
}

/// The words a refused ceiling is answered with: the smallest ceiling that
/// would work, in GB of 2^30 bytes, which the memory controls show beside
/// their red slider.
#[must_use]
pub fn needs_at_least_message(minimum_mlx_memory_ceiling_bytes: u64) -> String {
    format!(
        "needs at least {:.1} GB",
        minimum_mlx_memory_ceiling_bytes as f64 / 1_073_741_824.0
    )
}
