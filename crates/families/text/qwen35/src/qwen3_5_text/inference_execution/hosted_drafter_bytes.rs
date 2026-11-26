//! What a hosted drafter holds, measured across its load.
//!
//! MLX's active memory rises by the drafter's weights while it loads, but
//! under a tight limit the memory governor gives kept weights back during
//! that same load to make the room, so the rise alone understates it: a
//! dense 27B at a 12 GB limit read a rise of nothing for a drafter of
//! 1.02 GB, counted it as holding nothing, and planned its kept layers into
//! the drafter's room (issue #2572). The drafter is what MLX holds outside
//! the kept weights after the load, less what it held outside them before.

/// MLX's active memory and the kept weights' share of it at one moment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MemoryAroundALoad {
    pub(super) active_bytes: u64,
    pub(super) kept_weight_bytes: u64,
}

impl MemoryAroundALoad {
    fn outside_the_kept_weights(self) -> u64 {
        self.active_bytes.saturating_sub(self.kept_weight_bytes)
    }
}

/// Bytes a load added outside the kept weights, whatever the kept weights
/// did meanwhile; zero when memory outside them did not grow.
pub(super) fn bytes_a_load_added(
    before_the_load: MemoryAroundALoad,
    after_the_load: MemoryAroundALoad,
) -> u64 {
    after_the_load
        .outside_the_kept_weights()
        .saturating_sub(before_the_load.outside_the_kept_weights())
}

#[cfg(test)]
#[path = "hosted_drafter_bytes_tests.rs"]
mod tests;
