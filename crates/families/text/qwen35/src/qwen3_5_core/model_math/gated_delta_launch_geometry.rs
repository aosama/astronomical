//! The launch geometry of the software-pipelined Qwen3.5 gated-delta kernels
//! (sequence, sequence with boundary checkpoints, tree and replay).
//!
//! The kernels themselves are the `.metal` files under `metal/`
//! (`astro_family_qwen35_gated_delta_sequence.metal` explains the pipelining
//! and why its text repeats). They hard-code these two numbers, so the
//! launchers take them from here and a unit test checks the kernel text still
//! agrees.

/// Value rows owned by one kernel block.
pub const VALUE_ROW_BLOCK_SIZE: usize = 16;
/// Eight lanes cooperate on each value row.
pub const THREADGROUP_THREAD_COUNT: usize = VALUE_ROW_BLOCK_SIZE * 8;

#[cfg(test)]
#[path = "gated_delta_launch_geometry_tests.rs"]
mod gated_delta_launch_geometry_tests;
