//! How much of the limit to leave unused so macOS keeps MLX's memory wired.
//!
//! macOS wires the memory the graphics processor works on, and its kernel
//! publishes how much it will keep wired (`vm.global_user_wire_limit`).
//! Measured with an MLX-only probe on a 48 GB Mac
//! (`cargo measure-memory-fill-speed`), whose kernel limit is 37.92 GB:
//! fixed work ran at full speed until the memory wired across the whole
//! system reached about 38.3 GB and slowed from there whatever limit MLX had
//! been given; a large sparse model's reply ran at less than half speed half
//! a gigabyte past it. Past the line the system's wired total stops rising
//! with what MLX holds, so some of it is no longer kept wired.
//!
//! The system's wired total is MLX's memory plus whatever the kernel and
//! other applications have wired, which changes as a person opens and
//! closes them. So the room is not a fixed share of the limit: every reading
//! of the system's wired total moves the reserve by how far that total is
//! over or under the kernel's limit.

/// How far under the kernel's limit the system may sit before the reserve
/// is given back, in parts per hundred of that limit.
///
/// Without it the reserve chases every reading: what MLX holds moves by a
/// few hundred megabytes from token to token as its buffer cache fills and
/// empties, and a reserve that followed it would have the engine drop and
/// re-read experts on every token.
const SYSTEM_WIRED_SLACK_PARTS_PER_HUNDRED: u64 = 1;

/// The reserve after one reading of the system's wired total.
///
/// Over the kernel's limit, the reserve grows by exactly the excess, so one
/// reading corrects all of it once the kept weights have followed the
/// capacity. Under it, the reserve is given back only by what the system is
/// under beyond the slack. It never goes below zero: a Mac with room to
/// spare reserves nothing.
#[must_use]
pub const fn next_system_wired_reserve_bytes(
    previous_reserve_bytes: u64,
    system_wired_bytes: u64,
    system_wire_limit_bytes: u64,
) -> u64 {
    let slack_bytes = system_wire_limit_bytes / 100 * SYSTEM_WIRED_SLACK_PARTS_PER_HUNDRED;
    if system_wired_bytes >= system_wire_limit_bytes {
        return previous_reserve_bytes.saturating_add(system_wired_bytes - system_wire_limit_bytes);
    }
    let room_under_limit_bytes = system_wire_limit_bytes - system_wired_bytes;
    previous_reserve_bytes.saturating_sub(room_under_limit_bytes.saturating_sub(slack_bytes))
}

/// One reading of the memory macOS has wired across the whole system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SystemWiredReading {
    pub system_wired_bytes: u64,
    /// The kernel's `vm.global_user_wire_limit`.
    pub system_wire_limit_bytes: u64,
    /// What MLX held at the reading: its active memory and its
    /// `mlx_buffer_cache`. The rest of the wired total is the kernel's and
    /// other applications'.
    pub mlx_held_bytes: u64,
    /// When the reading was made, on a monotonic clock, in milliseconds: the
    /// reserve weighs a reading against the ones before it
    /// (`wired_outside_mlx_window`).
    pub reading_milliseconds: u64,
}

/// The part of the memory limit that would put the system's wired total
/// past the kernel's limit, less the slack, if MLX used all of it while
/// everything else stayed wired as it is.
///
/// The reserve that follows each reading only moves after a step has
/// crossed the line, and the step that crosses it is the first layers of a
/// prompt: a reply leaves the `astro_expert_cache` full, and the prompt's
/// streamed layers and working buffers then come on top of it. On a 48 GB
/// Mac at a 36 GB limit that put the wired total at 38.3 GB against a
/// 37.92 GB line; macOS un-wired most of MLX's memory, compressed it and
/// swapped up to 1.4 GB in the prompt's first seconds (issue #2533). Keeping
/// this much of the limit unused from the start leaves room for any phase's
/// peak; at a 34 GB limit it is zero.
#[must_use]
pub const fn limit_past_system_wired_ceiling_bytes(
    limit_bytes: u64,
    reading: SystemWiredReading,
) -> u64 {
    let wired_outside_mlx_bytes = reading
        .system_wired_bytes
        .saturating_sub(reading.mlx_held_bytes);
    let slack_bytes = reading.system_wire_limit_bytes / 100 * SYSTEM_WIRED_SLACK_PARTS_PER_HUNDRED;
    let room_for_mlx_bytes = reading
        .system_wire_limit_bytes
        .saturating_sub(slack_bytes)
        .saturating_sub(wired_outside_mlx_bytes);
    limit_bytes.saturating_sub(room_for_mlx_bytes)
}

/// The memory wired outside MLX at one reading: the system's wired total
/// less what MLX holds, never below zero.
#[must_use]
pub const fn wired_outside_mlx_bytes(reading: SystemWiredReading) -> u64 {
    reading
        .system_wired_bytes
        .saturating_sub(reading.mlx_held_bytes)
}

/// The same reading with its wired total set so that the memory wired
/// outside MLX is `wired_outside_mlx_bytes`.
#[must_use]
pub const fn with_wired_outside_mlx_bytes(
    reading: SystemWiredReading,
    wired_outside_mlx_bytes: u64,
) -> SystemWiredReading {
    SystemWiredReading {
        system_wired_bytes: reading
            .mlx_held_bytes
            .saturating_add(wired_outside_mlx_bytes),
        ..reading
    }
}

#[cfg(test)]
#[path = "system_wired_ceiling_tests.rs"]
mod tests;
