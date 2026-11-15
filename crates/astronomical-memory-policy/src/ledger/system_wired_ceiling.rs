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

#[cfg(test)]
#[path = "system_wired_ceiling_tests.rs"]
mod tests;
