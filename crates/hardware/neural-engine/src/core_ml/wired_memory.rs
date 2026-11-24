//! Reading what running programs cost in wired memory.
//!
//! The driver wires a program's weights when it first runs, not when it loads, and
//! releases them 10 to 20 seconds after its last use (issue #161). Reading the system
//! across a whole set's first runs measures that set on this Mac, this macOS and this
//! weight form, so no ratio has to be assumed for any of them.

unsafe extern "C" {
    // native/system_wired_memory.c: zero when the kernel query fails.
    fn astronomical_system_wired_bytes() -> u64;
}

/// The whole system's wired memory; zero when the kernel query fails.
#[must_use]
pub fn system_wired_bytes() -> u64 {
    // SAFETY: the reader takes no arguments and only queries the kernel.
    unsafe { astronomical_system_wired_bytes() }
}

/// How much two readings grew. Memory can also be returned while a program runs, and
/// a failed query reads as zero, so any fall counts as no growth.
#[must_use]
pub const fn wired_growth_bytes(wired_before_bytes: u64, wired_after_bytes: u64) -> u64 {
    wired_after_bytes.saturating_sub(wired_before_bytes)
}

#[cfg(test)]
#[path = "wired_memory_tests.rs"]
mod tests;
