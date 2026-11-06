//! How much memory a Core ML projection costs once loaded.
//!
//! The question behind the 4-bit ML Program is whether the Neural Engine's
//! share of the weights stops costing four times its checkpoint bytes, so the
//! evidence is the growth of the process footprint across a load, with the
//! system wired count beside it as a cross-check.

unsafe extern "C" {
    // native/process_memory.c: zero when the kernel query fails.
    fn astronomical_process_physical_footprint_bytes() -> u64;
    fn astronomical_physical_memory_bytes() -> u64;
    fn astronomical_system_reclaimable_bytes() -> u64;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub process_physical_footprint_bytes: u64,
    pub system_wired_bytes: u64,
}

impl MemorySnapshot {
    #[must_use]
    pub fn capture() -> Self {
        // SAFETY: the reader takes no arguments and only queries the kernel.
        unsafe {
            Self {
                process_physical_footprint_bytes: astronomical_process_physical_footprint_bytes(),
                system_wired_bytes: astronomical_neural_engine::core_ml::system_wired_bytes(),
            }
        }
    }
}

/// This Mac's installed memory, read at run time so no journey assumes a
/// particular machine; zero when the kernel query fails.
#[must_use]
pub fn physical_memory_bytes() -> u64 {
    // SAFETY: the reader takes no arguments and only queries the kernel.
    unsafe { astronomical_physical_memory_bytes() }
}

/// Memory the system could hand out without paging; zero when the kernel query fails.
#[must_use]
pub fn system_reclaimable_bytes() -> u64 {
    // SAFETY: the reader takes no arguments and only queries the kernel.
    unsafe { astronomical_system_reclaimable_bytes() }
}

/// Signed growth between two snapshots; memory can also be returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryGrowth {
    pub process_physical_footprint_bytes: i64,
    pub system_wired_bytes: i64,
}

impl MemoryGrowth {
    #[must_use]
    pub fn between(before: MemorySnapshot, after: MemorySnapshot) -> Self {
        Self {
            process_physical_footprint_bytes: signed_difference(
                before.process_physical_footprint_bytes,
                after.process_physical_footprint_bytes,
            ),
            system_wired_bytes: signed_difference(
                before.system_wired_bytes,
                after.system_wired_bytes,
            ),
        }
    }
}

fn signed_difference(before_bytes: u64, after_bytes: u64) -> i64 {
    // i128 cannot overflow for two u64 inputs; the clamp only matters for
    // readings no Mac can produce.
    (i128::from(after_bytes) - i128::from(before_bytes))
        .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// Bytes in mebibytes, for evidence lines.
#[must_use]
pub fn mebibytes(byte_count: i64) -> f64 {
    byte_count as f64 / (1024.0 * 1024.0)
}

#[cfg(test)]
#[path = "process_memory_tests.rs"]
mod tests;
