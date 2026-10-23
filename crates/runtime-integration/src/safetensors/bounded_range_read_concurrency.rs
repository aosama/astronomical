//! Bounds parallel positional reads for large virtual SafeTensors intervals.
//!
//! # Why parallelism lives here and nowhere else
//!
//! The bounded reader must borrow MLX-owned destination storage (the raw
//! pointer MLX passed to the vtable callback), so it cannot copy into an
//! intermediate buffer and hand that to a thread pool. Instead, scoped
//! workers partition one large contiguous destination slice and read
//! positionally straight into it — no intermediate copy, and `thread::scope`
//! guarantees every worker has exited before the callback returns the
//! destination to MLX.
//!
//! # The measured numbers behind the constants
//!
//! Both constants below are conclusions, not guesses (measured on Apple
//! Silicon with direct-MLX streaming probes):
//!
//! - `PARALLEL_READ_MINIMUM_BYTES` (32 MiB, MLX's own threshold with four
//!   readers): matching the boundary avoids paying thread setup for routed
//!   expert pages while retaining large-layer parallelism.
//! - `MAXIMUM_CONCURRENT_POSITIONAL_READ_COUNT` = 16: four concurrent
//!   positional reads cap unpressured expert streaming at roughly 14 GB/s
//!   while sixteen reach roughly 23 GB/s, and under seated-expert memory
//!   pressure the extra permits overlap operating-system-stalled reads,
//!   cutting a paged prefill chunk's wall time by about a fifth. Physical
//!   SSDs have far deeper queues than four.
//!
//! # The process-wide permit pool
//!
//! MLX may evaluate several lazy tensors concurrently, so per-read
//! partitioning alone does not bound physical parallelism. A single
//! process-wide limiter (a mutex-and-condvar permit pool) caps concurrent
//! positional reads across *all* readers, keeping the queue depth the SSD
//! sees within the measured sweet spot even during multi-tensor evaluation.
//! `ASTRONOMICAL_POSITIONAL_READ_PARALLELISM` overrides the process default
//! within a bounded range (1..=32) for constrained machines — the default
//! itself stays the measured sixteen-permit ceiling (further clamped by the
//! machine's available parallelism so tiny containers never oversubscribe).

use std::{
    fs::File,
    os::unix::fs::FileExt,
    sync::{Condvar, Mutex, MutexGuard, OnceLock},
    thread,
};

use crate::safetensors::read_metrics::PositionalFileReadMetrics;

// MLX's path reader uses this threshold and four readers. Matching the threshold
// boundary avoids thread setup for routed expert pages while retaining
// large-layer parallelism.
const PARALLEL_READ_MINIMUM_BYTES: usize = 1 << 25;
// See the module documentation for the measurements behind this value.
const MAXIMUM_CONCURRENT_POSITIONAL_READ_COUNT: usize = 16;
// SSDs differ widely in queue depth, so the process cap is an adaptation
// parameter, not a constant: `ASTRONOMICAL_POSITIONAL_READ_PARALLELISM`
// overrides it within a bounded range; the default stays at the measured
// sixteen-permit ceiling above.
const POSITIONAL_READ_PARALLELISM_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_POSITIONAL_READ_PARALLELISM";
const POSITIONAL_READ_PARALLELISM_OVERRIDE_CEILING: usize = 32;
static CONFIGURED_POSITIONAL_READ_PARALLELISM: OnceLock<usize> = OnceLock::new();
static POSITIONAL_READ_CONCURRENCY_LIMITER: OnceLock<PositionalReadConcurrencyLimiter> =
    OnceLock::new();

/// Mutex-and-condvar permit pool bounding concurrent positional reads.
///
/// Chosen over a semaphore crate to stay dependency-free inside a callback
/// path that MLX may invoke from any thread.
struct PositionalReadConcurrencyLimiter {
    active_read_count: Mutex<usize>,
    read_finished: Condvar,
    maximum_concurrent_read_count: usize,
}

/// RAII permit: released on drop, waking one waiting reader.
struct PositionalReadPermit<'a> {
    limiter: &'a PositionalReadConcurrencyLimiter,
}

impl PositionalReadConcurrencyLimiter {
    fn new(maximum_concurrent_read_count: usize) -> Self {
        Self {
            active_read_count: Mutex::new(0),
            read_finished: Condvar::new(),
            maximum_concurrent_read_count,
        }
    }

    /// Blocks until a permit is available.
    ///
    /// Poisoned-lock recovery mirrors the rest of the crate: a panic in one
    /// reader must not wedge every future read behind a poisoned count.
    fn acquire(&self) -> PositionalReadPermit<'_> {
        let mut active_read_count = lock_after_possible_poison(&self.active_read_count);
        while *active_read_count >= self.maximum_concurrent_read_count {
            active_read_count = self
                .read_finished
                .wait(active_read_count)
                .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner());
        }
        *active_read_count += 1;
        PositionalReadPermit { limiter: self }
    }
}

impl Drop for PositionalReadPermit<'_> {
    fn drop(&mut self) {
        let mut active_read_count = lock_after_possible_poison(&self.limiter.active_read_count);
        debug_assert!(*active_read_count > 0);
        *active_read_count = active_read_count.saturating_sub(1);
        drop(active_read_count);
        self.limiter.read_finished.notify_one();
    }
}

/// Reads one source interval, partitioning large spans across scoped workers.
///
/// Returns `true` only when every partition read exactly; a single failed
/// partition fails the whole interval so the caller can latch its error
/// state. Small spans take the direct path — the partitioning and permit
/// machinery would cost more than the read itself.
pub(super) fn read_source_interval(
    source_file: &File,
    destination: &mut [u8],
    source_offset: u64,
    positional_file_read_metrics: Option<&PositionalFileReadMetrics>,
) -> bool {
    let read_partition_count = destination
        .len()
        .div_ceil(PARALLEL_READ_MINIMUM_BYTES)
        .min(configured_read_parallelism());
    if read_partition_count <= 1 {
        return measure_source_read(
            source_file,
            destination,
            source_offset,
            positional_file_read_metrics,
        );
    }

    let bytes_per_partition = destination.len().div_ceil(read_partition_count);
    thread::scope(|scope| {
        // Each worker reads its slice positionally; `scope` joins them all
        // before the destination escapes this function back to the vtable
        // caller, which is what makes borrowing MLX's storage sound.
        let partition_read_tasks = destination
            .chunks_mut(bytes_per_partition)
            .enumerate()
            .map(|(partition_index, destination_partition)| {
                let partition_source_offset =
                    source_offset + (partition_index * bytes_per_partition) as u64;
                scope.spawn(move || {
                    measure_source_read(
                        source_file,
                        destination_partition,
                        partition_source_offset,
                        positional_file_read_metrics,
                    )
                })
            })
            .collect::<Vec<_>>();
        partition_read_tasks
            .into_iter()
            .fold(true, |all_reads_succeeded, read_task| {
                read_task.join().unwrap_or(false) && all_reads_succeeded
            })
    })
}

/// Performs one permit-bounded positional read with metrics.
///
/// A panicked worker is reported as a failed read (`unwrap_or(false)`) rather
/// than propagating the panic across the C boundary.
fn measure_source_read(
    source_file: &File,
    destination: &mut [u8],
    source_offset: u64,
    positional_file_read_metrics: Option<&PositionalFileReadMetrics>,
) -> bool {
    let concurrency_limiter = POSITIONAL_READ_CONCURRENCY_LIMITER
        .get_or_init(|| PositionalReadConcurrencyLimiter::new(configured_read_parallelism()));
    let _positional_read_permit = concurrency_limiter.acquire();
    let destination_byte_count = destination.len();
    let mut read_operation = || {
        source_file
            .read_exact_at(destination, source_offset)
            .is_ok()
    };
    match positional_file_read_metrics {
        Some(read_metrics) => read_metrics.measure_read(destination_byte_count, read_operation),
        None => read_operation(),
    }
}

/// Resolves the process read-parallelism default once.
///
/// Precedence: the environment override (clamped to 1..=32), else the
/// machine's available parallelism clamped by the measured sixteen-permit
/// ceiling. Memoized in a `OnceLock` so every callback after the first reads
/// one atomic-free value — and so a mid-process environment change cannot
/// split readers across different limiter sizes.
fn configured_read_parallelism() -> usize {
    *CONFIGURED_POSITIONAL_READ_PARALLELISM.get_or_init(|| {
        let default_parallelism = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAXIMUM_CONCURRENT_POSITIONAL_READ_COUNT);
        std::env::var(POSITIONAL_READ_PARALLELISM_ENVIRONMENT_VARIABLE)
            .ok()
            .and_then(|parallelism_text| parallelism_text.parse::<usize>().ok())
            .map(|override_parallelism| {
                override_parallelism.clamp(1, POSITIONAL_READ_PARALLELISM_OVERRIDE_CEILING)
            })
            .unwrap_or(default_parallelism)
    })
}

fn lock_after_possible_poison(mutex: &Mutex<usize>) -> MutexGuard<'_, usize> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
}
