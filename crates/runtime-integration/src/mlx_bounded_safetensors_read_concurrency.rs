//! Bounds parallel positional reads for large virtual SafeTensors intervals.
//!
//! The custom SafeTensors reader must borrow MLX-owned destination storage, so scoped
//! workers partition large contiguous intervals without allocating an intermediate copy.
//! One process-wide permit pool bounds physical reads even when MLX evaluates several
//! tensors concurrently.

use std::{
    fs::File,
    os::unix::fs::FileExt,
    sync::{Condvar, Mutex, MutexGuard, OnceLock},
    thread,
};

use crate::PositionalFileReadMetrics;

// MLX's path reader uses this threshold and four readers. Matching that boundary
// avoids thread setup for routed expert pages while retaining large-layer parallelism.
const PARALLEL_READ_MINIMUM_BYTES: usize = 1 << 25;
const MAXIMUM_CONCURRENT_POSITIONAL_READ_COUNT: usize = 4;
static CONFIGURED_POSITIONAL_READ_PARALLELISM: OnceLock<usize> = OnceLock::new();
static POSITIONAL_READ_CONCURRENCY_LIMITER: OnceLock<PositionalReadConcurrencyLimiter> =
    OnceLock::new();

struct PositionalReadConcurrencyLimiter {
    active_read_count: Mutex<usize>,
    read_finished: Condvar,
    maximum_concurrent_read_count: usize,
}

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

fn configured_read_parallelism() -> usize {
    *CONFIGURED_POSITIONAL_READ_PARALLELISM.get_or_init(|| {
        thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAXIMUM_CONCURRENT_POSITIONAL_READ_COUNT)
    })
}

fn lock_after_possible_poison(mutex: &Mutex<usize>) -> MutexGuard<'_, usize> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
}
