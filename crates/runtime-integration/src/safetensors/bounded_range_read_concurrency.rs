//! Bounds parallel positional reads for virtual SafeTensors spans.
//!
//! # Where read parallelism lives
//!
//! The bounded reader must borrow MLX-owned destination storage (the raw
//! pointer MLX passed to the vtable callback), so it cannot copy into an
//! intermediate buffer first. Every worker therefore reads positionally
//! straight into a disjoint slice of that destination, and the callback does
//! not return until every worker is done with it. Two shapes of parallelism
//! follow:
//!
//! - **One large interval** is partitioned across scoped workers
//!   (`read_source_interval`); `thread::scope` joins them before returning.
//! - **One span made of many intervals** — MLX asks for one span per tensor,
//!   and a routed page's span holds one interval per run of selected
//!   experts — has its pieces read by the calling thread plus long-lived
//!   threads of `piece_read_pool` (`read_source_pieces`), whose completion
//!   guard waits for every piece before returning.
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
//! The permit count follows the machine's available parallelism, bounded
//! between one and the measured sixteen-permit ceiling so constrained
//! processes never oversubscribe.

use std::{
    fs::File,
    os::unix::fs::FileExt,
    sync::{Condvar, Mutex, MutexGuard, OnceLock},
    thread,
};

pub(super) use super::piece_read_pool::SourcePiece;
use super::piece_read_pool::{PieceReadPool, read_pieces_with_pool_helpers};
use crate::safetensors::read_metrics::PositionalFileReadMetrics;

// MLX's path reader uses this threshold and four readers. Matching the threshold
// boundary keeps one routed interval on a single read while retaining
// large-layer parallelism.
const PARALLEL_READ_MINIMUM_BYTES: usize = 1 << 25;
// See the module documentation for the measurements behind this value.
const MAXIMUM_CONCURRENT_POSITIONAL_READ_COUNT: usize = 16;
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
/// state, and a panicked partition worker is reported as a failed read
/// (`unwrap_or(false)`) rather than propagated. Small spans take the direct
/// path — the partitioning and permit machinery would cost more than the read
/// itself.
pub(super) fn read_source_interval(
    source_file: &File,
    destination: &mut [u8],
    source_offset: u64,
    positional_file_read_metrics: Option<&PositionalFileReadMetrics>,
) -> bool {
    let read_partition_count = destination
        .len()
        .div_ceil(PARALLEL_READ_MINIMUM_BYTES)
        .min(positional_read_concurrency_limiter().maximum_concurrent_read_count);
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
        // fold, not all(): every spawned read task must be joined even when
        // one already failed — all() short-circuits and would leak the
        // remaining threads unjoined.
        #[allow(clippy::unnecessary_fold)]
        partition_read_tasks
            .into_iter()
            .fold(true, |all_reads_succeeded, read_task| {
                read_task.join().unwrap_or(false) && all_reads_succeeded
            })
    })
}

/// Performs one permit-bounded positional read with metrics.
fn measure_source_read(
    source_file: &File,
    destination: &mut [u8],
    source_offset: u64,
    positional_file_read_metrics: Option<&PositionalFileReadMetrics>,
) -> bool {
    let _positional_read_permit = positional_read_concurrency_limiter().acquire();
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

/// The long-lived threads that help read multi-piece spans, started on first
/// use with one thread per read permit.
static PIECE_READ_POOL: PieceReadPool = PieceReadPool::new();

/// Reads every piece of one virtual span: on the calling thread alone for a
/// single piece, otherwise with up to one long-lived pool thread per further
/// piece, capped by the permit pool and by the threads the pool started.
///
/// MLX's Load hands the reader one span per tensor; for a routed page it is
/// one interval per selected expert. Reading those one after another kept the
/// SSD at a queue depth of one per tensor. Measured through MLX on one
/// 8-expert page (`cargo measure-cold-expert-page-reads`), read from the SSD
/// and from the file cache respectively:
///
/// | Model, page size | Serial | This reader |
/// |---|---|---|
/// | 35B A3B 8-bit, 26.7 MB | 4.40 / 0.69 ms | 2.77 / 0.49 ms |
/// | 122B 2-bit experts, 23.6 MB | 4.18 / 0.65 ms | 2.45 / 0.47 ms |
///
/// The pool threads are why the cached case improves too; see
/// `piece_read_pool`. Every piece is attempted even after a failure, and the
/// call returns only when every piece has finished; the result is `true` only
/// when all pieces read exactly.
pub(super) fn read_source_pieces(
    source_file: &File,
    source_pieces: Vec<SourcePiece<'_>>,
    positional_file_read_metrics: Option<&PositionalFileReadMetrics>,
) -> bool {
    let worker_count = piece_worker_count(
        source_pieces.len(),
        positional_read_concurrency_limiter().maximum_concurrent_read_count,
    );
    if worker_count <= 1 {
        let mut all_pieces_read = true;
        for source_piece in source_pieces {
            all_pieces_read &= read_source_interval(
                source_file,
                source_piece.destination,
                source_piece.source_file_offset,
                positional_file_read_metrics,
            );
        }
        return all_pieces_read;
    }
    PIECE_READ_POOL.start_once(positional_read_concurrency_limiter().maximum_concurrent_read_count);
    read_pieces_with_pool_helpers(
        &PIECE_READ_POOL,
        source_file,
        source_pieces,
        positional_file_read_metrics,
        worker_count - 1,
        read_source_interval,
    )
}

/// Workers for one span, the caller included: one per piece, but never more
/// than the permit pool could serve at once. Once workers were long-lived,
/// one per piece measured slightly faster than one per 64 KiB of span on both
/// models above (2.45 against 2.55 ms read from the SSD on the 122B) and the
/// same from the file cache, so no size rule remains. Zero only for an empty
/// span.
fn piece_worker_count(piece_count: usize, permit_count: usize) -> usize {
    piece_count.min(permit_count)
}

fn positional_read_concurrency_limiter() -> &'static PositionalReadConcurrencyLimiter {
    POSITIONAL_READ_CONCURRENCY_LIMITER.get_or_init(|| {
        let available_parallelism = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        PositionalReadConcurrencyLimiter::new(automatic_read_parallelism(available_parallelism))
    })
}

fn automatic_read_parallelism(available_parallelism: usize) -> usize {
    available_parallelism.clamp(1, MAXIMUM_CONCURRENT_POSITIONAL_READ_COUNT)
}

fn lock_after_possible_poison<Guarded>(mutex: &Mutex<Guarded>) -> MutexGuard<'_, Guarded> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
}

#[cfg(test)]
mod bounded_range_read_concurrency_tests {
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::time::Duration;

    use std::io::Write;

    use super::{
        PositionalReadConcurrencyLimiter, SourcePiece, automatic_read_parallelism,
        lock_after_possible_poison, piece_worker_count, positional_read_concurrency_limiter,
        read_source_pieces,
    };

    #[test]
    fn should_start_no_worker_for_an_empty_span() {
        assert_eq!(piece_worker_count(0, 16), 0);
    }

    #[test]
    fn should_read_a_single_piece_on_the_calling_thread_alone() {
        assert_eq!(piece_worker_count(1, 16), 1);
    }

    #[test]
    fn should_use_one_worker_per_piece_up_to_the_permit_count() {
        assert_eq!(piece_worker_count(8, 16), 8);
        assert_eq!(piece_worker_count(16, 16), 16);
        assert_eq!(piece_worker_count(64, 16), 16);
    }

    const PIECE_BYTES: usize = 64 * 1024;

    /// A source file whose every byte names its 64 KiB block, so a misplaced
    /// piece shows up as the wrong block number.
    fn numbered_block_source(block_count: usize) -> tempfile::NamedTempFile {
        let mut source_file = tempfile::NamedTempFile::new().expect("a scratch source file");
        for block_index in 0..block_count {
            source_file
                .write_all(&vec![block_index as u8; PIECE_BYTES])
                .expect("the source block writes");
        }
        source_file.flush().expect("the source flushes");
        source_file
    }

    /// Reads source blocks in reverse into consecutive destination slices.
    fn read_blocks_in_reverse(
        source_file: &std::fs::File,
        block_count: usize,
        destination: &mut [u8],
    ) -> bool {
        let source_pieces = destination
            .as_chunks_mut::<PIECE_BYTES>()
            .0
            .iter_mut()
            .enumerate()
            .map(|(destination_block_index, piece_destination)| SourcePiece {
                destination: piece_destination,
                source_file_offset: ((block_count - 1 - destination_block_index) * PIECE_BYTES)
                    as u64,
            })
            .collect();
        read_source_pieces(source_file, source_pieces, None)
    }

    #[test]
    fn should_place_every_piece_when_pieces_outnumber_read_permits() {
        let block_count = 64;
        let source_file = numbered_block_source(block_count);
        let mut destination = vec![0u8; block_count * PIECE_BYTES];
        assert!(read_blocks_in_reverse(
            source_file.as_file(),
            block_count,
            &mut destination
        ));
        for (destination_block_index, destination_block) in
            destination.as_chunks::<PIECE_BYTES>().0.iter().enumerate()
        {
            let expected_block_number = (block_count - 1 - destination_block_index) as u8;
            assert!(
                destination_block
                    .iter()
                    .all(|&block_number| block_number == expected_block_number)
            );
        }
    }

    #[test]
    fn should_place_every_piece_of_a_small_span() {
        let block_count = 4;
        let source_file = numbered_block_source(block_count);
        let mut destination = vec![0u8; block_count * PIECE_BYTES];
        assert!(read_blocks_in_reverse(
            source_file.as_file(),
            block_count,
            &mut destination
        ));
        assert_eq!(destination[0], 3);
        assert_eq!(destination[3 * PIECE_BYTES], 0);
    }

    #[test]
    fn should_report_success_for_an_empty_span() {
        let source_file = numbered_block_source(1);
        assert!(read_source_pieces(source_file.as_file(), Vec::new(), None));
    }

    #[test]
    fn should_fail_the_span_but_read_every_other_piece_when_one_piece_is_past_the_end() {
        let block_count = 32;
        let source_file = numbered_block_source(block_count);
        let mut destination = vec![0xAAu8; block_count * PIECE_BYTES];
        let mut source_pieces: Vec<SourcePiece<'_>> = destination
            .as_chunks_mut::<PIECE_BYTES>()
            .0
            .iter_mut()
            .enumerate()
            .map(|(block_index, piece_destination)| SourcePiece {
                destination: piece_destination,
                source_file_offset: (block_index * PIECE_BYTES) as u64,
            })
            .collect();
        source_pieces[5].source_file_offset = (block_count * PIECE_BYTES) as u64;
        assert!(!read_source_pieces(
            source_file.as_file(),
            source_pieces,
            None
        ));
        assert_eq!(
            destination[31 * PIECE_BYTES],
            31,
            "pieces after the failed one are still read"
        );
        assert_eq!(
            destination[4 * PIECE_BYTES],
            4,
            "pieces before the failed one are read"
        );
    }

    #[test]
    fn should_follow_available_parallelism_below_the_ceiling() {
        for available_parallelism in [1, 2, 8, 16] {
            assert_eq!(
                automatic_read_parallelism(available_parallelism),
                available_parallelism
            );
        }
    }

    #[test]
    fn should_keep_at_least_one_positional_read_worker() {
        assert_eq!(automatic_read_parallelism(0), 1);
    }

    #[test]
    fn should_cap_positional_read_parallelism_at_sixteen() {
        for available_parallelism in [17, 32, usize::MAX] {
            assert_eq!(automatic_read_parallelism(available_parallelism), 16);
        }
    }

    #[test]
    fn should_share_one_bounded_limiter_across_readers() {
        let first_reader_limiter = positional_read_concurrency_limiter();
        let second_reader_limiter = positional_read_concurrency_limiter();
        assert!(std::ptr::eq(first_reader_limiter, second_reader_limiter));
        assert!((1..=16).contains(&first_reader_limiter.maximum_concurrent_read_count));
    }

    #[test]
    fn should_return_a_read_permit_when_the_read_finishes() {
        let limiter = PositionalReadConcurrencyLimiter::new(1);
        let read_permit = limiter.acquire();
        assert_eq!(*lock_after_possible_poison(&limiter.active_read_count), 1);
        drop(read_permit);
        assert_eq!(*lock_after_possible_poison(&limiter.active_read_count), 0);
        let _next_read_permit = limiter.acquire();
        assert_eq!(*lock_after_possible_poison(&limiter.active_read_count), 1);
    }

    #[test]
    fn should_wait_for_a_permit_when_the_read_limit_is_reached() {
        let limiter = PositionalReadConcurrencyLimiter::new(1);
        let held_read_permit = limiter.acquire();
        let (permit_acquired_sender, permit_acquired_receiver) = mpsc::channel();

        std::thread::scope(|scope| {
            let waiting_reader_limiter = &limiter;
            scope.spawn(move || {
                let _read_permit = waiting_reader_limiter.acquire();
                permit_acquired_sender
                    .send(())
                    .expect("waiting reader could not report its acquired permit");
            });
            let waiting_reader_was_blocked = matches!(
                permit_acquired_receiver.recv_timeout(Duration::from_millis(20)),
                Err(RecvTimeoutError::Timeout)
            );
            drop(held_read_permit);
            assert!(waiting_reader_was_blocked);
            permit_acquired_receiver
                .recv_timeout(Duration::from_secs(5))
                .expect("waiting reader did not resume after a permit was returned");
        });
        assert_eq!(*lock_after_possible_poison(&limiter.active_read_count), 0);
    }
}
