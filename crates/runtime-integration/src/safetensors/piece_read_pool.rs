//! A pool of long-lived threads that help read the pieces of a bounded span.
//!
//! # Why a long-lived pool
//!
//! Starting scoped threads for every span cost more than it saved on pages
//! the file cache still holds: on a 122B 2-bit model, an 8-expert page read
//! from the cache went from 0.65 ms (serial) to 0.80 ms with freshly started
//! workers, while the same workers took the page read from the SSD from 4.18
//! to 2.64 ms. Waking a parked thread is far cheaper than starting one: with
//! this pool the same page takes 0.47 ms from the cache and 2.45 ms from the
//! SSD.
//!
//! # Why raw pointers are sound here
//!
//! A span's pieces borrow MLX's destination storage, which lives only for
//! the reader callback, while the pool's threads live for the process. A
//! pool thread touches a piece's pointers only after claiming that piece, and
//! the caller does not return until every claimed piece has finished:
//! `SpanCompletionGuard` withdraws the span's unused tickets, gives up any
//! piece nobody claimed (marking the span failed), and then waits for the
//! pieces already being read. `PieceFinishGuard` finishes a piece even when
//! its read panics, so that wait always ends, and a pool thread that panics
//! keeps serving.

use std::collections::VecDeque;
use std::fs::File;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Once, PoisonError};
use std::thread;

use crate::safetensors::read_metrics::PositionalFileReadMetrics;

/// Pool threads only park and issue positional reads; large reads partition
/// onto scoped threads of their own, so a small stack is plenty.
const POOL_THREAD_STACK_BYTES: usize = 512 * 1024;

/// One source read that fills one disjoint slice of MLX's destination.
pub(super) struct SourcePiece<'destination> {
    pub(super) destination: &'destination mut [u8],
    pub(super) source_file_offset: u64,
}

/// Reads one piece; `read_source_interval` in production. A plain function
/// pointer keeps this pool free of the concurrency module and lets tests
/// inject failures.
pub(super) type PieceReadFunction =
    fn(&File, &mut [u8], u64, Option<&PositionalFileReadMetrics>) -> bool;

/// One piece as raw parts, so it can cross into a long-lived thread.
struct RawPiece {
    destination_address: *mut u8,
    byte_count: usize,
    source_file_offset: u64,
}

/// What the helpers of one span share. A stale ticket can keep this alive
/// after its span returned; its pointers are then never read, because no
/// unclaimed piece is left.
struct SpanReadWork {
    unclaimed_pieces: Mutex<VecDeque<RawPiece>>,
    unfinished_piece_count: Mutex<usize>,
    piece_finished: Condvar,
    all_pieces_read: AtomicBool,
    source_file: *const File,
    /// Null when the span records no metrics.
    positional_file_read_metrics: *const PositionalFileReadMetrics,
    read_piece: PieceReadFunction,
}

// SAFETY: the raw pointers are dereferenced only for a claimed piece, and the
// span's caller keeps every pointee alive until all claimed pieces finish.
unsafe impl Send for SpanReadWork {}
// SAFETY: shared access goes through the mutexes and the atomic above.
unsafe impl Sync for SpanReadWork {}

/// Parked threads waiting for span tickets. `const`-constructible so the
/// production pool is a plain `static` and tests can build private pools.
pub(super) struct PieceReadPool {
    pending_span_tickets: Mutex<VecDeque<Arc<SpanReadWork>>>,
    span_ticket_available: Condvar,
    started_worker_count: AtomicUsize,
    start_once: Once,
}

/// Reads `source_pieces` on the calling thread plus up to `helper_count`
/// threads of `piece_read_pool`, returning `true` only when every piece read
/// exactly.
pub(super) fn read_pieces_with_pool_helpers(
    piece_read_pool: &'static PieceReadPool,
    source_file: &File,
    source_pieces: Vec<SourcePiece<'_>>,
    positional_file_read_metrics: Option<&PositionalFileReadMetrics>,
    helper_count: usize,
    read_piece: PieceReadFunction,
) -> bool {
    let piece_count = source_pieces.len();
    let span_work = Arc::new(SpanReadWork {
        unclaimed_pieces: Mutex::new(source_pieces.into_iter().map(raw_piece).collect()),
        unfinished_piece_count: Mutex::new(piece_count),
        piece_finished: Condvar::new(),
        all_pieces_read: AtomicBool::new(true),
        source_file: std::ptr::from_ref(source_file),
        positional_file_read_metrics: positional_file_read_metrics
            .map_or(std::ptr::null(), std::ptr::from_ref),
        read_piece,
    });
    let completion_guard = SpanCompletionGuard {
        piece_read_pool,
        span_work: &span_work,
    };
    piece_read_pool.post_span_tickets(&span_work, helper_count);
    drain_span_pieces(&span_work);
    drop(completion_guard);
    span_work.all_pieces_read.load(Ordering::Acquire)
}

fn raw_piece(source_piece: SourcePiece<'_>) -> RawPiece {
    RawPiece {
        destination_address: source_piece.destination.as_mut_ptr(),
        byte_count: source_piece.destination.len(),
        source_file_offset: source_piece.source_file_offset,
    }
}

/// Ends a span on the caller's side: withdraws its unused tickets, gives up
/// unclaimed pieces (only possible while the caller unwinds), and waits for
/// the pieces pool threads are still reading.
struct SpanCompletionGuard<'span> {
    piece_read_pool: &'static PieceReadPool,
    span_work: &'span Arc<SpanReadWork>,
}

impl Drop for SpanCompletionGuard<'_> {
    fn drop(&mut self) {
        self.piece_read_pool.withdraw_span_tickets(self.span_work);
        let abandoned_piece_count = {
            let mut unclaimed_pieces = lock_after_possible_poison(&self.span_work.unclaimed_pieces);
            let abandoned_piece_count = unclaimed_pieces.len();
            unclaimed_pieces.clear();
            abandoned_piece_count
        };
        let mut unfinished_piece_count =
            lock_after_possible_poison(&self.span_work.unfinished_piece_count);
        if abandoned_piece_count > 0 {
            self.span_work
                .all_pieces_read
                .store(false, Ordering::Release);
            *unfinished_piece_count -= abandoned_piece_count;
        }
        while *unfinished_piece_count > 0 {
            unfinished_piece_count = self
                .span_work
                .piece_finished
                .wait(unfinished_piece_count)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// Finishes one claimed piece on drop, even when its read panicked, so the
/// caller's wait always ends.
struct PieceFinishGuard<'span> {
    span_work: &'span SpanReadWork,
}

impl Drop for PieceFinishGuard<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.span_work
                .all_pieces_read
                .store(false, Ordering::Release);
        }
        let mut unfinished_piece_count =
            lock_after_possible_poison(&self.span_work.unfinished_piece_count);
        *unfinished_piece_count -= 1;
        if *unfinished_piece_count == 0 {
            // Only the span's caller ever waits on this condition.
            self.span_work.piece_finished.notify_one();
        }
    }
}

/// Claims and reads the span's pieces until none are left unclaimed.
fn drain_span_pieces(span_work: &SpanReadWork) {
    loop {
        let Some(claimed_piece) =
            lock_after_possible_poison(&span_work.unclaimed_pieces).pop_front()
        else {
            return;
        };
        let _piece_finish_guard = PieceFinishGuard { span_work };
        // SAFETY: the piece was claimed, so its span's caller is still
        // waiting and keeps the destination, file and metrics alive; no other
        // piece aliases this destination range.
        let piece_was_read = unsafe {
            (span_work.read_piece)(
                &*span_work.source_file,
                std::slice::from_raw_parts_mut(
                    claimed_piece.destination_address,
                    claimed_piece.byte_count,
                ),
                claimed_piece.source_file_offset,
                span_work.positional_file_read_metrics.as_ref(),
            )
        };
        if !piece_was_read {
            span_work.all_pieces_read.store(false, Ordering::Release);
        }
    }
}

impl PieceReadPool {
    pub(super) const fn new() -> Self {
        Self {
            pending_span_tickets: Mutex::new(VecDeque::new()),
            span_ticket_available: Condvar::new(),
            started_worker_count: AtomicUsize::new(0),
            start_once: Once::new(),
        }
    }

    /// Starts `worker_count` threads the first time it is called. A thread
    /// the operating system refuses is simply absent; with none started,
    /// callers read alone.
    pub(super) fn start_once(&'static self, worker_count: usize) {
        self.start_once.call_once(|| {
            for worker_index in 0..worker_count {
                let worker_spawn_outcome = thread::Builder::new()
                    .name(format!("astronomical-piece-reader-{worker_index}"))
                    .stack_size(POOL_THREAD_STACK_BYTES)
                    .spawn(move || run_pool_worker(self));
                if worker_spawn_outcome.is_ok() {
                    self.started_worker_count.fetch_add(1, Ordering::AcqRel);
                }
            }
        });
    }

    #[cfg(test)]
    pub(super) fn started_worker_count(&self) -> usize {
        self.started_worker_count.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(super) fn pending_span_ticket_count(&self) -> usize {
        lock_after_possible_poison(&self.pending_span_tickets).len()
    }

    /// Posts at most one ticket per started thread across all spans, and
    /// wakes one parked thread per ticket.
    fn post_span_tickets(&self, span_work: &Arc<SpanReadWork>, helper_count: usize) {
        let mut pending_span_tickets = lock_after_possible_poison(&self.pending_span_tickets);
        let ticket_count = helper_count.min(
            self.started_worker_count
                .load(Ordering::Acquire)
                .saturating_sub(pending_span_tickets.len()),
        );
        for _ in 0..ticket_count {
            pending_span_tickets.push_back(Arc::clone(span_work));
        }
        drop(pending_span_tickets);
        for _ in 0..ticket_count {
            self.span_ticket_available.notify_one();
        }
    }

    /// Drops the span's tickets no thread took, so later spans' tickets are
    /// not queued behind ones with nothing left to read.
    fn withdraw_span_tickets(&self, span_work: &Arc<SpanReadWork>) {
        lock_after_possible_poison(&self.pending_span_tickets)
            .retain(|pending_ticket| !Arc::ptr_eq(pending_ticket, span_work));
    }

    fn next_span_ticket(&self) -> Arc<SpanReadWork> {
        let mut pending_span_tickets = lock_after_possible_poison(&self.pending_span_tickets);
        loop {
            if let Some(span_ticket) = pending_span_tickets.pop_front() {
                return span_ticket;
            }
            pending_span_tickets = self
                .span_ticket_available
                .wait(pending_span_tickets)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// A read that panics has already failed its span through
/// `PieceFinishGuard`; the thread keeps serving so the pool never shrinks.
fn run_pool_worker(piece_read_pool: &'static PieceReadPool) {
    loop {
        let span_ticket = piece_read_pool.next_span_ticket();
        let _drain_outcome = catch_unwind(AssertUnwindSafe(|| drain_span_pieces(&span_ticket)));
    }
}

fn lock_after_possible_poison<Guarded>(mutex: &Mutex<Guarded>) -> MutexGuard<'_, Guarded> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
#[path = "piece_read_pool_tests.rs"]
mod piece_read_pool_tests;
