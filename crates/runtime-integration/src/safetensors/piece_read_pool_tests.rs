//! The long-lived piece-read pool, exercised on private pools so thread
//! counts, failures and ticket hygiene can be controlled.

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::AtomicBool;
use std::sync::{Condvar, Mutex};

use super::{
    PieceReadFunction, PieceReadPool, SourcePiece, SpanReadWork, drain_span_pieces,
    read_pieces_with_pool_helpers,
};
use crate::safetensors::read_metrics::PositionalFileReadMetrics;

const PIECE_BYTES: usize = 64 * 1024;
const PANICKING_PIECE_OFFSET: u64 = 3 * PIECE_BYTES as u64;

fn private_pool(worker_count: usize) -> &'static PieceReadPool {
    let piece_read_pool: &'static PieceReadPool = Box::leak(Box::new(PieceReadPool::new()));
    piece_read_pool.start_once(worker_count);
    piece_read_pool
}

/// A source file whose every byte names its block, so a misplaced piece
/// shows up as the wrong block number.
fn numbered_block_source(block_count: usize, block_bytes: usize) -> tempfile::NamedTempFile {
    let mut source_file = tempfile::NamedTempFile::new().expect("a scratch source file");
    for block_index in 0..block_count {
        source_file
            .write_all(&vec![block_index as u8; block_bytes])
            .expect("the source block writes");
    }
    source_file.flush().expect("the source flushes");
    source_file
}

fn read_exact_piece(
    source_file: &File,
    destination: &mut [u8],
    source_file_offset: u64,
    _positional_file_read_metrics: Option<&PositionalFileReadMetrics>,
) -> bool {
    source_file
        .read_exact_at(destination, source_file_offset)
        .is_ok()
}

fn fail_every_piece(
    _source_file: &File,
    _destination: &mut [u8],
    _source_file_offset: u64,
    _positional_file_read_metrics: Option<&PositionalFileReadMetrics>,
) -> bool {
    false
}

fn panic_on_one_piece(
    source_file: &File,
    destination: &mut [u8],
    source_file_offset: u64,
    positional_file_read_metrics: Option<&PositionalFileReadMetrics>,
) -> bool {
    assert_ne!(
        source_file_offset, PANICKING_PIECE_OFFSET,
        "an injected read failure"
    );
    read_exact_piece(
        source_file,
        destination,
        source_file_offset,
        positional_file_read_metrics,
    )
}

/// Reads source blocks in reverse into consecutive destination blocks and
/// checks every block landed where it belongs.
fn read_reversed_blocks(
    piece_read_pool: &'static PieceReadPool,
    source_file: &File,
    block_count: usize,
    block_bytes: usize,
    helper_count: usize,
    read_piece: PieceReadFunction,
) -> (bool, Vec<u8>) {
    let mut destination = vec![0xAAu8; block_count * block_bytes];
    let source_pieces = destination
        .chunks_exact_mut(block_bytes)
        .enumerate()
        .map(|(destination_block_index, piece_destination)| SourcePiece {
            destination: piece_destination,
            source_file_offset: ((block_count - 1 - destination_block_index) * block_bytes) as u64,
        })
        .collect();
    let all_pieces_read = read_pieces_with_pool_helpers(
        piece_read_pool,
        source_file,
        source_pieces,
        None,
        helper_count,
        read_piece,
    );
    (all_pieces_read, destination)
}

fn assert_reversed_blocks(destination: &[u8], block_count: usize, block_bytes: usize) {
    for (destination_block_index, destination_block) in
        destination.chunks_exact(block_bytes).enumerate()
    {
        let expected_block_number = (block_count - 1 - destination_block_index) as u8;
        assert!(
            destination_block
                .iter()
                .all(|&block_number| block_number == expected_block_number),
            "destination block {destination_block_index} holds the wrong source block"
        );
    }
}

#[test]
fn should_read_every_piece_on_the_caller_when_the_pool_started_no_threads() {
    let piece_read_pool = private_pool(0);
    let source_file = numbered_block_source(8, PIECE_BYTES);
    let (all_pieces_read, destination) = read_reversed_blocks(
        piece_read_pool,
        source_file.as_file(),
        8,
        PIECE_BYTES,
        7,
        read_exact_piece,
    );
    assert!(all_pieces_read);
    assert_reversed_blocks(&destination, 8, PIECE_BYTES);
    assert_eq!(piece_read_pool.started_worker_count(), 0);
    assert_eq!(piece_read_pool.pending_span_ticket_count(), 0);
}

#[test]
fn should_read_a_two_piece_span_with_one_helper() {
    let piece_read_pool = private_pool(1);
    let source_file = numbered_block_source(2, PIECE_BYTES);
    let (all_pieces_read, destination) = read_reversed_blocks(
        piece_read_pool,
        source_file.as_file(),
        2,
        PIECE_BYTES,
        1,
        read_exact_piece,
    );
    assert!(all_pieces_read);
    assert_reversed_blocks(&destination, 2, PIECE_BYTES);
}

#[test]
fn should_read_every_piece_when_more_helpers_are_requested_than_threads_started() {
    let piece_read_pool = private_pool(2);
    let source_file = numbered_block_source(32, PIECE_BYTES);
    let (all_pieces_read, destination) = read_reversed_blocks(
        piece_read_pool,
        source_file.as_file(),
        32,
        PIECE_BYTES,
        15,
        read_exact_piece,
    );
    assert!(all_pieces_read);
    assert_reversed_blocks(&destination, 32, PIECE_BYTES);
}

#[test]
fn should_leave_no_ticket_queued_after_any_span_returns() {
    let piece_read_pool = private_pool(4);
    let source_file = numbered_block_source(8, PIECE_BYTES);
    for _ in 0..500 {
        let (all_pieces_read, _) = read_reversed_blocks(
            piece_read_pool,
            source_file.as_file(),
            8,
            PIECE_BYTES,
            7,
            read_exact_piece,
        );
        assert!(all_pieces_read);
        assert_eq!(piece_read_pool.pending_span_ticket_count(), 0);
    }
    assert_eq!(piece_read_pool.started_worker_count(), 4);
}

#[test]
fn should_ignore_a_ticket_whose_span_already_finished() {
    // Null pointers stand for a finished span's expired borrows: draining
    // must find no piece and never touch them.
    let finished_span_work = SpanReadWork {
        unclaimed_pieces: Mutex::new(VecDeque::new()),
        unfinished_piece_count: Mutex::new(0),
        piece_finished: Condvar::new(),
        all_pieces_read: AtomicBool::new(true),
        source_file: std::ptr::null(),
        positional_file_read_metrics: std::ptr::null(),
        read_piece: read_exact_piece,
    };
    drain_span_pieces(&finished_span_work);
    assert_eq!(
        *finished_span_work
            .unfinished_piece_count
            .lock()
            .expect("count lock"),
        0
    );
}

#[test]
fn should_fail_the_span_when_every_piece_fails() {
    let piece_read_pool = private_pool(3);
    let source_file = numbered_block_source(8, PIECE_BYTES);
    let (all_pieces_read, _) = read_reversed_blocks(
        piece_read_pool,
        source_file.as_file(),
        8,
        PIECE_BYTES,
        7,
        fail_every_piece,
    );
    assert!(!all_pieces_read);
}

#[test]
fn should_end_the_span_and_keep_every_pool_thread_when_a_piece_read_panics() {
    let piece_read_pool = private_pool(2);
    let source_file = numbered_block_source(8, PIECE_BYTES);
    // The panicking piece may be claimed by a pool thread (the span fails) or
    // by the calling thread (the caller unwinds); either way the call ends.
    let span_outcome = catch_unwind(AssertUnwindSafe(|| {
        read_reversed_blocks(
            piece_read_pool,
            source_file.as_file(),
            8,
            PIECE_BYTES,
            7,
            panic_on_one_piece,
        )
        .0
    }));
    assert!(matches!(span_outcome, Ok(false) | Err(_)));
    assert_eq!(piece_read_pool.pending_span_ticket_count(), 0);
    for _ in 0..20 {
        let (all_pieces_read, destination) = read_reversed_blocks(
            piece_read_pool,
            source_file.as_file(),
            8,
            PIECE_BYTES,
            7,
            read_exact_piece,
        );
        assert!(
            all_pieces_read,
            "the pool keeps serving after a panicked read"
        );
        assert_reversed_blocks(&destination, 8, PIECE_BYTES);
    }
    assert_eq!(piece_read_pool.started_worker_count(), 2);
}

#[test]
fn should_give_each_concurrent_caller_its_own_bytes() {
    let piece_read_pool = private_pool(4);
    let caller_sources: Vec<_> = (0..4)
        .map(|_| numbered_block_source(24, PIECE_BYTES))
        .collect();
    std::thread::scope(|scope| {
        for caller_source in &caller_sources {
            scope.spawn(move || {
                for _ in 0..25 {
                    let (all_pieces_read, destination) = read_reversed_blocks(
                        piece_read_pool,
                        caller_source.as_file(),
                        24,
                        PIECE_BYTES,
                        7,
                        read_exact_piece,
                    );
                    assert!(all_pieces_read);
                    assert_reversed_blocks(&destination, 24, PIECE_BYTES);
                }
            });
        }
    });
}

#[test]
fn should_read_pieces_of_at_least_32_mib_through_the_partitioning_reader() {
    let large_piece_bytes = 32 << 20;
    let piece_read_pool = private_pool(2);
    let source_file = numbered_block_source(2, large_piece_bytes);
    let (all_pieces_read, destination) = read_reversed_blocks(
        piece_read_pool,
        source_file.as_file(),
        2,
        large_piece_bytes,
        1,
        super::super::bounded_range_read_concurrency::read_source_interval,
    );
    assert!(all_pieces_read);
    assert_reversed_blocks(&destination, 2, large_piece_bytes);
}
