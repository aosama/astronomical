//! Presents validated source-file intervals as one virtual SafeTensors payload to MLX.
//!
//! # The mechanism, end to end
//!
//! An expert-page manifest (built in `model-serving`) knows the exact byte
//! ranges of a shard that belong to the router-selected experts. Asking the
//! ordinary loader to open the whole shard would turn a top-K decode miss
//! into a whole-shard load. Instead:
//!
//! 1. The caller builds a **synthetic safetensors header** naming only the
//!    selected tensors, with offsets rebased into a compact *virtual*
//!    payload space.
//! 2. This reader presents `[synthetic header | virtual payload]` as one
//!    contiguous stream to MLX. Header bytes are served from memory; payload
//!    bytes are translated through the interval table into exact
//!    positional reads of the real file.
//! 3. Validation at construction guarantees the virtual payload has no
//!    holes (virtual contiguity), no duplicated source bytes
//!    (non-overlap), and exact declared size — a malformed manifest fails
//!    before MLX sees a single byte.
//!
//! The result is lazy arrays over exactly the selected expert rows; gaps and
//! unselected experts are never part of any request.
//!
//! # Lifetime contract
//!
//! Same as the descriptor reader: MLX owns the boxed
//! [`BoundedMultiRangeReaderState`] through the vtable and releases it via
//! `bounded_reader_free` when the last lazy array releases the reader. The
//! loading result keeps a holder alive for that window (see `lazy_map`).

use std::{
    ffi::{c_char, c_int, c_void},
    fs::File,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{MlxRuntimeError, safetensors::read_metrics::PositionalFileReadMetrics};
use mlx_c_rust::raw;

use super::bounded_range_read_concurrency::{SourcePiece, read_source_pieces};
use super::lazy_map::BoundedReadInterval;

/// Label MLX-C can surface for this reader in diagnostics.
const BOUNDED_READER_LABEL: &[u8] = b"bounded multi-range expert page reader\0";

/// Internal state for the bounded multi-range reader.
pub(super) struct BoundedMultiRangeReaderState {
    source_file: File,
    synthetic_header_bytes: Vec<u8>,
    /// Sorted by virtual payload offset; the binary search in
    /// `plan_interval_pieces` relies on that ordering.
    intervals: Vec<BoundedReadInterval>,
    total_payload_bytes: u64,
    cursor: Mutex<BoundedReaderCursor>,
    is_good: AtomicBool,
    expert_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
}

struct BoundedReaderCursor {
    position_bytes: u64,
}

impl BoundedMultiRangeReaderState {
    /// Validates the interval table and constructs the reader state.
    ///
    /// # Errors
    ///
    /// Three structural checks, each with a concrete failure mode it
    /// prevents:
    ///
    /// - **source overlap** — the same source bytes mapped twice would let
    ///   two virtual regions race the same file positions;
    /// - **virtual contiguity** — a hole would hand MLX zeros for tensor
    ///   data with no error anywhere;
    /// - **exact payload coverage** — a mismatch with the declared
    ///   `total_payload_bytes` means the synthetic header and the interval
    ///   table disagree, so one of them is stale.
    pub(super) fn new(
        source_file: File,
        synthetic_header_bytes: Vec<u8>,
        intervals: Vec<BoundedReadInterval>,
        total_payload_bytes: u64,
        expert_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Result<Self, MlxRuntimeError> {
        // Overlap is judged on source offsets, sorted independently of the
        // virtual ordering stored below.
        let mut intervals_sorted_by_source = intervals.clone();
        intervals_sorted_by_source.sort_by_key(|interval| interval.source_file_offset);
        for adjacent_intervals in intervals_sorted_by_source.windows(2) {
            let first_interval_end = adjacent_intervals[0].source_file_offset
                + adjacent_intervals[0].source_byte_count as u64;
            if first_interval_end > adjacent_intervals[1].source_file_offset {
                return Err(MlxRuntimeError::RuntimeOperation {
                    operation: "validate bounded reader intervals",
                    description: "source intervals must not overlap".to_owned(),
                });
            }
        }

        let mut intervals_sorted_by_virtual_offset = intervals;
        intervals_sorted_by_virtual_offset.sort_by_key(|interval| interval.virtual_payload_offset);
        let mut expected_virtual_offset = 0_u64;
        for interval in &intervals_sorted_by_virtual_offset {
            if interval.virtual_payload_offset != expected_virtual_offset {
                return Err(MlxRuntimeError::RuntimeOperation {
                    operation: "validate bounded reader intervals",
                    description: format!(
                        "virtual intervals must be contiguous: expected offset {expected_virtual_offset}, got {}",
                        interval.virtual_payload_offset
                    ),
                });
            }
            expected_virtual_offset += interval.source_byte_count as u64;
        }
        if expected_virtual_offset != total_payload_bytes {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "validate bounded reader intervals",
                description: format!(
                    "virtual intervals do not cover the declared payload: expected {total_payload_bytes}, got {expected_virtual_offset}"
                ),
            });
        }
        Ok(Self {
            source_file,
            synthetic_header_bytes,
            intervals: intervals_sorted_by_virtual_offset,
            total_payload_bytes,
            cursor: Mutex::new(BoundedReaderCursor { position_bytes: 0 }),
            is_good: AtomicBool::new(true),
            expert_file_read_metrics,
        })
    }

    /// Size of the stream MLX sees: synthetic header plus virtual payload.
    fn total_virtual_size(&self) -> u64 {
        self.synthetic_header_bytes.len() as u64 + self.total_payload_bytes
    }
}

/// The constructed reader handle, not yet surrendered to MLX-C.
pub(super) struct OwnedBoundedMultiRangeReader(pub(super) raw::mlx_io_reader);

impl OwnedBoundedMultiRangeReader {
    /// Hands the boxed state to MLX-C behind a fresh reader handle.
    pub(super) fn new(reader_state: BoundedMultiRangeReaderState) -> Result<Self, MlxRuntimeError> {
        let descriptor = Box::into_raw(Box::new(reader_state)).cast::<c_void>();
        let reader_vtable = raw::mlx_io_vtable {
            is_open: Some(bounded_reader_is_open),
            good: Some(bounded_reader_is_good),
            tell: Some(bounded_reader_tell),
            seek: Some(bounded_reader_seek),
            read: Some(bounded_reader_read),
            read_at_offset: Some(bounded_reader_read_at_offset),
            write: Some(bounded_reader_write_is_unsupported),
            label: Some(bounded_reader_label),
            free: Some(bounded_reader_free),
        };
        // SAFETY: `descriptor` points to an owned boxed reader state and every
        // callback matches the official C ABI. On success MLX owns descriptor
        // cleanup through the vtable.
        let raw_reader = unsafe { raw::mlx_io_reader_new(descriptor, reader_vtable) };
        if raw_reader.ctx.is_null() {
            // SAFETY: A null handle leaves the original descriptor with this caller.
            unsafe {
                drop(Box::from_raw(
                    descriptor.cast::<BoundedMultiRangeReaderState>(),
                ));
            }
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "allocate a bounded multi-range expert page reader",
                description: "MLX returned an empty handle".to_owned(),
            });
        }
        Ok(Self(raw_reader))
    }
}

/// A second handle to the same MLX reader, used only to keep it alive.
///
/// MLX-C readers are reference-counted; the loading result holds one
/// reference so the vtable state survives as long as the returned lazy
/// arrays might still read.
pub(super) struct OwnedBoundedMultiRangeReaderHolder(pub(super) raw::mlx_io_reader);

impl Drop for OwnedBoundedMultiRangeReaderHolder {
    fn drop(&mut self) {
        // SAFETY: This owner releases the MLX reader exactly once.
        unsafe {
            raw::mlx_io_reader_free(self.0);
        }
    }
}

unsafe extern "C" fn bounded_reader_is_open(descriptor: *mut c_void) -> bool {
    !descriptor.is_null()
}

unsafe extern "C" fn bounded_reader_is_good(descriptor: *mut c_void) -> bool {
    // SAFETY: MLX invokes this callback only while it owns the descriptor.
    unsafe {
        with_bounded_reader_state(descriptor, |reader_state| {
            reader_state.is_good.load(Ordering::Acquire)
        })
    }
    .unwrap_or(false)
}

unsafe extern "C" fn bounded_reader_tell(descriptor: *mut c_void) -> usize {
    // SAFETY: MLX invokes this callback only while it owns the descriptor.
    unsafe {
        with_bounded_reader_state(descriptor, |reader_state| {
            let reader_cursor = reader_state
                .cursor
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            usize::try_from(reader_cursor.position_bytes).ok()
        })
    }
    .flatten()
    .unwrap_or(0)
}

unsafe extern "C" fn bounded_reader_seek(
    descriptor: *mut c_void,
    offset: i64,
    whence: c_int,
) -> c_int {
    // mlx-c treats a negative seek return as the seek-failure signal.
    let seek_succeeded = unsafe {
        with_bounded_reader_state(descriptor, |reader_state| {
            let mut reader_cursor = reader_state
                .cursor
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            bounded_reader_seek_position(reader_state, &mut reader_cursor, offset, whence)
        })
    }
    .unwrap_or(false);
    if seek_succeeded { 0 } else { -1 }
}

/// Applies one C-style seek to the bounded reader's cursor.
///
/// i128 arithmetic validates (offset, whence) before any u64 write. SEEK_END
/// anchors at the *virtual* size, not the source file size — MLX must never
/// learn the real shard is larger. Any failure flips the shared good flag.
fn bounded_reader_seek_position(
    reader_state: &BoundedMultiRangeReaderState,
    reader_cursor: &mut BoundedReaderCursor,
    offset: i64,
    whence: c_int,
) -> bool {
    let base_position_bytes = match whence {
        libc::SEEK_SET => 0_i128,
        libc::SEEK_CUR => i128::from(reader_cursor.position_bytes),
        libc::SEEK_END => i128::from(reader_state.total_virtual_size()),
        _ => {
            reader_state.is_good.store(false, Ordering::Release);
            return false;
        }
    };
    let requested_position_bytes = base_position_bytes + i128::from(offset);
    match u64::try_from(requested_position_bytes) {
        Ok(position_bytes) => {
            reader_cursor.position_bytes = position_bytes;
            true
        }
        Err(_) => {
            reader_state.is_good.store(false, Ordering::Release);
            false
        }
    }
}

unsafe extern "C" fn bounded_reader_read(
    descriptor: *mut c_void,
    destination: *mut c_char,
    byte_count: usize,
) -> usize {
    // SAFETY: MLX invokes this callback only while it owns the descriptor.
    unsafe {
        with_bounded_reader_state(descriptor, |reader_state| {
            let mut reader_cursor = reader_state
                .cursor
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let read_position_bytes = reader_cursor.position_bytes;
            let consumed_byte_count =
                read_virtual_bytes(reader_state, destination, byte_count, read_position_bytes);
            reader_cursor.position_bytes += consumed_byte_count as u64;
            // The mlx-c wrapper rejects any count short of the request, so a
            // read past the virtual payload now fails loudly instead of
            // handing MLX an unfilled tail.
            consumed_byte_count
        })
    }
    .unwrap_or(0)
}

unsafe extern "C" fn bounded_reader_read_at_offset(
    descriptor: *mut c_void,
    destination: *mut c_char,
    byte_count: usize,
    offset_bytes: usize,
) -> usize {
    let Ok(offset_bytes) = u64::try_from(offset_bytes) else {
        // SAFETY: MLX invokes this callback only while it owns the descriptor.
        unsafe {
            with_bounded_reader_state(descriptor, |reader_state| {
                reader_state.is_good.store(false, Ordering::Release);
            });
        }
        return 0;
    };
    // SAFETY: MLX invokes this callback only while it owns the descriptor.
    unsafe {
        with_bounded_reader_state(descriptor, |reader_state| {
            read_virtual_bytes(reader_state, destination, byte_count, offset_bytes)
        })
    }
    .unwrap_or(0)
}

/// Serves `byte_count` virtual bytes at `offset_bytes`, header first, and
/// returns how many were served.
///
/// A failed payload read returns zero bytes. MLX-C's reader checks only the
/// returned count and MLX's Load never asks `is_good`, so returning the full
/// count after a failed source read would hand MLX whatever its (possibly
/// recycled) buffer already held as expert weights, with no error anywhere.
fn read_virtual_bytes(
    reader_state: &BoundedMultiRangeReaderState,
    destination: *mut c_char,
    byte_count: usize,
    offset_bytes: u64,
) -> usize {
    let Some(destination_bytes) =
        claim_virtual_destination(reader_state, destination, byte_count, offset_bytes)
    else {
        return 0;
    };
    if serve_virtual_bytes(reader_state, destination_bytes, offset_bytes) {
        destination_bytes.len()
    } else {
        0
    }
}

/// The destination trimmed to the virtual bytes that exist at `offset_bytes`,
/// or `None` when there is nothing to serve. An offset past the end or a null
/// destination also latches the reader's error state.
fn claim_virtual_destination<'destination>(
    reader_state: &BoundedMultiRangeReaderState,
    destination: *mut c_char,
    byte_count: usize,
    offset_bytes: u64,
) -> Option<&'destination mut [u8]> {
    let virtual_size_bytes = reader_state.total_virtual_size();
    if offset_bytes >= virtual_size_bytes {
        reader_state.is_good.store(false, Ordering::Release);
        return None;
    }
    let bytes_to_read = byte_count.min((virtual_size_bytes - offset_bytes) as usize);
    if bytes_to_read == 0 {
        return None;
    }
    if destination.is_null() {
        reader_state.is_good.store(false, Ordering::Release);
        return None;
    }
    // SAFETY: MLX supplies writable storage for exactly `byte_count` bytes,
    // of which this borrows the first `bytes_to_read`, for this callback only.
    Some(unsafe { std::slice::from_raw_parts_mut(destination.cast::<u8>(), bytes_to_read) })
}

/// Fills `destination_bytes` from virtual offset `offset_bytes`. A read may
/// straddle the header/payload boundary: the header remainder is copied from
/// memory and the read continues straight into payload bytes.
fn serve_virtual_bytes(
    reader_state: &BoundedMultiRangeReaderState,
    destination_bytes: &mut [u8],
    offset_bytes: u64,
) -> bool {
    let header_byte_count = reader_state.synthetic_header_bytes.len() as u64;
    let header_bytes_to_copy = if offset_bytes < header_byte_count {
        let header_offset = offset_bytes as usize;
        let header_bytes_to_copy = destination_bytes
            .len()
            .min(header_byte_count as usize - header_offset);
        destination_bytes[..header_bytes_to_copy].copy_from_slice(
            &reader_state.synthetic_header_bytes
                [header_offset..header_offset + header_bytes_to_copy],
        );
        header_bytes_to_copy
    } else {
        0
    };
    let payload_byte_count = destination_bytes.len() - header_bytes_to_copy;
    // Saturating: a read that ends inside the header has no payload bytes,
    // and its payload offset is never used.
    let virtual_payload_offset =
        (offset_bytes + header_bytes_to_copy as u64).saturating_sub(header_byte_count);
    read_payload_at_offset(
        reader_state,
        destination_bytes,
        header_bytes_to_copy,
        payload_byte_count,
        virtual_payload_offset,
    )
}

/// Writes are structurally impossible on a virtual payload: there is no
/// backing store that could honor them.
unsafe extern "C" fn bounded_reader_write_is_unsupported(
    descriptor: *mut c_void,
    _source: *const c_char,
    _byte_count: usize,
) -> usize {
    // SAFETY: MLX invokes this callback only while it owns the descriptor.
    unsafe {
        with_bounded_reader_state(descriptor, |reader_state| {
            reader_state.is_good.store(false, Ordering::Release);
        });
    }
    // Zero processed bytes makes the mlx-c C++ wrapper reject the write.
    0
}

unsafe extern "C" fn bounded_reader_label(_descriptor: *mut c_void) -> *const c_char {
    BOUNDED_READER_LABEL.as_ptr().cast::<c_char>()
}

unsafe extern "C" fn bounded_reader_free(descriptor: *mut c_void) {
    if descriptor.is_null() {
        return;
    }
    // SAFETY: `descriptor` originated from `Box::into_raw` in the constructor,
    // and MLX invokes this callback exactly once when its final reader dies.
    unsafe {
        drop(Box::from_raw(
            descriptor.cast::<BoundedMultiRangeReaderState>(),
        ));
    }
}

unsafe fn with_bounded_reader_state<Output>(
    descriptor: *mut c_void,
    operation: impl FnOnce(&BoundedMultiRangeReaderState) -> Output,
) -> Option<Output> {
    if descriptor.is_null() {
        return None;
    }
    // SAFETY: MLX retains the boxed state until the final reader release.
    let reader_state = unsafe { &*descriptor.cast::<BoundedMultiRangeReaderState>() };
    Some(operation(reader_state))
}

/// One source read that fills part of a virtual-payload span.
#[derive(Debug)]
struct IntervalPiece {
    destination_offset: usize,
    byte_count: usize,
    source_file_offset: u64,
}

/// Translates one contiguous virtual-payload span into source-file reads and
/// performs them, returning whether every byte was read.
///
/// The span is cut into one piece per interval it touches, then the pieces
/// are read by `read_source_pieces`, concurrently whenever there is more than
/// one (see its documentation for the measurement). A failure latches the
/// reader's error state as well as returning `false`.
fn read_payload_at_offset(
    reader_state: &BoundedMultiRangeReaderState,
    destination: &mut [u8],
    destination_offset: usize,
    byte_count: usize,
    virtual_payload_offset: u64,
) -> bool {
    if byte_count == 0 {
        return true;
    }
    let payload_was_read = plan_interval_pieces(
        &reader_state.intervals,
        destination_offset,
        byte_count,
        virtual_payload_offset,
    )
    .is_some_and(|interval_pieces| {
        read_source_pieces(
            &reader_state.source_file,
            claim_piece_destinations(destination, &interval_pieces),
            reader_state.expert_file_read_metrics.as_deref(),
        )
    });
    if !payload_was_read {
        reader_state.is_good.store(false, Ordering::Release);
    }
    payload_was_read
}

/// The pieces that tile the span, in destination order, or `None` when an
/// offset lands between intervals or past the last one: impossible after
/// construction validation, but defensive here so the reader fails rather
/// than inventing bytes.
fn plan_interval_pieces(
    intervals: &[BoundedReadInterval],
    destination_offset: usize,
    byte_count: usize,
    virtual_payload_offset: u64,
) -> Option<Vec<IntervalPiece>> {
    let mut interval_pieces = Vec::new();
    let mut bytes_remaining = byte_count;
    let mut current_virtual_offset = virtual_payload_offset;
    let mut current_destination_offset = destination_offset;
    while bytes_remaining > 0 {
        let interval_index = intervals
            .binary_search_by(|interval| {
                interval_order_for_offset(interval, current_virtual_offset)
            })
            .ok()?;
        let interval = &intervals[interval_index];
        let offset_within_interval = current_virtual_offset - interval.virtual_payload_offset;
        let bytes_from_interval =
            bytes_remaining.min(interval.source_byte_count - offset_within_interval as usize);
        interval_pieces.push(IntervalPiece {
            destination_offset: current_destination_offset,
            byte_count: bytes_from_interval,
            source_file_offset: interval.source_file_offset + offset_within_interval,
        });
        current_virtual_offset += bytes_from_interval as u64;
        current_destination_offset += bytes_from_interval;
        bytes_remaining -= bytes_from_interval;
    }
    Some(interval_pieces)
}

fn interval_order_for_offset(
    interval: &BoundedReadInterval,
    virtual_payload_offset: u64,
) -> std::cmp::Ordering {
    let interval_end = interval.virtual_payload_offset + interval.source_byte_count as u64;
    if virtual_payload_offset < interval.virtual_payload_offset {
        std::cmp::Ordering::Greater
    } else if virtual_payload_offset >= interval_end {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Equal
    }
}

/// Splits `destination` into one disjoint mutable slice per piece. The
/// planner emits pieces that tile the span in order, so each slice starts
/// where the previous one ended.
fn claim_piece_destinations<'destination>(
    destination: &'destination mut [u8],
    interval_pieces: &[IntervalPiece],
) -> Vec<SourcePiece<'destination>> {
    let mut unclaimed_destination = &mut destination[interval_pieces[0].destination_offset..];
    let mut source_pieces = Vec::with_capacity(interval_pieces.len());
    for interval_piece in interval_pieces {
        let (piece_destination, remaining_destination) =
            std::mem::take(&mut unclaimed_destination).split_at_mut(interval_piece.byte_count);
        source_pieces.push(SourcePiece {
            destination: piece_destination,
            source_file_offset: interval_piece.source_file_offset,
        });
        unclaimed_destination = remaining_destination;
    }
    source_pieces
}

#[cfg(test)]
#[path = "bounded_range_reader_tests.rs"]
mod bounded_range_reader_tests;
