use std::{
    ffi::{c_char, c_int, c_void},
    fs::File,
    os::unix::fs::FileExt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    MlxRuntimeError, PositionalFileReadMetrics, mlx_safetensors::BoundedReadInterval, raw,
};

const BOUNDED_READER_LABEL: &[u8] = b"bounded multi-range expert page reader\0";

/// Internal state for the bounded multi-range reader.
pub(super) struct BoundedMultiRangeReaderState {
    source_file: File,
    synthetic_header_bytes: Vec<u8>,
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
    pub(super) fn new(
        source_file: File,
        synthetic_header_bytes: Vec<u8>,
        intervals: Vec<BoundedReadInterval>,
        total_payload_bytes: u64,
        expert_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Result<Self, MlxRuntimeError> {
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

    fn total_virtual_size(&self) -> u64 {
        self.synthetic_header_bytes.len() as u64 + self.total_payload_bytes
    }
}

pub(super) struct OwnedBoundedMultiRangeReader(pub(super) raw::mlx_io_reader);

impl OwnedBoundedMultiRangeReader {
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

unsafe extern "C" fn bounded_reader_seek(descriptor: *mut c_void, offset: i64, whence: c_int) {
    // SAFETY: MLX invokes this callback only while it owns the descriptor.
    unsafe {
        with_bounded_reader_state(descriptor, |reader_state| {
            let mut reader_cursor = reader_state
                .cursor
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let base_position_bytes = match whence {
                libc::SEEK_SET => 0_i128,
                libc::SEEK_CUR => i128::from(reader_cursor.position_bytes),
                libc::SEEK_END => i128::from(reader_state.total_virtual_size()),
                _ => {
                    reader_state.is_good.store(false, Ordering::Release);
                    return;
                }
            };
            let requested_position_bytes = base_position_bytes + i128::from(offset);
            match u64::try_from(requested_position_bytes) {
                Ok(position_bytes) => reader_cursor.position_bytes = position_bytes,
                Err(_) => reader_state.is_good.store(false, Ordering::Release),
            }
        });
    }
}

unsafe extern "C" fn bounded_reader_read(
    descriptor: *mut c_void,
    destination: *mut c_char,
    byte_count: usize,
) {
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
        });
    }
}

unsafe extern "C" fn bounded_reader_read_at_offset(
    descriptor: *mut c_void,
    destination: *mut c_char,
    byte_count: usize,
    offset_bytes: usize,
) {
    let Ok(offset_bytes) = u64::try_from(offset_bytes) else {
        // SAFETY: MLX invokes this callback only while it owns the descriptor.
        unsafe {
            with_bounded_reader_state(descriptor, |reader_state| {
                reader_state.is_good.store(false, Ordering::Release);
            });
        }
        return;
    };
    // SAFETY: MLX invokes this callback only while it owns the descriptor.
    unsafe {
        with_bounded_reader_state(descriptor, |reader_state| {
            read_virtual_bytes(reader_state, destination, byte_count, offset_bytes);
        });
    }
}

fn read_virtual_bytes(
    reader_state: &BoundedMultiRangeReaderState,
    destination: *mut c_char,
    byte_count: usize,
    offset_bytes: u64,
) -> usize {
    let virtual_size_bytes = reader_state.total_virtual_size();
    if offset_bytes >= virtual_size_bytes {
        reader_state.is_good.store(false, Ordering::Release);
        return 0;
    }
    let bytes_to_read = byte_count.min((virtual_size_bytes - offset_bytes) as usize);
    if bytes_to_read == 0 {
        return 0;
    }
    if destination.is_null() {
        reader_state.is_good.store(false, Ordering::Release);
        return 0;
    }

    // SAFETY: MLX supplies writable storage for exactly `byte_count` bytes.
    let destination_bytes =
        unsafe { std::slice::from_raw_parts_mut(destination.cast::<u8>(), bytes_to_read) };
    let header_byte_count = reader_state.synthetic_header_bytes.len() as u64;
    if offset_bytes < header_byte_count {
        let header_offset = offset_bytes as usize;
        let header_bytes_remaining = header_byte_count as usize - header_offset;
        let header_bytes_to_read = bytes_to_read.min(header_bytes_remaining);
        destination_bytes[..header_bytes_to_read].copy_from_slice(
            &reader_state.synthetic_header_bytes
                [header_offset..header_offset + header_bytes_to_read],
        );
        if header_bytes_to_read < bytes_to_read {
            read_payload_at_offset(
                reader_state,
                destination_bytes,
                header_bytes_to_read,
                bytes_to_read - header_bytes_to_read,
                offset_bytes + header_bytes_to_read as u64 - header_byte_count,
            );
        }
        return bytes_to_read;
    }

    read_payload_at_offset(
        reader_state,
        destination_bytes,
        0,
        bytes_to_read,
        offset_bytes - header_byte_count,
    );
    bytes_to_read
}

unsafe extern "C" fn bounded_reader_write_is_unsupported(
    descriptor: *mut c_void,
    _source: *const c_char,
    _byte_count: usize,
) {
    // SAFETY: MLX invokes this callback only while it owns the descriptor.
    unsafe {
        with_bounded_reader_state(descriptor, |reader_state| {
            reader_state.is_good.store(false, Ordering::Release);
        });
    }
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

fn read_payload_at_offset(
    reader_state: &BoundedMultiRangeReaderState,
    destination: &mut [u8],
    destination_offset: usize,
    byte_count: usize,
    payload_offset: u64,
) {
    if byte_count == 0 {
        return;
    }
    let mut bytes_remaining = byte_count;
    let mut current_payload_offset = payload_offset;
    let mut current_destination_offset = destination_offset;

    while bytes_remaining > 0 {
        let interval_index = match reader_state.intervals.binary_search_by(|interval| {
            let interval_start = interval.virtual_payload_offset;
            let interval_end = interval.virtual_payload_offset + interval.source_byte_count as u64;
            if current_payload_offset < interval_start {
                std::cmp::Ordering::Greater
            } else if current_payload_offset >= interval_end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        }) {
            Ok(interval_index) => interval_index,
            Err(_) => {
                reader_state.is_good.store(false, Ordering::Release);
                return;
            }
        };
        let interval = &reader_state.intervals[interval_index];
        let offset_within_interval = current_payload_offset - interval.virtual_payload_offset;
        let bytes_available_in_interval =
            interval.source_byte_count - offset_within_interval as usize;
        let bytes_from_interval = bytes_remaining.min(bytes_available_in_interval);
        let source_offset = interval.source_file_offset + offset_within_interval;
        let destination_end = current_destination_offset + bytes_from_interval;
        let mut read_operation = || {
            reader_state
                .source_file
                .read_exact_at(
                    &mut destination[current_destination_offset..destination_end],
                    source_offset,
                )
                .is_ok()
        };
        let read_succeeded = match reader_state.expert_file_read_metrics.as_ref() {
            Some(expert_file_read_metrics) => {
                expert_file_read_metrics.measure_read(bytes_from_interval, read_operation)
            }
            None => read_operation(),
        };
        if !read_succeeded {
            reader_state.is_good.store(false, Ordering::Release);
            return;
        }
        current_payload_offset += bytes_from_interval as u64;
        current_destination_offset = destination_end;
        bytes_remaining -= bytes_from_interval;
    }
}
