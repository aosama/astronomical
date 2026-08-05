use std::{
    ffi::{c_char, c_int, c_void},
    fs::File,
    os::unix::fs::FileExt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{MlxRuntimeError, PositionalFileReadMetrics, raw};

const READER_LABEL: &[u8] = b"validated descriptor-backed weights\0";

/// MLX reader backed by one retained read-only file descriptor.
#[derive(Debug)]
pub(super) struct OwnedFileReader(pub(super) raw::mlx_io_reader);

impl OwnedFileReader {
    pub(super) fn new(
        weights_file: File,
        positional_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Result<Self, MlxRuntimeError> {
        let descriptor = Box::into_raw(Box::new(FileReaderState::new(
            weights_file,
            positional_file_read_metrics,
        )))
        .cast::<c_void>();
        let reader_vtable = raw::mlx_io_vtable {
            is_open: Some(reader_is_open),
            good: Some(reader_is_good),
            tell: Some(reader_tell),
            seek: Some(reader_seek),
            read: Some(reader_read),
            read_at_offset: Some(reader_read_at_offset),
            write: Some(reader_write_is_unsupported),
            label: Some(reader_label),
            free: Some(reader_free),
        };
        // SAFETY: The descriptor owns callback state and every callback matches the official ABI.
        let raw_reader = unsafe { raw::mlx_io_reader_new(descriptor, reader_vtable) };
        if raw_reader.ctx.is_null() {
            // SAFETY: A null handle leaves the original descriptor with this caller.
            unsafe {
                drop(Box::from_raw(descriptor.cast::<FileReaderState>()));
            }
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "allocate an MLX descriptor reader",
                description: "MLX returned an empty handle".to_owned(),
            });
        }
        Ok(Self(raw_reader))
    }
}

impl Drop for OwnedFileReader {
    fn drop(&mut self) {
        // SAFETY: MLX retains callback state until all lazy arrays release the shared reader.
        unsafe {
            raw::mlx_io_reader_free(self.0);
        }
    }
}

#[derive(Debug)]
struct FileReaderState {
    file: File,
    cursor_and_file_read_mutex: Mutex<FileReaderCursor>,
    is_good: AtomicBool,
    positional_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
}

#[derive(Debug)]
struct FileReaderCursor {
    position_bytes: u64,
}

impl FileReaderState {
    fn new(
        file: File,
        positional_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Self {
        Self {
            file,
            cursor_and_file_read_mutex: Mutex::new(FileReaderCursor { position_bytes: 0 }),
            is_good: AtomicBool::new(true),
            positional_file_read_metrics,
        }
    }
}

unsafe extern "C" fn reader_is_open(descriptor: *mut c_void) -> bool {
    !descriptor.is_null()
}

unsafe extern "C" fn reader_is_good(descriptor: *mut c_void) -> bool {
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe {
        with_reader_state(descriptor, |reader_state| {
            reader_state.is_good.load(Ordering::Acquire)
        })
    }
    .unwrap_or(false)
}

unsafe extern "C" fn reader_tell(descriptor: *mut c_void) -> usize {
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe {
        with_reader_cursor(descriptor, |reader_state, reader_cursor| {
            usize::try_from(reader_cursor.position_bytes).map_err(|_| {
                reader_state.is_good.store(false, Ordering::Release);
            })
        })
    }
    .and_then(Result::ok)
    .unwrap_or(0)
}

unsafe extern "C" fn reader_seek(descriptor: *mut c_void, offset: i64, whence: c_int) {
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe {
        with_reader_cursor(descriptor, |reader_state, reader_cursor| {
            let base_position_bytes = match whence {
                libc::SEEK_SET => 0_i128,
                libc::SEEK_CUR => i128::from(reader_cursor.position_bytes),
                libc::SEEK_END => match reader_state.file.metadata() {
                    Ok(metadata) => i128::from(metadata.len()),
                    Err(_) => {
                        reader_state.is_good.store(false, Ordering::Release);
                        return;
                    }
                },
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

unsafe extern "C" fn reader_read(
    descriptor: *mut c_void,
    destination: *mut c_char,
    byte_count: usize,
) {
    // SAFETY: Sequential reads share cursor state and remain serialized.
    unsafe {
        with_reader_cursor(descriptor, |reader_state, reader_cursor| {
            if read_exact_at(
                &reader_state.file,
                destination,
                byte_count,
                reader_cursor.position_bytes,
            )
            .is_err()
            {
                reader_state.is_good.store(false, Ordering::Release);
                return;
            }
            let Ok(consumed_byte_count) = u64::try_from(byte_count) else {
                reader_state.is_good.store(false, Ordering::Release);
                return;
            };
            match reader_cursor
                .position_bytes
                .checked_add(consumed_byte_count)
            {
                Some(position_bytes) => reader_cursor.position_bytes = position_bytes,
                None => reader_state.is_good.store(false, Ordering::Release),
            }
        });
    }
}

unsafe extern "C" fn reader_read_at_offset(
    descriptor: *mut c_void,
    destination: *mut c_char,
    byte_count: usize,
    offset_bytes: usize,
) {
    let Ok(offset_bytes) = u64::try_from(offset_bytes) else {
        // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
        unsafe {
            with_reader_state(descriptor, |reader_state| {
                reader_state.is_good.store(false, Ordering::Release);
            });
        }
        return;
    };
    // SAFETY: Representative model loading regressed with concurrent reads, so
    // positional reads deliberately share the cursor's file-read mutex.
    unsafe {
        with_reader_cursor(descriptor, |reader_state, _reader_cursor| {
            let read_operation =
                || read_exact_at(&reader_state.file, destination, byte_count, offset_bytes).is_ok();
            let read_succeeded = match reader_state.positional_file_read_metrics.as_ref() {
                Some(positional_file_read_metrics) => {
                    positional_file_read_metrics.measure_read(byte_count, read_operation)
                }
                None => read_operation(),
            };
            if !read_succeeded {
                reader_state.is_good.store(false, Ordering::Release);
            }
        });
    }
}

unsafe extern "C" fn reader_write_is_unsupported(
    descriptor: *mut c_void,
    _source: *const c_char,
    _byte_count: usize,
) {
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe {
        with_reader_state(descriptor, |reader_state| {
            reader_state.is_good.store(false, Ordering::Release);
        });
    }
}

unsafe extern "C" fn reader_label(_descriptor: *mut c_void) -> *const c_char {
    READER_LABEL.as_ptr().cast::<c_char>()
}

unsafe extern "C" fn reader_free(descriptor: *mut c_void) {
    if descriptor.is_null() {
        return;
    }
    // SAFETY: The descriptor originated from Box::into_raw and is released exactly once.
    unsafe {
        drop(Box::from_raw(descriptor.cast::<FileReaderState>()));
    }
}

unsafe fn with_reader_state<Output>(
    descriptor: *mut c_void,
    operation: impl FnOnce(&FileReaderState) -> Output,
) -> Option<Output> {
    if descriptor.is_null() {
        return None;
    }
    // SAFETY: MLX retains the boxed state until the final reader release.
    let reader_state = unsafe { &*descriptor.cast::<FileReaderState>() };
    Some(operation(reader_state))
}

unsafe fn with_reader_cursor<Output>(
    descriptor: *mut c_void,
    operation: impl FnOnce(&FileReaderState, &mut FileReaderCursor) -> Output,
) -> Option<Output> {
    // SAFETY: The shared state remains live; cursor and intentionally serialized
    // positional file-read operations acquire this mutex.
    unsafe {
        with_reader_state(descriptor, |reader_state| {
            let mut reader_cursor = reader_state
                .cursor_and_file_read_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            operation(reader_state, &mut reader_cursor)
        })
    }
}

fn read_exact_at(
    file: &File,
    destination: *mut c_char,
    byte_count: usize,
    offset_bytes: u64,
) -> std::io::Result<()> {
    if byte_count == 0 {
        return Ok(());
    }
    if destination.is_null() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "MLX reader supplied a null destination",
        ));
    }
    // SAFETY: MLX supplies writable storage for exactly byte_count bytes for this callback.
    let destination_bytes =
        unsafe { std::slice::from_raw_parts_mut(destination.cast::<u8>(), byte_count) };
    file.read_exact_at(destination_bytes, offset_bytes)
}
