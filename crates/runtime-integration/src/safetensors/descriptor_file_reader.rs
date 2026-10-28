//! MLX reader backed by one retained read-only file descriptor.
//!
//! # Why reads are deliberately serialized
//!
//! A representative model-loading probe *regressed* when positional reads ran
//! concurrently on this reader: the file's page-in behavior under macOS made
//! parallel positional reads slower than serialized ones for ordinary weight
//! loading. The cursor mutex therefore doubles as the file-read mutex —
//! sequential and positional reads share it, so all descriptor traffic is
//! serialized by design. (Large *expert-page* reads are a different workload;
//! those flow through the bounded reader, whose concurrency module measured
//! that parallelism there does pay.)
//!
//! # The vtable lifetime contract
//!
//! The boxed [`FileReaderState`] is handed to MLX-C through
//! `mlx_io_reader_new`; MLX retains it and invokes the vtable callbacks —
//! potentially from any thread evaluating lazy arrays — until the final
//! reader handle is released, at which point `reader_free` drops the box
//! exactly once. The `Drop` impl on [`OwnedFileReader`] releases the MLX
//! handle; MLX in turn invokes `reader_free`. Never drop the box from Rust
//! directly.

use std::{
    ffi::{c_char, c_int, c_void},
    fs::File,
    os::unix::fs::FileExt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{MlxRuntimeError, safetensors::read_metrics::PositionalFileReadMetrics};
use mlx_c_rust::raw;

/// Label MLX-C can surface for this reader in diagnostics.
const READER_LABEL: &[u8] = b"validated descriptor-backed weights\0";

/// MLX reader backed by one retained read-only file descriptor.
///
/// The handle is moved into loading results (see `lazy_map`) so the
/// descriptor outlives the loading call for as long as arrays can still
/// trigger reads.
#[derive(Debug)]
pub(super) struct OwnedFileReader(pub(super) raw::mlx_io_reader);

impl OwnedFileReader {
    /// Wraps a caller-owned read-only file as an MLX reader.
    ///
    /// Takes the `File` by value: ownership of the descriptor must move into
    /// the boxed state, because closing it early would break lazy reads.
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

/// Shared callback state behind the vtable.
#[derive(Debug)]
struct FileReaderState {
    file: File,
    /// Guards both the logical cursor and physical reads: positional reads
    /// deliberately share this mutex (see the module documentation).
    cursor_and_file_read_mutex: Mutex<FileReaderCursor>,
    /// Latched on the first failure; every later callback reports "not good"
    /// so MLX fails the dependent operation instead of proceeding with
    /// partially-read bytes.
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

unsafe extern "C" fn reader_seek(descriptor: *mut c_void, offset: i64, whence: c_int) -> c_int {
    // mlx-c v0.7.0 treats a negative seek return as the seek-failure signal.
    let seek_succeeded = unsafe {
        with_reader_cursor(descriptor, |reader_state, reader_cursor| {
            reader_seek_position(reader_state, reader_cursor, offset, whence)
        })
    }
    .unwrap_or(false);
    if seek_succeeded { 0 } else { -1 }
}

/// The absolute base position one seek origin refers to.
///
/// `None` means the origin is unsupported or the end-of-file length could
/// not be read; both flip the shared good flag so the owning read fails.
fn reader_seek_base_position_bytes(
    reader_state: &FileReaderState,
    reader_cursor_position_bytes: u64,
    whence: c_int,
) -> Option<i128> {
    match whence {
        libc::SEEK_SET => Some(0_i128),
        libc::SEEK_CUR => Some(i128::from(reader_cursor_position_bytes)),
        libc::SEEK_END => match reader_state.file.metadata() {
            Ok(metadata) => Some(i128::from(metadata.len())),
            Err(_) => {
                reader_state.is_good.store(false, Ordering::Release);
                None
            }
        },
        _ => {
            reader_state.is_good.store(false, Ordering::Release);
            None
        }
    }
}

/// Applies one C-style seek to the reader cursor.
///
/// Arithmetic in i128 so a malicious or corrupt (offset, whence) combination
/// is validated before it can wrap the u64 cursor; any failure flips the
/// shared good flag.
fn reader_seek_position(
    reader_state: &FileReaderState,
    reader_cursor: &mut FileReaderCursor,
    offset: i64,
    whence: c_int,
) -> bool {
    let Some(base_position_bytes) =
        reader_seek_base_position_bytes(reader_state, reader_cursor.position_bytes, whence)
    else {
        return false;
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

unsafe extern "C" fn reader_read(
    descriptor: *mut c_void,
    destination: *mut c_char,
    byte_count: usize,
) -> usize {
    unsafe {
        with_reader_cursor(descriptor, |reader_state, reader_cursor| {
            reader_read_bytes(reader_state, reader_cursor, destination, byte_count)
        })
    }
    .unwrap_or(0)
}

/// Reads one caller-owned buffer at the cursor and advances the cursor by
/// exactly what was consumed.
///
/// Zero reports the read as failed to MLX — the v0.7.0 wrapper compares the
/// returned count against the request, so an exact read reports the full
/// count. Every failure flips the shared good flag.
fn reader_read_bytes(
    reader_state: &FileReaderState,
    reader_cursor: &mut FileReaderCursor,
    destination: *mut c_char,
    byte_count: usize,
) -> usize {
    // `read_exact_at` is safe to call here: sequential reads share cursor
    // state and remain serialized by the cursor mutex.
    if read_exact_at(
        &reader_state.file,
        destination,
        byte_count,
        reader_cursor.position_bytes,
    )
    .is_err()
    {
        reader_state.is_good.store(false, Ordering::Release);
        return 0;
    }
    let Ok(consumed_byte_count) = u64::try_from(byte_count) else {
        reader_state.is_good.store(false, Ordering::Release);
        return 0;
    };
    match reader_cursor
        .position_bytes
        .checked_add(consumed_byte_count)
    {
        Some(position_bytes) => {
            reader_cursor.position_bytes = position_bytes;
            byte_count
        }
        None => {
            reader_state.is_good.store(false, Ordering::Release);
            0
        }
    }
}

unsafe extern "C" fn reader_read_at_offset(
    descriptor: *mut c_void,
    destination: *mut c_char,
    byte_count: usize,
    offset_bytes: usize,
) -> usize {
    let Ok(offset_bytes) = u64::try_from(offset_bytes) else {
        // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
        unsafe {
            with_reader_state(descriptor, |reader_state| {
                reader_state.is_good.store(false, Ordering::Release);
            });
        }
        return 0;
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
            if read_succeeded {
                byte_count
            } else {
                reader_state.is_good.store(false, Ordering::Release);
                0
            }
        })
    }
    .unwrap_or(0)
}

/// Writes are rejected, not ignored: acknowledging a write on a read-only
/// weights descriptor would let a caller believe mutated state reached disk.
unsafe extern "C" fn reader_write_is_unsupported(
    descriptor: *mut c_void,
    _source: *const c_char,
    _byte_count: usize,
) -> usize {
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe {
        with_reader_state(descriptor, |reader_state| {
            reader_state.is_good.store(false, Ordering::Release);
        });
    }
    // Zero processed bytes makes the v0.7.0 C++ wrapper reject the write.
    0
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

/// Performs one exact positional read into MLX-provided storage.
///
/// `read_exact_at` (not `read_at`) so a short read fails the callback — the
/// v0.7.0 wrapper validates returned counts, and handing MLX a partially
/// filled buffer would be memory-unsafe, not merely incorrect.
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
