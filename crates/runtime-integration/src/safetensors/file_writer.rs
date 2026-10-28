//! Rust ownership and callback bridge for MLX's descriptor-backed writer.
//!
//! # Why a descriptor instead of bytes
//!
//! Prompt-cache state can be many gigabytes. Serializing it into one Rust
//! byte vector would temporarily duplicate the complete capture. This bridge
//! instead gives MLX one retained writable descriptor, allowing MLX to
//! materialize and write one tensor at a time while Rust retains explicit
//! file and error ownership.
//!
//! # The error-latching pattern
//!
//! Native callbacks cannot return Rust `Result`, so an input/output failure
//! inside a callback is latched into [`FileWriterState::first_io_error`] and
//! the callback reports the failure to MLX through its return count.
//! [`OwnedFileWriter::finish`] is the single synchronization point that
//! restores ordinary Rust error propagation: it replays the first latched
//! error (the earliest failure usually identifies the root cause; later
//! callbacks may keep running after `good()` turns false, so overwriting
//! that evidence would hide it) and makes the file durable with `sync_all`.
//! Success of [`MlxRuntime::save_safetensors`] therefore means native
//! serialization *and* descriptor input/output *and* the durability sync all
//! succeeded.
//!
//! # Lifetime contract
//!
//! `writer_free` owns and releases the boxed state when MLX releases the
//! writer handle; the Rust-side `Drop` on [`OwnedFileWriter`] releases the
//! MLX handle. The raw state pointer stored beside the handle is used only
//! by `finish`, strictly before `Drop` runs.

use std::{
    ffi::{c_char, c_int, c_void},
    fs::File,
    os::unix::fs::FileExt,
    sync::Mutex,
};

use crate::{MlxRuntime, MlxRuntimeError, mlx_runtime::check_status};
use mlx_c_rust::{MlxArray, raw};

use super::name_maps::{OwnedMetadataMap, OwnedTensorMap};

/// Label MLX-C can surface for this writer in diagnostics.
const WRITER_LABEL: &[u8] = b"retained descriptor-backed safetensors output\0";

/// Summary of one successful descriptor-backed save.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxSafetensorsWriteOutcome {
    written_byte_count: u64,
}

impl MlxSafetensorsWriteOutcome {
    /// Bytes the callbacks actually placed on the descriptor. Persistent
    /// store sizing decisions consume this number, not the callers' own
    /// estimate of the capture.
    #[must_use]
    pub const fn written_byte_count(self) -> u64 {
        self.written_byte_count
    }
}

/// Failures of a descriptor-backed save, split by owner.
///
/// [`MlxSafetensorsWriterError::DescriptorIo`] means the filesystem side
/// failed (the latched first error); [`MlxSafetensorsWriterError::Native`]
/// means MLX-C rejected the serialization itself. The split tells the
/// caller whether retrying on the same descriptor can help.
#[derive(Debug, thiserror::Error)]
pub enum MlxSafetensorsWriterError {
    #[error("safetensors descriptor I/O failed")]
    DescriptorIo {
        #[source]
        source: std::io::Error,
    },
    #[error("MLX safetensors publication failed")]
    Native {
        #[source]
        source: MlxRuntimeError,
    },
}

impl MlxRuntime {
    /// Saves named arrays and string metadata through one retained writable descriptor.
    ///
    /// This call is synchronous: tensor/map references and the descriptor remain
    /// live until native serialization returns and the file has been synchronized.
    /// Success therefore means native serialization and descriptor I/O both succeeded.
    ///
    /// # Errors
    ///
    /// Descriptor input/output problems surface as
    /// [`MlxSafetensorsWriterError::DescriptorIo`] (reported first — a
    /// callback usually holds the more concrete filesystem cause of a
    /// native failure); native serialization failures surface as
    /// [`MlxSafetensorsWriterError::Native`].
    pub fn save_safetensors(
        &self,
        output_file: File,
        named_arrays: &[(&str, &MlxArray)],
        metadata_entries: &[(&str, &str)],
    ) -> Result<MlxSafetensorsWriteOutcome, MlxSafetensorsWriterError> {
        if named_arrays.is_empty() {
            return Err(MlxSafetensorsWriterError::Native {
                source: writer_error("save safetensors", "at least one named array is required"),
            });
        }

        let tensor_map = OwnedTensorMap::from_entries(named_arrays)
            .map_err(|source| MlxSafetensorsWriterError::Native { source })?;
        let metadata_map = OwnedMetadataMap::from_entries(metadata_entries)
            .map_err(|source| MlxSafetensorsWriterError::Native { source })?;
        let writer = OwnedFileWriter::new(output_file)
            .map_err(|source| MlxSafetensorsWriterError::Native { source })?;
        // SAFETY: The writer and both maps remain live for this synchronous save.
        let save_status = unsafe {
            raw::mlx_save_safetensors_writer(writer.raw_writer, tensor_map.0, metadata_map.0)
        };
        // Preserve native status while still calling `finish`: a callback can
        // have recorded the more concrete filesystem error that caused MLX to
        // fail. `finish` reports descriptor I/O first, then native status below.
        let native_save_result = check_status(
            save_status,
            "save safetensors through a retained file descriptor",
        );
        let written_byte_count = writer
            .finish()
            .map_err(|source| MlxSafetensorsWriterError::DescriptorIo { source })?;
        native_save_result.map_err(|source| MlxSafetensorsWriterError::Native { source })?;
        Ok(MlxSafetensorsWriteOutcome { written_byte_count })
    }
}

struct OwnedFileWriter {
    // MLX owns callback invocation; this Rust owner owns the MLX writer handle.
    // The handle's `free` callback owns and releases the boxed state.
    raw_writer: raw::mlx_io_writer,
    // Non-owning pointer used only while `raw_writer` is alive so `finish` can
    // inspect callback state before Drop asks MLX to invoke `writer_free`.
    writer_state: *mut Mutex<FileWriterState>,
}

impl OwnedFileWriter {
    fn new(output_file: File) -> Result<Self, MlxRuntimeError> {
        let writer_state = Box::into_raw(Box::new(Mutex::new(FileWriterState {
            output_file,
            position_bytes: 0,
            first_io_error: None,
        })));
        let writer_vtable = raw::mlx_io_vtable {
            is_open: Some(writer_is_open),
            good: Some(writer_is_good),
            tell: Some(writer_tell),
            seek: Some(writer_seek),
            read: Some(writer_read_is_unsupported),
            read_at_offset: Some(writer_read_at_offset_is_unsupported),
            write: Some(writer_write),
            label: Some(writer_label),
            free: Some(writer_free),
        };
        // SAFETY: The descriptor owns boxed state and every callback matches the C ABI.
        let raw_writer = unsafe { raw::mlx_io_writer_new(writer_state.cast(), writer_vtable) };
        if raw_writer.ctx.is_null() {
            // SAFETY: MLX did not accept ownership when it returned an empty handle.
            unsafe {
                drop(Box::from_raw(writer_state));
            }
            return Err(writer_error(
                "allocate an MLX descriptor writer",
                "MLX returned an empty handle",
            ));
        }
        Ok(Self {
            raw_writer,
            writer_state,
        })
    }

    /// The error-and-durability synchronization point; see the module
    /// documentation for the full contract.
    fn finish(&self) -> Result<u64, std::io::Error> {
        // SAFETY: `raw_writer` is alive, so MLX has not yet invoked
        // `writer_free` and the state pointer is valid.
        let writer_mutex = unsafe { &*self.writer_state };
        let mut writer_state = writer_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(first_io_error) = writer_state.first_io_error.take() {
            return Err(first_io_error);
        }
        writer_state.output_file.sync_all()?;
        Ok(writer_state.position_bytes)
    }
}

impl Drop for OwnedFileWriter {
    fn drop(&mut self) {
        // SAFETY: This owner releases the writer exactly once; its vtable frees the boxed state.
        unsafe {
            raw::mlx_io_writer_free(self.raw_writer);
        }
    }
}

struct FileWriterState {
    // Positional writes use this file without mutating an operating-system file
    // cursor. `position_bytes` is the logical cursor exposed to MLX callbacks.
    output_file: File,
    position_bytes: u64,
    first_io_error: Option<std::io::Error>,
}

unsafe extern "C" fn writer_is_open(descriptor: *mut c_void) -> bool {
    !descriptor.is_null()
}

unsafe extern "C" fn writer_is_good(descriptor: *mut c_void) -> bool {
    unsafe {
        with_writer_state(descriptor, |writer_state| {
            writer_state.first_io_error.is_none()
        })
    }
    .unwrap_or(false)
}

unsafe extern "C" fn writer_tell(descriptor: *mut c_void) -> usize {
    unsafe {
        with_writer_state(descriptor, |writer_state| {
            writer_tell_position(writer_state)
        })
    }
    .flatten()
    .unwrap_or(0)
}

/// The current writer position as a C `tell` result.
///
/// `None` reports the position as unusable to MLX; the u64-to-usize range
/// failure latches the first error on the writer state so `finish` surfaces
/// it.
fn writer_tell_position(writer_state: &mut FileWriterState) -> Option<usize> {
    match usize::try_from(writer_state.position_bytes) {
        Ok(position_bytes) => Some(position_bytes),
        Err(_) => {
            record_writer_io_error(
                writer_state,
                std::io::Error::other("safetensors writer position exceeds usize"),
            );
            None
        }
    }
}

/// The absolute base position one seek origin refers to.
///
/// `None` means the origin is unsupported or its end-of-file length could
/// not be read; both failures latch an error on the writer state so `finish`
/// surfaces them.
fn seek_base_position_bytes(writer_state: &mut FileWriterState, whence: c_int) -> Option<i128> {
    match whence {
        libc::SEEK_SET => Some(0_i128),
        libc::SEEK_CUR => Some(i128::from(writer_state.position_bytes)),
        libc::SEEK_END => match writer_state.output_file.metadata() {
            Ok(file_metadata) => Some(i128::from(file_metadata.len())),
            Err(source) => {
                record_writer_io_error(writer_state, source);
                None
            }
        },
        _ => {
            record_writer_io_error(
                writer_state,
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "unsupported safetensors writer seek origin",
                ),
            );
            None
        }
    }
}

/// Applies one C-style seek to the writer position.
///
/// The target is computed in i128 so negative offsets and additions are
/// validated before conversion to the writer's u64 position; any failure
/// latches the first error on the writer state.
fn writer_seek_position(writer_state: &mut FileWriterState, offset: i64, whence: c_int) -> bool {
    let Some(base_position_bytes) = seek_base_position_bytes(writer_state, whence) else {
        return false;
    };
    match u64::try_from(base_position_bytes + i128::from(offset)) {
        Ok(position_bytes) => {
            writer_state.position_bytes = position_bytes;
            true
        }
        Err(_) => {
            record_writer_io_error(
                writer_state,
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "safetensors writer seek position is invalid",
                ),
            );
            false
        }
    }
}

unsafe extern "C" fn writer_seek(descriptor: *mut c_void, offset: i64, whence: c_int) -> c_int {
    // mlx-c v0.7.0 treats a negative seek return as the seek-failure signal.
    let seek_succeeded = unsafe {
        with_writer_state(descriptor, |writer_state| {
            writer_seek_position(writer_state, offset, whence)
        })
    }
    .unwrap_or(false);
    if seek_succeeded { 0 } else { -1 }
}

/// Reads on the write-only output descriptor are rejected outright rather
/// than acknowledged: the v0.7.0 C++ wrapper fails the call on a zero count,
/// which is the loud outcome a caller mixing up descriptors needs.
unsafe extern "C" fn writer_read_is_unsupported(
    descriptor: *mut c_void,
    _destination: *mut c_char,
    _byte_count: usize,
) -> usize {
    unsafe {
        with_writer_state(descriptor, |writer_state| {
            record_writer_io_error(
                writer_state,
                std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "safetensors output descriptor does not support reads",
                ),
            )
        });
    }
    // Reporting zero processed bytes makes the v0.7.0 C++ wrapper reject the
    // read outright instead of proceeding with an unfilled destination.
    0
}

unsafe extern "C" fn writer_read_at_offset_is_unsupported(
    descriptor: *mut c_void,
    _destination: *mut c_char,
    _byte_count: usize,
    _offset_bytes: usize,
) -> usize {
    unsafe {
        with_writer_state(descriptor, |writer_state| {
            record_writer_io_error(
                writer_state,
                std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "safetensors output descriptor does not support positional reads",
                ),
            )
        });
    }
    0
}

unsafe extern "C" fn writer_write(
    descriptor: *mut c_void,
    source: *const c_char,
    byte_count: usize,
) -> usize {
    // The v0.7.0 wrapper compares the returned count against the requested
    // byte_count and fails the write on any shortfall, so each path must
    // return exactly what reached the descriptor.
    unsafe {
        with_writer_state(descriptor, |writer_state| {
            writer_write_bytes(writer_state, source, byte_count)
        })
    }
    .flatten()
    .unwrap_or(0)
}

/// Writes one caller-owned buffer at the current writer position and
/// advances the position by exactly what reached the descriptor.
///
/// The byte count reports success to MLX (the v0.7.0 wrapper fails the
/// write on any shortfall, so the count must be exact); `None` reports
/// failure. Every failure latches the first error on the writer state, and
/// when the bytes are already on disk but a count cannot be represented,
/// the honest count keeps MLX moving while the latched error surfaces
/// through `finish`.
fn writer_write_bytes(
    writer_state: &mut FileWriterState,
    source: *const c_char,
    byte_count: usize,
) -> Option<usize> {
    if byte_count == 0 {
        return Some(byte_count);
    }
    if source.is_null() {
        record_writer_io_error(
            writer_state,
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "safetensors writer received a null source",
            ),
        );
        return None;
    }
    // MLX guarantees the source remains live for this callback only. Do
    // not retain the slice; copy it completely before returning to C++.
    // SAFETY: `source` is non-null and points at `byte_count` readable
    // bytes the caller owns for the duration of this callback.
    let source_bytes = unsafe { std::slice::from_raw_parts(source.cast::<u8>(), byte_count) };
    if let Err(source) = write_all_at(
        &writer_state.output_file,
        source_bytes,
        writer_state.position_bytes,
    ) {
        record_writer_io_error(writer_state, source);
        return None;
    }
    let Ok(written_byte_count) = u64::try_from(byte_count) else {
        // The bytes are on disk; the latched state error surfaces
        // through `finish` while the honest count keeps MLX moving.
        record_writer_io_error(
            writer_state,
            std::io::Error::other("safetensors write byte count exceeds u64"),
        );
        return Some(byte_count);
    };
    match writer_state.position_bytes.checked_add(written_byte_count) {
        Some(position_bytes) => {
            writer_state.position_bytes = position_bytes;
            Some(byte_count)
        }
        None => {
            record_writer_io_error(
                writer_state,
                std::io::Error::other("safetensors writer position overflowed"),
            );
            Some(byte_count)
        }
    }
}

fn record_writer_io_error(writer_state: &mut FileWriterState, source: std::io::Error) {
    // The earliest failure usually identifies the root cause. Later callbacks
    // may continue after `good()` becomes false; never overwrite that evidence.
    if writer_state.first_io_error.is_none() {
        writer_state.first_io_error = Some(source);
    }
}

unsafe extern "C" fn writer_label(_descriptor: *mut c_void) -> *const c_char {
    WRITER_LABEL.as_ptr().cast()
}

unsafe extern "C" fn writer_free(descriptor: *mut c_void) {
    if descriptor.is_null() {
        return;
    }
    // SAFETY: The descriptor originated from Box::into_raw and MLX releases it once.
    unsafe {
        drop(Box::from_raw(descriptor.cast::<Mutex<FileWriterState>>()));
    }
}

unsafe fn with_writer_state<Output>(
    descriptor: *mut c_void,
    operation: impl FnOnce(&mut FileWriterState) -> Output,
) -> Option<Output> {
    if descriptor.is_null() {
        return None;
    }
    // SAFETY: MLX retains the boxed mutex until writer_free is called.
    let writer_mutex = unsafe { &*descriptor.cast::<Mutex<FileWriterState>>() };
    let mut writer_state = writer_mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Some(operation(&mut writer_state))
}

/// Loops a positional write to completion.
///
/// `write_at` may legally perform a short write. Looping until this
/// callback's complete byte range reaches disk keeps the v0.7.0
/// returned-count contract honest, and rejecting zero progress avoids a
/// spin on a descriptor that can never advance.
fn write_all_at(
    output_file: &File,
    mut source_bytes: &[u8],
    mut offset_bytes: u64,
) -> std::io::Result<()> {
    while !source_bytes.is_empty() {
        let written_byte_count = output_file.write_at(source_bytes, offset_bytes)?;
        if written_byte_count == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "safetensors descriptor writer made no progress",
            ));
        }
        source_bytes = &source_bytes[written_byte_count..];
        offset_bytes = offset_bytes
            .checked_add(written_byte_count as u64)
            .ok_or_else(|| std::io::Error::other("safetensors write offset overflowed"))?;
    }
    Ok(())
}

/// Shared constructor for plain operation failures in the writer family.
///
/// `pub(crate)` so the in-memory writer (a sibling engine, not a sub-part of
/// this one) can produce identical wording for identical conditions.
pub(crate) fn writer_error(
    operation: &'static str,
    description: impl Into<String>,
) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.into(),
    }
}
