use std::ffi::{c_char, c_int, c_void};
use std::sync::Mutex;

use crate::mlx_runtime::check_status;
use crate::mlx_safetensors_writer::{OwnedMetadataMap, OwnedTensorMap, writer_error};
use crate::{MlxArray, MlxRuntime, MlxRuntimeError, raw};

const MEMORY_WRITER_LABEL: &[u8] = b"bounded in-memory safetensors output\0";

impl MlxRuntime {
    /// Serializes named arrays without touching the filesystem.
    ///
    /// The caller can transfer the returned bytes to a non-MLX writer thread;
    /// MLX arrays and runtime state remain confined to their owner thread.
    pub fn serialize_safetensors(
        &self,
        named_arrays: &[(&str, &MlxArray)],
        metadata_entries: &[(&str, &str)],
    ) -> Result<Vec<u8>, MlxRuntimeError> {
        if named_arrays.is_empty() {
            return Err(writer_error(
                "serialize safetensors",
                "at least one named array is required",
            ));
        }

        let tensor_map = OwnedTensorMap::from_entries(named_arrays)?;
        let metadata_map = OwnedMetadataMap::from_entries(metadata_entries)?;
        let memory_writer = OwnedMemoryWriter::new()?;
        // SAFETY: The writer and both maps remain live for this synchronous serialization.
        let save_status = unsafe {
            raw::mlx_save_safetensors_writer(memory_writer.raw_writer, tensor_map.0, metadata_map.0)
        };
        check_status(save_status, "serialize safetensors into bounded memory")?;
        memory_writer.into_bytes()
    }
}

struct OwnedMemoryWriter {
    raw_writer: raw::mlx_io_writer,
    writer_state: *mut Mutex<MemoryWriterState>,
}

impl OwnedMemoryWriter {
    fn new() -> Result<Self, MlxRuntimeError> {
        let writer_state = Box::into_raw(Box::new(Mutex::new(MemoryWriterState {
            output_bytes: Vec::new(),
            position_bytes: 0,
            is_good: true,
        })));
        let writer_vtable = raw::mlx_io_vtable {
            is_open: Some(memory_writer_is_open),
            good: Some(memory_writer_is_good),
            tell: Some(memory_writer_tell),
            seek: Some(memory_writer_seek),
            read: Some(memory_writer_read_is_unsupported),
            read_at_offset: Some(memory_writer_read_at_offset_is_unsupported),
            write: Some(memory_writer_write),
            label: Some(memory_writer_label),
            free: Some(memory_writer_free),
        };
        // SAFETY: The descriptor owns boxed state and every callback matches the C ABI.
        let raw_writer = unsafe { raw::mlx_io_writer_new(writer_state.cast(), writer_vtable) };
        if raw_writer.ctx.is_null() {
            // SAFETY: MLX did not accept ownership when it returned an empty handle.
            unsafe {
                drop(Box::from_raw(writer_state));
            }
            return Err(writer_error(
                "allocate an MLX memory writer",
                "MLX returned an empty handle",
            ));
        }
        Ok(Self {
            raw_writer,
            writer_state,
        })
    }

    fn into_bytes(self) -> Result<Vec<u8>, MlxRuntimeError> {
        let writer_mutex = unsafe { &*self.writer_state };
        let mut writer_state = writer_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !writer_state.is_good {
            return Err(writer_error(
                "serialize safetensors into bounded memory",
                "the memory writer reported a failure",
            ));
        }
        Ok(std::mem::take(&mut writer_state.output_bytes))
    }
}

impl Drop for OwnedMemoryWriter {
    fn drop(&mut self) {
        // SAFETY: This owner releases the writer exactly once; its vtable frees the boxed state.
        unsafe {
            raw::mlx_io_writer_free(self.raw_writer);
        }
    }
}

struct MemoryWriterState {
    output_bytes: Vec<u8>,
    position_bytes: usize,
    is_good: bool,
}

unsafe extern "C" fn memory_writer_is_open(descriptor: *mut c_void) -> bool {
    !descriptor.is_null()
}

unsafe extern "C" fn memory_writer_is_good(descriptor: *mut c_void) -> bool {
    unsafe { with_memory_writer_state(descriptor, |writer_state| writer_state.is_good) }
        .unwrap_or(false)
}

unsafe extern "C" fn memory_writer_tell(descriptor: *mut c_void) -> usize {
    unsafe { with_memory_writer_state(descriptor, |writer_state| writer_state.position_bytes) }
        .unwrap_or(0)
}

unsafe extern "C" fn memory_writer_seek(descriptor: *mut c_void, offset: i64, whence: c_int) {
    unsafe {
        with_memory_writer_state(descriptor, |writer_state| {
            let base_position_bytes = match whence {
                libc::SEEK_SET => 0_i128,
                libc::SEEK_CUR => writer_state.position_bytes as i128,
                libc::SEEK_END => writer_state.output_bytes.len() as i128,
                _ => {
                    writer_state.is_good = false;
                    return;
                }
            };
            match usize::try_from(base_position_bytes + i128::from(offset)) {
                Ok(position_bytes) => writer_state.position_bytes = position_bytes,
                Err(_) => writer_state.is_good = false,
            }
        });
    }
}

unsafe extern "C" fn memory_writer_read_is_unsupported(
    descriptor: *mut c_void,
    _destination: *mut c_char,
    _byte_count: usize,
) {
    unsafe {
        with_memory_writer_state(descriptor, |writer_state| writer_state.is_good = false);
    }
}

unsafe extern "C" fn memory_writer_read_at_offset_is_unsupported(
    descriptor: *mut c_void,
    _destination: *mut c_char,
    _byte_count: usize,
    _offset_bytes: usize,
) {
    unsafe {
        with_memory_writer_state(descriptor, |writer_state| writer_state.is_good = false);
    }
}

unsafe extern "C" fn memory_writer_write(
    descriptor: *mut c_void,
    source: *const c_char,
    byte_count: usize,
) {
    unsafe {
        with_memory_writer_state(descriptor, |writer_state| {
            if byte_count == 0 {
                return;
            }
            if source.is_null() {
                writer_state.is_good = false;
                return;
            }
            let Some(write_end_bytes) = writer_state.position_bytes.checked_add(byte_count) else {
                writer_state.is_good = false;
                return;
            };
            if write_end_bytes > writer_state.output_bytes.len() {
                writer_state.output_bytes.resize(write_end_bytes, 0);
            }
            let source_bytes = std::slice::from_raw_parts(source.cast::<u8>(), byte_count);
            writer_state.output_bytes[writer_state.position_bytes..write_end_bytes]
                .copy_from_slice(source_bytes);
            writer_state.position_bytes = write_end_bytes;
        });
    }
}

unsafe extern "C" fn memory_writer_label(_descriptor: *mut c_void) -> *const c_char {
    MEMORY_WRITER_LABEL.as_ptr().cast()
}

unsafe extern "C" fn memory_writer_free(descriptor: *mut c_void) {
    if descriptor.is_null() {
        return;
    }
    // SAFETY: The descriptor originated from Box::into_raw and MLX releases it once.
    unsafe {
        drop(Box::from_raw(descriptor.cast::<Mutex<MemoryWriterState>>()));
    }
}

unsafe fn with_memory_writer_state<Output>(
    descriptor: *mut c_void,
    operation: impl FnOnce(&mut MemoryWriterState) -> Output,
) -> Option<Output> {
    if descriptor.is_null() {
        return None;
    }
    // SAFETY: MLX retains the boxed mutex until memory_writer_free is called.
    let writer_mutex = unsafe { &*descriptor.cast::<Mutex<MemoryWriterState>>() };
    let mut writer_state = writer_mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Some(operation(&mut writer_state))
}
