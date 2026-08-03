use std::{
    ffi::{CString, c_char, c_int, c_void},
    fs::File,
    os::unix::fs::FileExt,
    sync::Mutex,
};

use crate::{MlxArray, MlxRuntime, MlxRuntimeError, mlx_runtime::check_status, raw};

const WRITER_LABEL: &[u8] = b"retained descriptor-backed safetensors output\0";

impl MlxRuntime {
    /// Saves named arrays and string metadata through one retained writable descriptor.
    pub fn save_safetensors(
        &self,
        output_file: File,
        named_arrays: &[(&str, &MlxArray)],
        metadata_entries: &[(&str, &str)],
    ) -> Result<(), MlxRuntimeError> {
        if named_arrays.is_empty() {
            return Err(writer_error(
                "save safetensors",
                "at least one named array is required",
            ));
        }

        let tensor_map = OwnedTensorMap::from_entries(named_arrays)?;
        let metadata_map = OwnedMetadataMap::from_entries(metadata_entries)?;
        let writer = OwnedFileWriter::new(output_file)?;
        // SAFETY: The writer and both maps remain live for this synchronous save.
        let save_status = unsafe {
            raw::mlx_save_safetensors_writer(writer.raw_writer, tensor_map.0, metadata_map.0)
        };
        check_status(
            save_status,
            "save safetensors through a retained file descriptor",
        )?;
        writer.sync_all()?;
        Ok(())
    }
}

pub(crate) struct OwnedTensorMap(pub(crate) raw::mlx_map_string_to_array);

impl OwnedTensorMap {
    pub(crate) fn from_entries(
        named_arrays: &[(&str, &MlxArray)],
    ) -> Result<Self, MlxRuntimeError> {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let tensor_map = unsafe { raw::mlx_map_string_to_array_new() };
        if tensor_map.ctx.is_null() {
            return Err(writer_error(
                "allocate an MLX tensor map",
                "MLX returned an empty handle",
            ));
        }
        let owned_map = Self(tensor_map);
        for (tensor_name, tensor) in named_arrays {
            let tensor_name = checked_c_string(tensor_name, "insert a safetensors tensor name")?;
            // SAFETY: The map and tensor are live; MLX copies the key and retains array ownership.
            let insert_status = unsafe {
                raw::mlx_map_string_to_array_insert(owned_map.0, tensor_name.as_ptr(), tensor.raw())
            };
            check_status(insert_status, "insert a safetensors tensor")?;
        }
        Ok(owned_map)
    }
}

impl Drop for OwnedTensorMap {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live map exactly once.
        unsafe {
            raw::mlx_map_string_to_array_free(self.0);
        }
    }
}

pub(crate) struct OwnedMetadataMap(pub(crate) raw::mlx_map_string_to_string);

impl OwnedMetadataMap {
    pub(crate) fn from_entries(metadata_entries: &[(&str, &str)]) -> Result<Self, MlxRuntimeError> {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let metadata_map = unsafe { raw::mlx_map_string_to_string_new() };
        if metadata_map.ctx.is_null() {
            return Err(writer_error(
                "allocate an MLX metadata map",
                "MLX returned an empty handle",
            ));
        }
        let owned_map = Self(metadata_map);
        for (metadata_name, metadata_value) in metadata_entries {
            let metadata_name = checked_c_string(metadata_name, "insert a metadata name")?;
            let metadata_value = checked_c_string(metadata_value, "insert a metadata value")?;
            // SAFETY: The map is live and MLX copies both strings during this call.
            let insert_status = unsafe {
                raw::mlx_map_string_to_string_insert(
                    owned_map.0,
                    metadata_name.as_ptr(),
                    metadata_value.as_ptr(),
                )
            };
            check_status(insert_status, "insert safetensors metadata")?;
        }
        Ok(owned_map)
    }
}

impl Drop for OwnedMetadataMap {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live map exactly once.
        unsafe {
            raw::mlx_map_string_to_string_free(self.0);
        }
    }
}

struct OwnedFileWriter {
    raw_writer: raw::mlx_io_writer,
    writer_state: *mut Mutex<FileWriterState>,
}

impl OwnedFileWriter {
    fn new(output_file: File) -> Result<Self, MlxRuntimeError> {
        let writer_state = Box::into_raw(Box::new(Mutex::new(FileWriterState {
            output_file,
            position_bytes: 0,
            is_good: true,
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

    fn sync_all(&self) -> Result<(), MlxRuntimeError> {
        let writer_mutex = unsafe { &*self.writer_state };
        let mut writer_state = writer_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !writer_state.is_good {
            return Err(writer_error(
                "save safetensors through a retained file descriptor",
                "the descriptor writer reported an I/O failure",
            ));
        }
        writer_state.output_file.sync_all().map_err(|source| {
            writer_state.is_good = false;
            writer_error("synchronize a saved safetensors file", source.to_string())
        })
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
    output_file: File,
    position_bytes: u64,
    is_good: bool,
}

unsafe extern "C" fn writer_is_open(descriptor: *mut c_void) -> bool {
    !descriptor.is_null()
}

unsafe extern "C" fn writer_is_good(descriptor: *mut c_void) -> bool {
    unsafe { with_writer_state(descriptor, |writer_state| writer_state.is_good) }.unwrap_or(false)
}

unsafe extern "C" fn writer_tell(descriptor: *mut c_void) -> usize {
    unsafe {
        with_writer_state(descriptor, |writer_state| {
            usize::try_from(writer_state.position_bytes).ok()
        })
    }
    .flatten()
    .unwrap_or(0)
}

unsafe extern "C" fn writer_seek(descriptor: *mut c_void, offset: i64, whence: c_int) {
    unsafe {
        with_writer_state(descriptor, |writer_state| {
            let base_position_bytes = match whence {
                libc::SEEK_SET => 0_i128,
                libc::SEEK_CUR => i128::from(writer_state.position_bytes),
                libc::SEEK_END => match writer_state.output_file.metadata() {
                    Ok(file_metadata) => i128::from(file_metadata.len()),
                    Err(_) => {
                        writer_state.is_good = false;
                        return;
                    }
                },
                _ => {
                    writer_state.is_good = false;
                    return;
                }
            };
            match u64::try_from(base_position_bytes + i128::from(offset)) {
                Ok(position_bytes) => writer_state.position_bytes = position_bytes,
                Err(_) => writer_state.is_good = false,
            }
        });
    }
}

unsafe extern "C" fn writer_read_is_unsupported(
    descriptor: *mut c_void,
    _destination: *mut c_char,
    _byte_count: usize,
) {
    unsafe {
        with_writer_state(descriptor, |writer_state| writer_state.is_good = false);
    }
}

unsafe extern "C" fn writer_read_at_offset_is_unsupported(
    descriptor: *mut c_void,
    _destination: *mut c_char,
    _byte_count: usize,
    _offset_bytes: usize,
) {
    unsafe {
        with_writer_state(descriptor, |writer_state| writer_state.is_good = false);
    }
}

unsafe extern "C" fn writer_write(
    descriptor: *mut c_void,
    source: *const c_char,
    byte_count: usize,
) {
    unsafe {
        with_writer_state(descriptor, |writer_state| {
            if byte_count == 0 {
                return;
            }
            if source.is_null() {
                writer_state.is_good = false;
                return;
            }
            let source_bytes = std::slice::from_raw_parts(source.cast::<u8>(), byte_count);
            if write_all_at(
                &writer_state.output_file,
                source_bytes,
                writer_state.position_bytes,
            )
            .is_err()
            {
                writer_state.is_good = false;
                return;
            }
            let Ok(written_byte_count) = u64::try_from(byte_count) else {
                writer_state.is_good = false;
                return;
            };
            match writer_state.position_bytes.checked_add(written_byte_count) {
                Some(position_bytes) => writer_state.position_bytes = position_bytes,
                None => writer_state.is_good = false,
            }
        });
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

fn checked_c_string(text: &str, operation: &'static str) -> Result<CString, MlxRuntimeError> {
    CString::new(text).map_err(|_| writer_error(operation, "text contains an interior NUL byte"))
}

pub(crate) fn writer_error(
    operation: &'static str,
    description: impl Into<String>,
) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.into(),
    }
}
