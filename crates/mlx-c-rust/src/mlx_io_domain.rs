//! The MLX input/output domain: file loading and saving, safetensors maps,
//! GGUF containers, and the reader/writer transport types.

use std::os::raw::c_void;

use crate::error::check_status;
use crate::mlx_map_types::{MlxMapStringToArray, MlxMapStringToString};
use crate::mlx_vector_types::MlxVectorString;
use crate::raw;
use crate::{MlxArray, MlxCError, MlxStream, MlxString};

/// A host path argument translated into a C string.
fn path_argument(path: &str, operation: &'static str) -> Result<std::ffi::CString, MlxCError> {
    std::ffi::CString::new(path).map_err(|_| MlxCError {
        operation,
        description: "path contains an interior null byte".to_owned(),
    })
}

impl MlxArray {
    /// Loads one array from a file through MLX-C `mlx_load`.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when loading fails.
    pub fn load_from_file(path: &str, stream: &MlxStream) -> Result<Self, MlxCError> {
        let operation: &'static str = "load an MLX array from a file";
        let path_value = path_argument(path, operation)?;
        let mut output = Self::empty();
        // SAFETY: The path remains valid for the call, the stream handle is
        // live, and the output pointer is valid writable storage.
        let status = unsafe { raw::mlx_load(output.raw_mut(), path_value.as_ptr(), stream.raw()) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Saves this array to a file through MLX-C `mlx_save`.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when saving fails.
    pub fn save_to_file(&self, path: &str) -> Result<(), MlxCError> {
        let operation: &'static str = "save an MLX array to a file";
        let path_value = path_argument(path, operation)?;
        // SAFETY: The path remains valid for the call and the array handle
        // is live.
        let status = unsafe { raw::mlx_save(path_value.as_ptr(), self.raw()) };
        check_status(status, operation)
    }
}

/// Loads the string-to-array and string-to-string safetensors maps from a
/// file.
///
/// # Errors
/// Returns the captured MLX-C description when loading fails.
pub fn load_safetensors(
    path: &str,
    stream: &MlxStream,
) -> Result<(MlxMapStringToArray, MlxMapStringToString), MlxCError> {
    let operation: &'static str = "load MLX safetensors maps from a file";
    let path_value = path_argument(path, operation)?;
    let mut arrays = MlxMapStringToArray::new()?;
    let mut metadata = MlxMapStringToString::new()?;
    // SAFETY: The path remains valid for the call and both output pointers
    // are valid writable storage.
    let status = unsafe {
        raw::mlx_load_safetensors(
            arrays.raw_mut(),
            metadata.raw_mut(),
            path_value.as_ptr(),
            stream.raw(),
        )
    };
    check_status(status, operation)?;
    Ok((arrays, metadata))
}

/// Saves the parameter and metadata maps to a safetensors file.
///
/// # Errors
/// Returns the captured MLX-C description when saving fails.
pub fn save_safetensors(
    path: &str,
    arrays: &MlxMapStringToArray,
    metadata: &MlxMapStringToString,
) -> Result<(), MlxCError> {
    let operation: &'static str = "save MLX safetensors maps to a file";
    let path_value = path_argument(path, operation)?;
    // SAFETY: The path remains valid for the call and both handles are live.
    let status =
        unsafe { raw::mlx_save_safetensors(path_value.as_ptr(), arrays.raw(), metadata.raw()) };
    check_status(status, operation)
}

/// Owned MLX reader transport released exactly once.
///
/// The descriptor and vtable follow MLX-C's `mlx_io_reader` contract; the
/// safe surface keeps the pair together so the transport stays valid for its
/// whole lifetime.
#[derive(Debug)]
pub struct MlxIoReader {
    raw_reader: raw::mlx_io_reader,
}

impl MlxIoReader {
    /// Builds an owned reader from a descriptor and vtable following the
    /// MLX-C io contract.
    ///
    /// # Safety
    /// The descriptor must satisfy the vtable's expectations for the
    /// reader's whole lifetime, and every vtable slot must be a valid C
    /// ABI function.
    pub unsafe fn from_parts(
        descriptor: *mut c_void,
        vtable: raw::mlx_io_vtable,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX reader transport";
        // SAFETY: The caller upholds the descriptor/vtable contract and the
        // returned handle enters RAII ownership.
        let raw_reader = unsafe { raw::mlx_io_reader_new(descriptor, vtable) };
        if raw_reader.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty reader handle".to_owned(),
            });
        }
        Ok(Self { raw_reader })
    }

    /// The descriptor this transport was built with.
    pub fn descriptor(&self) -> Result<*mut c_void, MlxCError> {
        let operation: &'static str = "read the MLX reader descriptor";
        let mut descriptor: *mut c_void = std::ptr::null_mut();
        // SAFETY: `self` owns a live reader and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_io_reader_descriptor(&mut descriptor, self.raw_reader) };
        check_status(status, operation)?;
        Ok(descriptor)
    }

    /// Renders the reader description text.
    pub fn to_string_lossy(&self) -> Result<String, MlxCError> {
        let operation: &'static str = "describe an MLX reader transport";
        let mut output = MlxString::empty();
        // SAFETY: `self` owns a live reader and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_io_reader_tostring(output.raw_mut(), self.raw_reader) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output.to_string_lossy())
    }

    pub(crate) const fn raw(&self) -> raw::mlx_io_reader {
        self.raw_reader
    }
}

impl Drop for MlxIoReader {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live reader exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_io_reader_free(self.raw_reader);
        }
    }
}

/// Owned MLX writer transport released exactly once.
#[derive(Debug)]
pub struct MlxIoWriter {
    raw_writer: raw::mlx_io_writer,
}

impl MlxIoWriter {
    /// Builds an owned writer from a descriptor and vtable following the
    /// MLX-C io contract.
    ///
    /// # Safety
    /// The descriptor must satisfy the vtable's expectations for the
    /// writer's whole lifetime, and every vtable slot must be a valid C
    /// ABI function.
    pub unsafe fn from_parts(
        descriptor: *mut c_void,
        vtable: raw::mlx_io_vtable,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX writer transport";
        // SAFETY: The caller upholds the descriptor/vtable contract and the
        // returned handle enters RAII ownership.
        let raw_writer = unsafe { raw::mlx_io_writer_new(descriptor, vtable) };
        if raw_writer.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty writer handle".to_owned(),
            });
        }
        Ok(Self { raw_writer })
    }

    /// The descriptor this transport was built with.
    pub fn descriptor(&self) -> Result<*mut c_void, MlxCError> {
        let operation: &'static str = "read the MLX writer descriptor";
        let mut descriptor: *mut c_void = std::ptr::null_mut();
        // SAFETY: `self` owns a live writer and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_io_writer_descriptor(&mut descriptor, self.raw_writer) };
        check_status(status, operation)?;
        Ok(descriptor)
    }

    /// Renders the writer description text.
    pub fn to_string_lossy(&self) -> Result<String, MlxCError> {
        let operation: &'static str = "describe an MLX writer transport";
        let mut output = MlxString::empty();
        // SAFETY: `self` owns a live writer and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_io_writer_tostring(output.raw_mut(), self.raw_writer) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output.to_string_lossy())
    }

    pub(crate) const fn raw(&self) -> raw::mlx_io_writer {
        self.raw_writer
    }
}

impl Drop for MlxIoWriter {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live writer exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_io_writer_free(self.raw_writer);
        }
    }
}

/// Loads one array from an open reader transport.
///
/// # Errors
/// Returns the captured MLX-C description when loading fails.
pub fn load_array_from_reader(
    reader: &MlxIoReader,
    stream: &MlxStream,
) -> Result<MlxArray, MlxCError> {
    let operation: &'static str = "load an MLX array from a reader";
    let mut output = MlxArray::empty();
    // SAFETY: The reader handle is live and the output pointer is valid
    // writable storage.
    let status = unsafe { raw::mlx_load_reader(output.raw_mut(), reader.raw(), stream.raw()) };
    check_status(status, operation)?;
    output.require_populated(operation)?;
    Ok(output)
}

/// Loads the safetensors maps from an open reader transport.
///
/// # Errors
/// Returns the captured MLX-C description when loading fails.
pub fn load_safetensors_from_reader(
    reader: &MlxIoReader,
    stream: &MlxStream,
) -> Result<(MlxMapStringToArray, MlxMapStringToString), MlxCError> {
    let operation: &'static str = "load MLX safetensors maps from a reader";
    let mut arrays = MlxMapStringToArray::new()?;
    let mut metadata = MlxMapStringToString::new()?;
    // SAFETY: The reader handle is live and both output pointers are valid
    // writable storage.
    let status = unsafe {
        raw::mlx_load_safetensors_reader(
            arrays.raw_mut(),
            metadata.raw_mut(),
            reader.raw(),
            stream.raw(),
        )
    };
    check_status(status, operation)?;
    Ok((arrays, metadata))
}

/// Saves one array through an open writer transport.
///
/// # Errors
/// Returns the captured MLX-C description when saving fails.
pub fn save_array_to_writer(writer: &MlxIoWriter, array: &MlxArray) -> Result<(), MlxCError> {
    let operation: &'static str = "save an MLX array through a writer";
    // SAFETY: Both handles are live for the duration of the call.
    let status = unsafe { raw::mlx_save_writer(writer.raw(), array.raw()) };
    check_status(status, operation)
}

/// Saves the parameter and metadata maps through an open writer transport.
///
/// # Errors
/// Returns the captured MLX-C description when saving fails.
pub fn save_safetensors_to_writer(
    writer: &MlxIoWriter,
    arrays: &MlxMapStringToArray,
    metadata: &MlxMapStringToString,
) -> Result<(), MlxCError> {
    let operation: &'static str = "save MLX safetensors maps through a writer";
    // SAFETY: All handles are live for the duration of the call.
    let status =
        unsafe { raw::mlx_save_safetensors_writer(writer.raw(), arrays.raw(), metadata.raw()) };
    check_status(status, operation)
}

/// Owned MLX GGUF container handle released exactly once.
#[derive(Debug)]
pub struct MlxIoGguf(raw::mlx_io_gguf);

impl MlxIoGguf {
    /// Creates an empty GGUF container.
    pub fn new() -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX GGUF container";
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_gguf = unsafe { raw::mlx_io_gguf_new() };
        if raw_gguf.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty GGUF handle".to_owned(),
            });
        }
        Ok(Self(raw_gguf))
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_io_gguf {
        &mut self.0
    }

    /// Loads a GGUF container from a file.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when loading fails.
    pub fn load_from_file(path: &str, stream: &MlxStream) -> Result<Self, MlxCError> {
        let operation: &'static str = "load an MLX GGUF container from a file";
        let path_value = path_argument(path, operation)?;
        let mut output = Self::new()?;
        // SAFETY: The path remains valid for the call, the stream handle is
        // live, and the output pointer is valid writable storage.
        let status =
            unsafe { raw::mlx_load_gguf(output.raw_mut(), path_value.as_ptr(), stream.raw()) };
        check_status(status, operation)?;
        Ok(output)
    }

    /// Saves this GGUF container to a file.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when saving fails.
    pub fn save_to_file(&self, path: &str) -> Result<(), MlxCError> {
        let operation: &'static str = "save an MLX GGUF container to a file";
        let path_value = path_argument(path, operation)?;
        // SAFETY: The path remains valid for the call and the handle is live.
        let status = unsafe { raw::mlx_save_gguf(path_value.as_ptr(), self.0) };
        check_status(status, operation)
    }

    /// Every array key in the container.
    pub fn array_keys(&self) -> Result<MlxVectorString, MlxCError> {
        let operation: &'static str = "list MLX GGUF array keys";
        let mut output = MlxVectorString::empty();
        // SAFETY: `self` owns a live container and the output pointer is
        // valid writable storage.
        let status = unsafe { raw::mlx_io_gguf_get_keys(output.raw_mut(), self.0) };
        check_status(status, operation)?;
        Ok(output)
    }

    /// Copies the array stored under a key.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the key is missing.
    pub fn array_at(&self, key: &str) -> Result<MlxArray, MlxCError> {
        let operation: &'static str = "read an MLX GGUF array";
        let key_argument = path_argument(key, operation)?;
        let mut output = MlxArray::empty();
        // SAFETY: `self` owns a live container, the key outlives the call,
        // and the output pointer is valid writable storage.
        let status =
            unsafe { raw::mlx_io_gguf_get_array(output.raw_mut(), self.0, key_argument.as_ptr()) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Copies the metadata array stored under a key.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the key is missing.
    pub fn metadata_array_at(&self, key: &str) -> Result<MlxArray, MlxCError> {
        let operation: &'static str = "read an MLX GGUF metadata array";
        let key_argument = path_argument(key, operation)?;
        let mut output = MlxArray::empty();
        // SAFETY: `self` owns a live container, the key outlives the call,
        // and the output pointer is valid writable storage.
        let status = unsafe {
            raw::mlx_io_gguf_get_metadata_array(output.raw_mut(), self.0, key_argument.as_ptr())
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Copies the metadata string stored under a key.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the key is missing.
    pub fn metadata_string_at(&self, key: &str) -> Result<String, MlxCError> {
        let operation: &'static str = "read an MLX GGUF metadata string";
        let key_argument = path_argument(key, operation)?;
        let mut output = MlxString::empty();
        // SAFETY: `self` owns a live container, the key outlives the call,
        // and the output pointer is valid writable storage.
        let status = unsafe {
            raw::mlx_io_gguf_get_metadata_string(output.raw_mut(), self.0, key_argument.as_ptr())
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output.to_string_lossy())
    }

    /// Copies the metadata string vector stored under a key.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the key is missing.
    pub fn metadata_string_vector_at(&self, key: &str) -> Result<MlxVectorString, MlxCError> {
        let operation: &'static str = "read an MLX GGUF metadata string vector";
        let key_argument = path_argument(key, operation)?;
        let mut output = MlxVectorString::empty();
        // SAFETY: `self` owns a live container, the key outlives the call,
        // and the output pointer is valid writable storage.
        let status = unsafe {
            raw::mlx_io_gguf_get_metadata_vector_string(
                output.raw_mut(),
                self.0,
                key_argument.as_ptr(),
            )
        };
        check_status(status, operation)?;
        Ok(output)
    }

    /// Whether metadata of the requested kind exists under the key.
    fn has_metadata(
        &self,
        key: &str,
        probe: unsafe extern "C" fn(
            *mut bool,
            raw::mlx_io_gguf,
            *const std::os::raw::c_char,
        ) -> i32,
        operation: &'static str,
    ) -> Result<bool, MlxCError> {
        let key_argument = path_argument(key, operation)?;
        let mut exists = false;
        // SAFETY: `self` owns a live container, the key outlives the call,
        // and the output pointer is valid writable storage.
        let status = unsafe { probe(&mut exists, self.0, key_argument.as_ptr()) };
        check_status(status, operation)?;
        Ok(exists)
    }

    /// Whether array metadata exists under the key.
    pub fn has_metadata_array(&self, key: &str) -> Result<bool, MlxCError> {
        self.has_metadata(
            key,
            raw::mlx_io_gguf_has_metadata_array,
            "probe MLX GGUF array metadata",
        )
    }

    /// Whether string metadata exists under the key.
    pub fn has_metadata_string(&self, key: &str) -> Result<bool, MlxCError> {
        self.has_metadata(
            key,
            raw::mlx_io_gguf_has_metadata_string,
            "probe MLX GGUF string metadata",
        )
    }

    /// Whether string-vector metadata exists under the key.
    pub fn has_metadata_string_vector(&self, key: &str) -> Result<bool, MlxCError> {
        self.has_metadata(
            key,
            raw::mlx_io_gguf_has_metadata_vector_string,
            "probe MLX GGUF string-vector metadata",
        )
    }

    /// Stores one array under the key.
    pub fn set_array(&mut self, key: &str, value: &MlxArray) -> Result<(), MlxCError> {
        let operation: &'static str = "store an MLX GGUF array";
        let key_argument = path_argument(key, operation)?;
        // SAFETY: `self` owns a live container, the key outlives the call,
        // and the array handle is live.
        let status =
            unsafe { raw::mlx_io_gguf_set_array(self.0, key_argument.as_ptr(), value.raw()) };
        check_status(status, operation)
    }

    /// Stores one metadata array under the key.
    pub fn set_metadata_array(&mut self, key: &str, value: &MlxArray) -> Result<(), MlxCError> {
        let operation: &'static str = "store MLX GGUF metadata array";
        let key_argument = path_argument(key, operation)?;
        // SAFETY: `self` owns a live container, the key outlives the call,
        // and the array handle is live.
        let status = unsafe {
            raw::mlx_io_gguf_set_metadata_array(self.0, key_argument.as_ptr(), value.raw())
        };
        check_status(status, operation)
    }

    /// Stores one metadata string under the key.
    pub fn set_metadata_string(&mut self, key: &str, value: &str) -> Result<(), MlxCError> {
        let operation: &'static str = "store MLX GGUF metadata string";
        let key_argument = path_argument(key, operation)?;
        let value_argument = path_argument(value, operation)?;
        // SAFETY: `self` owns a live container and both strings outlive the
        // call.
        let status = unsafe {
            raw::mlx_io_gguf_set_metadata_string(
                self.0,
                key_argument.as_ptr(),
                value_argument.as_ptr(),
            )
        };
        check_status(status, operation)
    }

    /// Stores one metadata string vector under the key.
    pub fn set_metadata_string_vector(
        &mut self,
        key: &str,
        values: &MlxVectorString,
    ) -> Result<(), MlxCError> {
        let operation: &'static str = "store MLX GGUF metadata string vector";
        let key_argument = path_argument(key, operation)?;
        // SAFETY: `self` owns a live container, the key outlives the call,
        // and the vector handle is live.
        let status = unsafe {
            raw::mlx_io_gguf_set_metadata_vector_string(self.0, key_argument.as_ptr(), values.raw())
        };
        check_status(status, operation)
    }
}

impl Drop for MlxIoGguf {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live container exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_io_gguf_free(self.0);
        }
    }
}
