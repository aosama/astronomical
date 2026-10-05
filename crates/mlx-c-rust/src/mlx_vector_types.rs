//! Owned MLX vector handles for integers, strings, streams, and nested
//! array vectors.

use std::ffi::{CStr, CString};

use crate::error::check_status;
use crate::raw;
use crate::{MlxArrayVector, MlxCError, MlxStream};

/// Owned MLX integer vector handle released exactly once through the C API.
#[derive(Debug)]
pub struct MlxVectorInt(raw::mlx_vector_int);

impl MlxVectorInt {
    /// Creates an empty vector handle.
    pub fn empty() -> Self {
        // SAFETY: The runtime error handler is installed before model loading
        // and the returned handle enters RAII ownership.
        let raw_vector = unsafe { raw::mlx_vector_int_new() };
        Self(raw_vector)
    }

    /// Creates an owned vector holding copies of the provided values.
    pub fn from_values(values: &[i32]) -> Result<Self, MlxCError> {
        // SAFETY: The value storage remains valid for this copying
        // constructor; MLX C declares a mutable pointer but does not mutate
        // the values, and the returned handle enters RAII ownership.
        let raw_vector =
            unsafe { raw::mlx_vector_int_new_data(values.as_ptr().cast_mut(), values.len()) };
        Self::from_raw(raw_vector, "create an MLX integer vector")
    }

    /// Creates an owned vector holding one copied value.
    pub fn from_value(value: i32) -> Result<Self, MlxCError> {
        // SAFETY: The value is copied and the returned handle enters RAII
        // ownership.
        let raw_vector = unsafe { raw::mlx_vector_int_new_value(value) };
        Self::from_raw(raw_vector, "create an MLX integer vector")
    }

    /// Copies the source vector handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned vector value.
        let status = unsafe { raw::mlx_vector_int_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX integer vector")
    }

    /// Replaces the vector contents with copies of the provided values.
    pub fn set_values(&mut self, values: &[i32]) -> Result<(), MlxCError> {
        // SAFETY: The value storage remains valid for this copying call; the
        // mutable pointer is a C declaration quirk and the values are read
        // only.
        let status = unsafe {
            raw::mlx_vector_int_set_data(&mut self.0, values.as_ptr().cast_mut(), values.len())
        };
        check_status(status, "replace MLX integer vector contents")
    }

    /// Replaces the vector contents with one copied value.
    pub fn set_one(&mut self, value: i32) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the value is copied.
        let status = unsafe { raw::mlx_vector_int_set_value(&mut self.0, value) };
        check_status(status, "replace MLX integer vector contents")
    }

    /// Appends copies of the provided values.
    pub fn append_values(&mut self, values: &[i32]) -> Result<(), MlxCError> {
        // SAFETY: The value storage remains valid for this copying call and
        // the values are read only despite the mutable pointer declaration.
        let status = unsafe {
            raw::mlx_vector_int_append_data(self.0, values.as_ptr().cast_mut(), values.len())
        };
        check_status(status, "append to an MLX integer vector")
    }

    /// Appends one copied value.
    pub fn append_one(&mut self, value: i32) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the value is copied.
        let status = unsafe { raw::mlx_vector_int_append_value(self.0, value) };
        check_status(status, "append to an MLX integer vector")
    }

    /// The number of values in the vector.
    #[must_use]
    pub fn len(&self) -> usize {
        // SAFETY: `self` owns a live MLX vector handle.
        unsafe { raw::mlx_vector_int_size(self.0) }
    }

    /// Whether the vector holds no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies the value at `element_index` out of the vector.
    pub fn value_at(&self, element_index: usize) -> Result<i32, MlxCError> {
        let operation: &'static str = "read an MLX integer vector element";
        let mut element = 0_i32;
        // SAFETY: `self` owns a live vector and the output pointer is valid
        // writable storage for one value.
        let status = unsafe { raw::mlx_vector_int_get(&mut element, self.0, element_index) };
        check_status(status, operation)?;
        Ok(element)
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_vector_int {
        &mut self.0
    }

    fn from_raw(
        raw_vector: raw::mlx_vector_int,
        operation: &'static str,
    ) -> Result<Self, MlxCError> {
        if raw_vector.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty vector handle".to_owned(),
            });
        }
        Ok(Self(raw_vector))
    }
}

impl Drop for MlxVectorInt {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live vector exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_vector_int_free(self.0);
        }
    }
}

/// Owned MLX vector of streams; elements are copied out as owned stream
/// handles.
#[derive(Debug)]
pub struct MlxVectorStream(raw::mlx_vector_stream);

impl MlxVectorStream {
    /// Creates an empty vector handle.
    pub fn empty() -> Self {
        // SAFETY: The runtime error handler is installed before model loading
        // and the returned handle enters RAII ownership.
        let raw_vector = unsafe { raw::mlx_vector_stream_new() };
        Self(raw_vector)
    }

    /// Creates an owned vector holding copies of the provided streams.
    pub fn from_streams(streams: &[&MlxStream]) -> Result<Self, MlxCError> {
        let raw_streams: Vec<raw::mlx_stream> = streams.iter().map(|stream| stream.raw()).collect();
        // SAFETY: The handle storage remains valid for this copying
        // constructor and the returned handle enters RAII ownership.
        let raw_vector =
            unsafe { raw::mlx_vector_stream_new_data(raw_streams.as_ptr(), raw_streams.len()) };
        Self::from_raw(raw_vector, "create an MLX stream vector")
    }

    /// Copies the source vector handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned vector value.
        let status = unsafe { raw::mlx_vector_stream_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX stream vector")
    }

    /// Appends one copied stream handle.
    pub fn append_stream(&mut self, stream: &MlxStream) -> Result<(), MlxCError> {
        // SAFETY: The stream handle is live and copied into the vector.
        let status = unsafe { raw::mlx_vector_stream_append_value(self.0, stream.raw()) };
        check_status(status, "append to an MLX stream vector")
    }

    /// The number of streams in the vector.
    #[must_use]
    pub fn len(&self) -> usize {
        // SAFETY: `self` owns a live MLX vector handle.
        unsafe { raw::mlx_vector_stream_size(self.0) }
    }

    /// Whether the vector holds no streams.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies the stream at `element_index` into an owned handle.
    pub fn stream_at(&self, element_index: usize) -> Result<MlxStream, MlxCError> {
        let operation: &'static str = "read an MLX stream vector element";
        let mut raw_stream = raw::mlx_stream {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: `self` owns a live vector and the output pointer is valid
        // writable storage for one stream handle; MLX copies the handle
        // reference into the output.
        let status = unsafe { raw::mlx_vector_stream_get(&mut raw_stream, self.0, element_index) };
        check_status(status, operation)?;
        MlxStream::from_live_raw(raw_stream, operation)
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_vector_stream {
        &mut self.0
    }

    fn from_raw(
        raw_vector: raw::mlx_vector_stream,
        operation: &'static str,
    ) -> Result<Self, MlxCError> {
        if raw_vector.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty vector handle".to_owned(),
            });
        }
        Ok(Self(raw_vector))
    }
}

impl Drop for MlxVectorStream {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live vector exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_vector_stream_free(self.0);
        }
    }
}

/// Owned MLX vector of array vectors.
#[derive(Debug)]
pub struct MlxVectorVectorArray(raw::mlx_vector_vector_array);

impl MlxVectorVectorArray {
    /// Creates an empty vector handle.
    pub fn empty() -> Self {
        // SAFETY: The runtime error handler is installed before model loading
        // and the returned handle enters RAII ownership.
        let raw_vector = unsafe { raw::mlx_vector_vector_array_new() };
        Self(raw_vector)
    }

    /// Creates an owned vector holding copies of the provided vectors.
    pub fn from_vectors(vectors: &[MlxArrayVector]) -> Result<Self, MlxCError> {
        let raw_vectors: Vec<raw::mlx_vector_array> =
            vectors.iter().map(|vector| vector.raw()).collect();
        // SAFETY: The handle storage remains valid for this copying
        // constructor and the returned handle enters RAII ownership.
        let raw_vector = unsafe {
            raw::mlx_vector_vector_array_new_data(raw_vectors.as_ptr(), raw_vectors.len())
        };
        Self::from_raw(raw_vector, "create an MLX nested array vector")
    }

    /// Copies the source vector handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned vector value.
        let status = unsafe { raw::mlx_vector_vector_array_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX nested array vector")
    }

    /// The number of vectors in this vector.
    #[must_use]
    pub fn len(&self) -> usize {
        // SAFETY: `self` owns a live MLX vector handle.
        unsafe { raw::mlx_vector_vector_array_size(self.0) }
    }

    /// Whether the vector holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies the vector at `element_index` into an owned handle.
    pub fn vector_at(&self, element_index: usize) -> Result<MlxArrayVector, MlxCError> {
        let operation: &'static str = "read an MLX nested array vector element";
        let mut raw_vector = raw::mlx_vector_array {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: `self` owns a live vector and the output pointer is valid
        // writable storage for one vector handle; MLX copies the handle
        // reference into the output.
        let status =
            unsafe { raw::mlx_vector_vector_array_get(&mut raw_vector, self.0, element_index) };
        check_status(status, operation)?;
        MlxArrayVector::from_live_raw(raw_vector, operation)
    }

    fn from_raw(
        raw_vector: raw::mlx_vector_vector_array,
        operation: &'static str,
    ) -> Result<Self, MlxCError> {
        if raw_vector.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty vector handle".to_owned(),
            });
        }
        Ok(Self(raw_vector))
    }
}

impl Drop for MlxVectorVectorArray {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live vector exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_vector_vector_array_free(self.0);
        }
    }
}

/// Owned MLX vector of strings; elements are borrowed from the vector for
/// each read and copied into Rust memory.
#[derive(Debug)]
pub struct MlxVectorString(raw::mlx_vector_string);

impl MlxVectorString {
    /// Creates an empty vector handle.
    pub fn empty() -> Self {
        // SAFETY: The runtime error handler is installed before model loading
        // and the returned handle enters RAII ownership.
        let raw_vector = unsafe { raw::mlx_vector_string_new() };
        Self(raw_vector)
    }

    /// Creates an owned vector holding copies of the provided strings.
    pub fn from_strings(values: &[&str]) -> Result<Self, MlxCError> {
        let owned_values: Vec<std::ffi::CString> = values
            .iter()
            .map(|value| {
                std::ffi::CString::new(*value).map_err(|_| MlxCError {
                    operation: "create an MLX string vector",
                    description: "string contains an interior null byte".to_owned(),
                })
            })
            .collect::<Result<_, _>>()?;
        let value_pointers: Vec<*const std::os::raw::c_char> = owned_values
            .iter()
            .map(|owned_value| owned_value.as_ptr())
            .collect();
        // SAFETY: The pointer storage remains valid for this copying
        // constructor and the returned handle enters RAII ownership. The
        // mutable pointer is a C declaration quirk; the strings are read
        // only.
        let raw_vector = unsafe {
            raw::mlx_vector_string_new_data(
                value_pointers.as_ptr().cast_mut(),
                value_pointers.len(),
            )
        };
        Self::from_raw(raw_vector, "create an MLX string vector")
    }

    /// Copies the source vector handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the owned
        // vector value.
        let status = unsafe { raw::mlx_vector_string_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX string vector")
    }

    /// Replaces the vector contents with copies of the provided strings.
    pub fn set_strings(&mut self, values: &[&str]) -> Result<(), MlxCError> {
        let mut replacement = Self::from_strings(values)?;
        std::mem::swap(&mut self.0, &mut replacement.0);
        Ok(())
    }

    /// Replaces the vector contents with one copied string.
    pub fn set_one(&mut self, value: &str) -> Result<(), MlxCError> {
        let value_argument = CString::new(value).map_err(|_| MlxCError {
            operation: "replace MLX string vector contents",
            description: "string contains an interior null byte".to_owned(),
        })?;
        // SAFETY: The handle is live and the string is copied.
        let status =
            unsafe { raw::mlx_vector_string_set_value(&mut self.0, value_argument.as_ptr()) };
        check_status(status, "replace MLX string vector contents")
    }

    /// Appends one copied string.
    pub fn append_one(&mut self, value: &str) -> Result<(), MlxCError> {
        let value_argument = CString::new(value).map_err(|_| MlxCError {
            operation: "append to an MLX string vector",
            description: "string contains an interior null byte".to_owned(),
        })?;
        // SAFETY: The handle is live and the string is copied.
        let status =
            unsafe { raw::mlx_vector_string_append_value(self.0, value_argument.as_ptr()) };
        check_status(status, "append to an MLX string vector")
    }

    /// The number of strings in the vector.
    #[must_use]
    pub fn len(&self) -> usize {
        // SAFETY: `self` owns a live MLX vector handle.
        unsafe { raw::mlx_vector_string_size(self.0) }
    }

    /// Whether the vector holds no strings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies the string at `element_index` out of the vector.
    ///
    /// The C API loans the text from the vector's storage, so the copy
    /// happens while the vector is alive and this call holds the borrow.
    pub fn string_at(&self, element_index: usize) -> Result<String, MlxCError> {
        let operation: &'static str = "read an MLX string vector element";
        let mut element_pointer: *mut std::os::raw::c_char = std::ptr::null_mut();
        // SAFETY: `self` owns a live vector and the output pointer is valid
        // writable storage for one borrowed string pointer.
        let status =
            unsafe { raw::mlx_vector_string_get(&mut element_pointer, self.0, element_index) };
        check_status(status, operation)?;
        if element_pointer.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned a null string element".to_owned(),
            });
        }
        // SAFETY: MLX loans a null-terminated element owned by the live
        // vector; the text is copied before this call returns.
        let borrowed = unsafe { CStr::from_ptr(element_pointer) };
        Ok(borrowed.to_string_lossy().into_owned())
    }

    pub(crate) const fn raw(&self) -> raw::mlx_vector_string {
        self.0
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_vector_string {
        &mut self.0
    }

    fn from_raw(
        raw_vector: raw::mlx_vector_string,
        operation: &'static str,
    ) -> Result<Self, MlxCError> {
        if raw_vector.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty vector handle".to_owned(),
            });
        }
        Ok(Self(raw_vector))
    }
}

impl Drop for MlxVectorString {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live vector exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_vector_string_free(self.0);
        }
    }
}
