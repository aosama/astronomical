//! The MLX string-keyed map domain: array-valued and string-valued maps
//! with their iterators.

use std::ffi::{CStr, CString};

use crate::error::check_status;
use crate::raw;
use crate::{MlxArray, MlxCError};

/// Owned MLX map from strings to arrays, released exactly once.
#[derive(Debug)]
pub struct MlxMapStringToArray(raw::mlx_map_string_to_array);

impl MlxMapStringToArray {
    /// Creates an empty map.
    pub fn new() -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX string-to-array map";
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_map = unsafe { raw::mlx_map_string_to_array_new() };
        if raw_map.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty map handle".to_owned(),
            });
        }
        Ok(Self(raw_map))
    }

    /// Copies the source map handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned map value.
        let status = unsafe { raw::mlx_map_string_to_array_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX string-to-array map")
    }

    /// Inserts one copied array under the key.
    pub fn insert(&mut self, key: &str, value: &MlxArray) -> Result<(), MlxCError> {
        let operation: &'static str = "insert into an MLX string-to-array map";
        let key_argument = CString::new(key).map_err(|_| MlxCError {
            operation,
            description: "key contains an interior null byte".to_owned(),
        })?;
        // SAFETY: The map handle is live, the key outlives the call, and the
        // array handle is copied.
        let status = unsafe {
            raw::mlx_map_string_to_array_insert(self.0, key_argument.as_ptr(), value.raw())
        };
        check_status(status, operation)
    }

    /// Copies the array stored under the key, if any.
    pub fn value_at(&self, key: &str) -> Result<Option<MlxArray>, MlxCError> {
        let operation: &'static str = "read an MLX string-to-array map entry";
        let key_argument = CString::new(key).map_err(|_| MlxCError {
            operation,
            description: "key contains an interior null byte".to_owned(),
        })?;
        let mut output = MlxArray::empty();
        // SAFETY: The map handle is live, the key outlives the call, and the
        // output pointer is valid writable storage for one array handle.
        let status = unsafe {
            raw::mlx_map_string_to_array_get(output.raw_mut(), self.0, key_argument.as_ptr())
        };
        check_status(status, operation)?;
        if output.is_empty() {
            return Ok(None);
        }
        Ok(Some(output))
    }

    /// An owned iterator over the map's entries.
    pub fn iter(&self) -> Result<MlxMapStringToArrayIterator, MlxCError> {
        let operation: &'static str = "iterate an MLX string-to-array map";
        // SAFETY: The map handle is live and the returned iterator enters
        // RAII ownership.
        let raw_iterator = unsafe { raw::mlx_map_string_to_array_iterator_new(self.0) };
        if raw_iterator.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty map iterator handle".to_owned(),
            });
        }
        Ok(MlxMapStringToArrayIterator(raw_iterator))
    }

    pub(crate) const fn raw(&self) -> raw::mlx_map_string_to_array {
        self.0
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_map_string_to_array {
        &mut self.0
    }

    /// A borrowed view over a live map handle owned elsewhere.
    pub(crate) fn from_borrowed_raw(raw_map: raw::mlx_map_string_to_array) -> Self {
        Self(raw_map)
    }
}

impl Drop for MlxMapStringToArray {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live map exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_map_string_to_array_free(self.0);
        }
    }
}

/// Owned iterator over an MLX string-to-array map.
#[derive(Debug)]
pub struct MlxMapStringToArrayIterator(raw::mlx_map_string_to_array_iterator);

impl MlxMapStringToArrayIterator {
    /// Advances the iterator, copying the next entry if one remains.
    pub fn next_entry(&mut self) -> Result<Option<(String, MlxArray)>, MlxCError> {
        let operation: &'static str = "advance an MLX string-to-array map iterator";
        let mut key_pointer: *const std::os::raw::c_char = std::ptr::null();
        let mut value = MlxArray::empty();
        // SAFETY: The iterator handle is live, and both output pointers are
        // valid writable storage for one borrowed key and one array handle.
        let status = unsafe {
            raw::mlx_map_string_to_array_iterator_next(&mut key_pointer, value.raw_mut(), self.0)
        };
        check_status(status, operation)?;
        if key_pointer.is_null() {
            return Ok(None);
        }
        // SAFETY: MLX loans the null-terminated key from the live map; the
        // text is copied before this call returns.
        let key = unsafe { CStr::from_ptr(key_pointer) }
            .to_string_lossy()
            .into_owned();
        Ok(Some((key, value)))
    }
}

impl Drop for MlxMapStringToArrayIterator {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live iterator exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_map_string_to_array_iterator_free(self.0);
        }
    }
}

/// Owned MLX map from strings to strings, released exactly once.
#[derive(Debug)]
pub struct MlxMapStringToString(raw::mlx_map_string_to_string);

impl MlxMapStringToString {
    /// Creates an empty map.
    pub fn new() -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX string-to-string map";
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_map = unsafe { raw::mlx_map_string_to_string_new() };
        if raw_map.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty map handle".to_owned(),
            });
        }
        Ok(Self(raw_map))
    }

    /// Copies the source map handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned map value.
        let status = unsafe { raw::mlx_map_string_to_string_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX string-to-string map")
    }

    /// Inserts one copied string value under the key.
    pub fn insert(&mut self, key: &str, value: &str) -> Result<(), MlxCError> {
        let operation: &'static str = "insert into an MLX string-to-string map";
        let key_argument = CString::new(key).map_err(|_| MlxCError {
            operation,
            description: "key contains an interior null byte".to_owned(),
        })?;
        let value_argument = CString::new(value).map_err(|_| MlxCError {
            operation,
            description: "value contains an interior null byte".to_owned(),
        })?;
        // SAFETY: The map handle is live and both strings outlive the call.
        let status = unsafe {
            raw::mlx_map_string_to_string_insert(
                self.0,
                key_argument.as_ptr(),
                value_argument.as_ptr(),
            )
        };
        check_status(status, operation)
    }

    /// Copies the string stored under the key, if any.
    pub fn value_at(&self, key: &str) -> Result<Option<String>, MlxCError> {
        let operation: &'static str = "read an MLX string-to-string map entry";
        let key_argument = CString::new(key).map_err(|_| MlxCError {
            operation,
            description: "key contains an interior null byte".to_owned(),
        })?;
        let mut value_pointer: *const std::os::raw::c_char = std::ptr::null();
        // SAFETY: The map handle is live, the key outlives the call, and the
        // output pointer is valid writable storage.
        let status = unsafe {
            raw::mlx_map_string_to_string_get(&mut value_pointer, self.0, key_argument.as_ptr())
        };
        check_status(status, operation)?;
        if value_pointer.is_null() {
            return Ok(None);
        }
        // SAFETY: MLX loans the null-terminated value from the live map; the
        // text is copied before this call returns.
        let borrowed = unsafe { CStr::from_ptr(value_pointer) };
        Ok(Some(borrowed.to_string_lossy().into_owned()))
    }

    /// An owned iterator over the map's entries.
    pub fn iter(&self) -> Result<MlxMapStringToStringIterator, MlxCError> {
        let operation: &'static str = "iterate an MLX string-to-string map";
        // SAFETY: The map handle is live and the returned iterator enters
        // RAII ownership.
        let raw_iterator = unsafe { raw::mlx_map_string_to_string_iterator_new(self.0) };
        if raw_iterator.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty map iterator handle".to_owned(),
            });
        }
        Ok(MlxMapStringToStringIterator(raw_iterator))
    }

    pub(crate) const fn raw(&self) -> raw::mlx_map_string_to_string {
        self.0
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_map_string_to_string {
        &mut self.0
    }
}

impl Drop for MlxMapStringToString {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live map exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_map_string_to_string_free(self.0);
        }
    }
}

/// Owned iterator over an MLX string-to-string map.
#[derive(Debug)]
pub struct MlxMapStringToStringIterator(raw::mlx_map_string_to_string_iterator);

impl MlxMapStringToStringIterator {
    /// Advances the iterator, copying the next entry if one remains.
    pub fn next_entry(&mut self) -> Result<Option<(String, String)>, MlxCError> {
        let operation: &'static str = "advance an MLX string-to-string map iterator";
        let mut key_pointer: *const std::os::raw::c_char = std::ptr::null();
        let mut value_pointer: *const std::os::raw::c_char = std::ptr::null();
        // SAFETY: The iterator handle is live and both output pointers are
        // valid writable storage for borrowed key and value pointers.
        let status = unsafe {
            raw::mlx_map_string_to_string_iterator_next(
                &mut key_pointer,
                &mut value_pointer,
                self.0,
            )
        };
        check_status(status, operation)?;
        if key_pointer.is_null() {
            return Ok(None);
        }
        // SAFETY: MLX loans the null-terminated entries from the live map;
        // both texts are copied before this call returns.
        let key = unsafe { CStr::from_ptr(key_pointer) }
            .to_string_lossy()
            .into_owned();
        let value = if value_pointer.is_null() {
            String::new()
        } else {
            // SAFETY: The value pointer loan is owned by the same live map.
            unsafe { CStr::from_ptr(value_pointer) }
                .to_string_lossy()
                .into_owned()
        };
        Ok(Some((key, value)))
    }
}

impl Drop for MlxMapStringToStringIterator {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live iterator exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_map_string_to_string_iterator_free(self.0);
        }
    }
}
