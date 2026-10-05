//! The MLX device domain: device handles, device information, and the
//! process default device.

use std::ffi::{CStr, CString};

use crate::MlxCError;
use crate::error::check_status;
use crate::mlx_vector_types::MlxVectorString;
use crate::raw;

/// Owned MLX device handle released exactly once through the official C API.
#[derive(Debug)]
pub struct MlxDeviceHandle(raw::mlx_device);

impl MlxDeviceHandle {
    /// Creates an empty device handle for an output parameter.
    pub(crate) fn empty() -> Self {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_device = unsafe { raw::mlx_device_new() };
        Self(raw_device)
    }

    /// Creates an owned device handle for one backend device type and index.
    pub fn of_type(
        device_type: raw::mlx_device_type,
        device_index: i32,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX device handle";
        // SAFETY: The device request takes plain values and the returned
        // handle enters RAII ownership.
        let raw_device = unsafe { raw::mlx_device_new_type(device_type, device_index) };
        Self::from_live_raw(raw_device, operation)
    }

    /// The process's default device.
    pub fn default() -> Result<Self, MlxCError> {
        let operation: &'static str = "acquire the MLX default device";
        let mut output = Self::empty();
        // SAFETY: The output pointer is valid writable storage for one
        // device handle.
        let status = unsafe { raw::mlx_get_default_device(output.raw_mut()) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Replaces the process's default device.
    pub fn set_as_default(&self) -> Result<(), MlxCError> {
        // SAFETY: The device handle is live.
        let status = unsafe { raw::mlx_set_default_device(self.0) };
        check_status(status, "set the MLX default device")
    }

    /// Copies the source device handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned device value.
        let status = unsafe { raw::mlx_device_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX device handle")
    }

    /// Whether both handles reference the same device.
    #[must_use]
    pub fn equals(&self, other: &Self) -> bool {
        // SAFETY: Both handles are live; equality is a pure query.
        unsafe { raw::mlx_device_equal(self.0, other.0) }
    }

    /// The device's index within its backend.
    pub fn index(&self) -> Result<i32, MlxCError> {
        let operation: &'static str = "read the MLX device index";
        let mut device_index: i32 = 0;
        // SAFETY: `self` owns a live device and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_device_get_index(&mut device_index, self.0) };
        check_status(status, operation)?;
        Ok(device_index)
    }

    /// The device's backend type.
    pub fn device_type(&self) -> Result<raw::mlx_device_type, MlxCError> {
        let operation: &'static str = "read the MLX device type";
        let mut device_type = raw::mlx_device_type__MLX_CPU;
        // SAFETY: `self` owns a live device and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_device_get_type(&mut device_type, self.0) };
        check_status(status, operation)?;
        Ok(device_type)
    }

    /// Whether the backend device is available in this process.
    pub fn is_available(&self) -> Result<bool, MlxCError> {
        let operation: &'static str = "query MLX device availability";
        let mut available = false;
        // SAFETY: `self` owns a live device and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_device_is_available(&mut available, self.0) };
        check_status(status, operation)?;
        Ok(available)
    }

    /// How many devices the backend exposes.
    pub fn count_of_type(device_type: raw::mlx_device_type) -> Result<i32, MlxCError> {
        let operation: &'static str = "count MLX devices";
        let mut device_count: i32 = 0;
        // SAFETY: The output pointer is valid writable storage.
        let status = unsafe { raw::mlx_device_count(&mut device_count, device_type) };
        check_status(status, operation)?;
        Ok(device_count)
    }

    /// Renders the device description text.
    pub fn to_string_lossy(&self) -> Result<String, MlxCError> {
        let operation: &'static str = "describe an MLX device";
        let mut output = crate::MlxString::empty();
        // SAFETY: `self` owns a live device and the output pointer is valid
        // writable storage for one string handle.
        let status = unsafe { raw::mlx_device_tostring(output.raw_mut(), self.0) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output.to_string_lossy())
    }

    /// The device's key/value information table.
    pub fn information(&self) -> Result<MlxDeviceInfo, MlxCError> {
        let operation: &'static str = "read MLX device information";
        let mut output = MlxDeviceInfo::empty();
        // SAFETY: `self` owns a live device and the output pointer is valid
        // writable storage for one device-info handle.
        let status = unsafe { raw::mlx_device_info_get(output.raw_mut(), self.0) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    pub(crate) const fn raw(&self) -> raw::mlx_device {
        self.0
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_device {
        &mut self.0
    }

    pub(crate) fn from_live_raw(
        raw_device: raw::mlx_device,
        operation: &'static str,
    ) -> Result<Self, MlxCError> {
        if raw_device.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty device handle".to_owned(),
            });
        }
        Ok(Self(raw_device))
    }

    fn require_populated(&self, operation: &'static str) -> Result<(), MlxCError> {
        if self.0.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty device handle".to_owned(),
            });
        }
        Ok(())
    }
}

impl Drop for MlxDeviceHandle {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live device exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_device_free(self.0);
        }
    }
}

/// Owned MLX device information table released exactly once.
#[derive(Debug)]
pub struct MlxDeviceInfo(raw::mlx_device_info);

impl MlxDeviceInfo {
    /// Creates an empty device-info handle for an output parameter.
    pub(crate) fn empty() -> Self {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_info = unsafe { raw::mlx_device_info_new() };
        Self(raw_info)
    }

    /// Whether the information table contains the key.
    pub fn has_key(&self, key: &str) -> Result<bool, MlxCError> {
        let operation: &'static str = "query MLX device information keys";
        let key_argument = CString::new(key).map_err(|_| MlxCError {
            operation,
            description: "key contains an interior null byte".to_owned(),
        })?;
        let mut exists = false;
        // SAFETY: `self` owns a live info table, the key outlives the call,
        // and the output pointer is valid writable storage.
        let status =
            unsafe { raw::mlx_device_info_has_key(&mut exists, self.0, key_argument.as_ptr()) };
        check_status(status, operation)?;
        Ok(exists)
    }

    /// Whether the value stored for the key is a string.
    pub fn is_string(&self, key: &str) -> Result<bool, MlxCError> {
        let operation: &'static str = "query MLX device information value types";
        let key_argument = CString::new(key).map_err(|_| MlxCError {
            operation,
            description: "key contains an interior null byte".to_owned(),
        })?;
        let mut is_string = false;
        // SAFETY: `self` owns a live info table, the key outlives the call,
        // and the output pointer is valid writable storage.
        let status = unsafe {
            raw::mlx_device_info_is_string(&mut is_string, self.0, key_argument.as_ptr())
        };
        check_status(status, operation)?;
        Ok(is_string)
    }

    /// Copies the string value stored for the key.
    pub fn string_value(&self, key: &str) -> Result<String, MlxCError> {
        let operation: &'static str = "read an MLX device information string";
        let key_argument = CString::new(key).map_err(|_| MlxCError {
            operation,
            description: "key contains an interior null byte".to_owned(),
        })?;
        let mut value_pointer: *const std::os::raw::c_char = std::ptr::null();
        // SAFETY: `self` owns a live info table, the key outlives the call,
        // and the output pointer is valid writable storage.
        let status = unsafe {
            raw::mlx_device_info_get_string(&mut value_pointer, self.0, key_argument.as_ptr())
        };
        check_status(status, operation)?;
        if value_pointer.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned a null device information string".to_owned(),
            });
        }
        // SAFETY: MLX loans a null-terminated value owned by the live info
        // table; the text is copied before this call returns.
        let borrowed = unsafe { CStr::from_ptr(value_pointer) };
        Ok(borrowed.to_string_lossy().into_owned())
    }

    /// Copies the size value stored for the key.
    pub fn size_value(&self, key: &str) -> Result<usize, MlxCError> {
        let operation: &'static str = "read an MLX device information size";
        let key_argument = CString::new(key).map_err(|_| MlxCError {
            operation,
            description: "key contains an interior null byte".to_owned(),
        })?;
        let mut value: usize = 0;
        // SAFETY: `self` owns a live info table, the key outlives the call,
        // and the output pointer is valid writable storage.
        let status =
            unsafe { raw::mlx_device_info_get_size(&mut value, self.0, key_argument.as_ptr()) };
        check_status(status, operation)?;
        Ok(value)
    }

    /// Every key present in the information table.
    pub fn keys(&self) -> Result<MlxVectorString, MlxCError> {
        let operation: &'static str = "list MLX device information keys";
        let mut output = MlxVectorString::empty();
        // SAFETY: `self` owns a live info table and the output pointer is
        // valid writable storage for one string vector handle.
        let status = unsafe { raw::mlx_device_info_get_keys(output.raw_mut(), self.0) };
        check_status(status, operation)?;
        Ok(output)
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_device_info {
        &mut self.0
    }

    fn require_populated(&self, operation: &'static str) -> Result<(), MlxCError> {
        if self.0.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty device info handle".to_owned(),
            });
        }
        Ok(())
    }
}

impl Drop for MlxDeviceInfo {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live info table exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_device_info_free(self.0);
        }
    }
}
