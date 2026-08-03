use std::ffi::c_char;

use crate::{
    MlxRuntimeError,
    mlx_runtime::{
        check_status, configure_metallib_path, configured_metallib_path,
        install_non_terminating_error_handler,
    },
    raw,
};

const MAXIMUM_RECOMMENDED_WORKING_SET_SIZE_DEVICE_INFO_KEY: &[u8] =
    b"max_recommended_working_set_size\0";

/// Reads Metal's effective recommended GPU working-set size through MLX-C.
///
/// MLX-C exposes a generic device-information map. The
/// `max_recommended_working_set_size` entry originates from Metal's device
/// recommendation and is measured in bytes. It is not a hardcoded model or
/// laptop value, and reading it does not change either the macOS sysctl or the
/// MLX process residency set. Worker startup uses it only when
/// `iogpu.wired_limit_mb` reports the zero-valued default-policy sentinel.
pub fn maximum_recommended_gpu_working_set_size_bytes() -> Result<usize, MlxRuntimeError> {
    install_non_terminating_error_handler();
    let metallib_path = configured_metallib_path()?;
    configure_metallib_path(&metallib_path)?;
    // SAFETY: `MLX_GPU` with index 0 is the same default GPU target used by the
    // rest of the runtime, and the returned handle is owned below.
    let raw_gpu_device = unsafe { raw::mlx_device_new_type(raw::mlx_device_type__MLX_GPU, 0) };
    let owned_gpu_device = OwnedMlxDevice(raw_gpu_device);

    // SAFETY: The returned empty device-info handle is owned below and filled by
    // `mlx_device_info_get` before any key lookup.
    let raw_device_info = unsafe { raw::mlx_device_info_new() };
    let mut owned_device_info = OwnedMlxDeviceInfo(raw_device_info);
    // SAFETY: Both handles are live for the duration of the call, and the output
    // pointer refers to owned writable storage.
    let device_info_status =
        unsafe { raw::mlx_device_info_get(&mut owned_device_info.0, owned_gpu_device.0) };
    check_status(device_info_status, "read MLX GPU device info")?;

    let mut maximum_recommended_working_set_size_bytes = 0;
    // SAFETY: The key is a static NUL-terminated string and the output pointer
    // refers to initialized writable storage.
    let working_set_status = unsafe {
        raw::mlx_device_info_get_size(
            &mut maximum_recommended_working_set_size_bytes,
            owned_device_info.0,
            MAXIMUM_RECOMMENDED_WORKING_SET_SIZE_DEVICE_INFO_KEY
                .as_ptr()
                .cast::<c_char>(),
        )
    };
    if working_set_status == 2 {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "read MLX GPU recommended working set",
            description:
                "MLX device info does not expose size key max_recommended_working_set_size"
                    .to_owned(),
        });
    }
    check_status(working_set_status, "read MLX GPU recommended working set")?;
    if maximum_recommended_working_set_size_bytes == 0 {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "read MLX GPU recommended working set",
            description: "MLX returned a zero-byte recommended working set".to_owned(),
        });
    }
    Ok(maximum_recommended_working_set_size_bytes)
}

struct OwnedMlxDevice(raw::mlx_device);

impl Drop for OwnedMlxDevice {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live device handle exactly once and
        // does not use it afterward.
        unsafe {
            raw::mlx_device_free(self.0);
        }
    }
}

struct OwnedMlxDeviceInfo(raw::mlx_device_info);

impl Drop for OwnedMlxDeviceInfo {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live device-info handle exactly once
        // and does not use it afterward.
        unsafe {
            raw::mlx_device_info_free(self.0);
        }
    }
}
