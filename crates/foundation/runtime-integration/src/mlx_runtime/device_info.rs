//! Reads two facts MLX's Metal device reports through MLX-C: the advertised
//! recommended GPU working-set size, and the GPU's architecture name.
//!
//! # Diagnostic-only by policy
//!
//! Nothing in the memory policy consumes this recommendation — the machine
//! ceiling is the whole of physical RAM (#1902), resolved by
//! the worker. This probe exists so measurements and probes can report what
//! the device *advertises* (a machine-independent fact) rather than a
//! hardcoded model or laptop value, and reading it changes neither kernel
//! settings nor the MLX process residency set.
//!
//! The one production consumer is the worker's wired-residency claim
//! (`apps/inference-worker/src/worker_startup.rs`): Metal rejects wired-limit
//! claims above the recommended working set, so the recommendation is the
//! absolute wiring ceiling even on machines where the configured memory
//! ceiling sits higher. Capping the claim at the recommendation keeps the
//! wired budget and the serving ceiling in lockstep — the divergence between
//! those two numbers was the 2026-10-16 fatal gap recorded in issue #1297.
//!
//! The architecture name (`applegpu_g17s` on an M5 Pro) tells model loading
//! which GPU generation it runs on, which the engine's automatic prompt chunk
//! depends on (`crate::apple_gpu_generation`).
//!
//! # Why this runs standalone, without an [`MlxRuntime`]
//!
//! The wired-residency claim happens in worker startup *before* any engine
//! constructs a runtime, so the probe performs the minimal native
//! initialization itself: it ensures the error handler and Metallib path are
//! installed (both idempotent — see `error_classification` and `metallib`)
//! and then queries device info through raw MLX-C handles it owns for the
//! duration of the call.

use std::ffi::c_char;

use crate::{
    MlxRuntimeError,
    mlx_runtime::{check_status, configure_metallib_path, configured_metallib_path},
};
use mlx_bridge::raw;

/// The Metal device-info key for the recommended working-set size, as a
/// NUL-terminated byte string ready for the C boundary.
const MAXIMUM_RECOMMENDED_WORKING_SET_SIZE_DEVICE_INFO_KEY: &[u8] =
    b"max_recommended_working_set_size\0";

/// The Metal device-info key for the GPU's architecture name.
const ARCHITECTURE_DEVICE_INFO_KEY: &[u8] = b"architecture\0";

/// Reads Metal's effective recommended GPU working-set size through MLX-C.
///
/// # Errors
///
/// - [`MlxRuntimeError::InvalidMetallibPath`] when no valid metallib can be
///   resolved (this probe may be the first native call in the process).
/// - [`MlxRuntimeError::RuntimeOperation`] when the device does not expose
///   the key (status 2), reports a zero working set, or the query fails.
pub fn maximum_recommended_gpu_working_set_size_bytes() -> Result<usize, MlxRuntimeError> {
    let gpu_device_info = read_gpu_device_info()?;
    let mut maximum_recommended_working_set_size_bytes = 0;
    // SAFETY: The key is a static NUL-terminated string and the output pointer
    // refers to initialized writable storage.
    let working_set_status = unsafe {
        raw::mlx_device_info_get_size(
            &mut maximum_recommended_working_set_size_bytes,
            gpu_device_info.device_info.0,
            MAXIMUM_RECOMMENDED_WORKING_SET_SIZE_DEVICE_INFO_KEY
                .as_ptr()
                .cast::<c_char>(),
        )
    };
    // MLX-C reports "key not present" as status 2; distinguishing it from a
    // generic failure keeps the error actionable on machines whose Metal
    // implementation omits the key.
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

/// Reads the GPU's architecture name as MLX's Metal device reports it, such
/// as `applegpu_g17s`. MLX's own choice of kernels (its M5 tensor-unit
/// kernels among them) is made from this same name.
///
/// # Errors
///
/// - [`MlxRuntimeError::InvalidMetallibPath`] when no valid metallib can be
///   resolved.
/// - [`MlxRuntimeError::RuntimeOperation`] when the device does not report an
///   architecture or the query fails.
pub fn gpu_architecture_name() -> Result<String, MlxRuntimeError> {
    let gpu_device_info = read_gpu_device_info()?;
    let mut architecture_name_pointer: *const c_char = std::ptr::null();
    // SAFETY: The key is a static NUL-terminated string; MLX-C points the
    // output at a string the device-info handle owns, which stays alive until
    // `gpu_device_info` drops at the end of this function.
    let architecture_status = unsafe {
        raw::mlx_device_info_get_string(
            &mut architecture_name_pointer,
            gpu_device_info.device_info.0,
            ARCHITECTURE_DEVICE_INFO_KEY.as_ptr().cast::<c_char>(),
        )
    };
    check_status(architecture_status, "read MLX GPU architecture")?;
    if architecture_name_pointer.is_null() {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "read MLX GPU architecture",
            description: "MLX returned no architecture name".to_owned(),
        });
    }
    // SAFETY: MLX-C returned a non-null NUL-terminated string that the
    // device-info handle still owns; it is copied before the handle drops.
    let architecture_name = unsafe { std::ffi::CStr::from_ptr(architecture_name_pointer) };
    Ok(architecture_name.to_string_lossy().into_owned())
}

/// The default GPU's device info, filled, with the handles that own it.
struct GpuDeviceInfo {
    device_info: OwnedMlxDeviceInfo,
    _gpu_device: OwnedMlxDevice,
}

/// Performs the minimal native initialization (error handler and Metallib
/// path, both idempotent) and fills the default GPU's device info.
fn read_gpu_device_info() -> Result<GpuDeviceInfo, MlxRuntimeError> {
    mlx_bridge::install_non_terminating_error_handler();
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
    Ok(GpuDeviceInfo {
        device_info: owned_device_info,
        _gpu_device: owned_gpu_device,
    })
}

/// RAII owner for a raw MLX device handle.
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

/// RAII owner for a raw MLX device-info handle.
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
