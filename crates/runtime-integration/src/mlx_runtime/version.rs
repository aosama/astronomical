use std::ffi::CStr;

use crate::{MlxRuntimeError, raw};

use super::check_status;

pub(super) fn read_mlx_version() -> Result<String, MlxRuntimeError> {
    // SAFETY: The error handler is installed before this constructor and the
    // returned owned handle is released on every subsequent path.
    let raw_version = unsafe { raw::mlx_string_new() };
    let mut owned_version = OwnedMlxString(raw_version);
    // SAFETY: `owned_version` contains a live MLX string handle and this call
    // only replaces its owned string value.
    let status = unsafe { raw::mlx_version(&mut owned_version.0) };
    check_status(status, "read the linked MLX version")?;
    // SAFETY: The pointer remains valid while `owned_version` is live.
    let version_pointer = unsafe { raw::mlx_string_data(owned_version.0) };
    if version_pointer.is_null() {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "read the linked MLX version",
            description: "MLX returned a null version string".to_owned(),
        });
    }
    // SAFETY: MLX C documents a null-terminated string owned by the handle.
    Ok(unsafe { CStr::from_ptr(version_pointer) }
        .to_string_lossy()
        .into_owned())
}

struct OwnedMlxString(raw::mlx_string);

impl Drop for OwnedMlxString {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live handle exactly once and does not
        // use it afterward. MLX C accepts the value form for destruction.
        unsafe {
            raw::mlx_string_free(self.0);
        }
    }
}
