//! The remaining MLX platform surfaces: Metal queries, wired memory limits,
//! CUDA availability, and the distributed collective domain.

use std::ffi::CString;

use crate::error::check_status;
use crate::raw;
use crate::{MlxArray, MlxCError, MlxDtype, MlxStream};

/// The linked MLX-Metal backend's metallib path.
///
/// # Errors
/// Returns the captured MLX-C description when the query fails.
pub fn metallib_path() -> Result<String, MlxCError> {
    let operation: &'static str = "read the MLX metallib path";
    let mut output = crate::MlxString::empty();
    // SAFETY: The output pointer is valid writable storage for one string
    // handle.
    let status = unsafe { raw::mlx_metal_get_metallib_path(output.raw_mut()) };
    check_status(status, operation)?;
    output.require_populated(operation)?;
    Ok(output.to_string_lossy())
}

/// Whether the MLX-Metal backend is available in this process.
///
/// # Errors
/// Returns the captured MLX-C description when the query fails.
pub fn metal_is_available() -> Result<bool, MlxCError> {
    let operation: &'static str = "query MLX Metal availability";
    let mut available = false;
    // SAFETY: The output pointer is valid writable storage.
    let status = unsafe { raw::mlx_metal_is_available(&mut available) };
    check_status(status, operation)?;
    Ok(available)
}

/// Raises the wired-memory ceiling for MLX allocations.
///
/// Returns the previously configured wired limit.
///
/// # Errors
/// Returns the captured MLX-C description when the platform rejects the
/// request.
pub fn set_wired_limit(limit_bytes: usize) -> Result<usize, MlxCError> {
    let operation: &'static str = "set the MLX wired memory limit";
    let mut previous_limit: usize = 0;
    // SAFETY: The output pointer is valid writable storage and the limit is
    // a plain value.
    let status = unsafe { raw::mlx_set_wired_limit(&mut previous_limit, limit_bytes) };
    check_status(status, operation)?;
    Ok(previous_limit)
}

/// Whether the CUDA backend reports availability in this process.
///
/// # Errors
/// Returns the captured MLX-C description when the query fails.
pub fn cuda_is_available() -> Result<bool, MlxCError> {
    let operation: &'static str = "query MLX CUDA availability";
    let mut available = false;
    // SAFETY: The output pointer is valid writable storage.
    let status = unsafe { raw::mlx_cuda_is_available(&mut available) };
    check_status(status, operation)?;
    Ok(available)
}

/// A backend name argument translated into a C string.
fn backend_argument(backend: &str, operation: &'static str) -> Result<CString, MlxCError> {
    CString::new(backend).map_err(|_| MlxCError {
        operation,
        description: "backend name contains an interior null byte".to_owned(),
    })
}

/// Whether MLX's distributed backend is initialized in this process.
///
/// # Errors
/// Returns the captured MLX-C description when the query fails.
pub fn distributed_is_available(backend: &str) -> Result<bool, MlxCError> {
    let operation: &'static str = "query MLX distributed availability";
    let backend_argument = backend_argument(backend, operation)?;
    // SAFETY: The backend name outlives the call.
    let available = unsafe { raw::mlx_distributed_is_available(backend_argument.as_ptr()) };
    Ok(available)
}

/// Initializes MLX's distributed backend.
///
/// # Errors
/// Returns the captured MLX-C description when initialization fails.
pub fn initialize_distributed(
    strict: bool,
    backend: &str,
) -> Result<MlxDistributedGroup, MlxCError> {
    let operation: &'static str = "initialize MLX distributed";
    let backend_argument = backend_argument(backend, operation)?;
    let mut output = MlxDistributedGroup::empty();
    // SAFETY: The backend name outlives the call and the output pointer is
    // valid writable storage.
    let status =
        unsafe { raw::mlx_distributed_init(output.raw_mut(), strict, backend_argument.as_ptr()) };
    check_status(status, operation)?;
    output.require_populated(operation)?;
    Ok(output)
}

/// Owned MLX distributed group handle released exactly once.
#[derive(Debug)]
pub struct MlxDistributedGroup(raw::mlx_distributed_group);

impl MlxDistributedGroup {
    /// Creates an empty group handle.
    pub(crate) fn empty() -> Self {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_group = unsafe { raw::mlx_distributed_group_new() };
        Self(raw_group)
    }

    /// This process's rank in the group.
    #[must_use]
    pub fn rank(&self) -> i32 {
        // SAFETY: `self` owns a live group handle.
        unsafe { raw::mlx_distributed_group_rank(self.0) }
    }

    /// The group's world size.
    #[must_use]
    pub fn size(&self) -> i32 {
        // SAFETY: `self` owns a live group handle.
        unsafe { raw::mlx_distributed_group_size(self.0) }
    }

    /// Splits the group by color and key.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the split fails.
    pub fn split(&self, color: i32, key: i32) -> Result<Self, MlxCError> {
        let operation: &'static str = "split an MLX distributed group";
        let mut output = Self::empty();
        // SAFETY: `self` owns a live group and the output pointer is valid
        // writable storage.
        let status =
            unsafe { raw::mlx_distributed_group_split(output.raw_mut(), self.0, color, key) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_distributed_group {
        &mut self.0
    }

    pub(crate) const fn raw(&self) -> raw::mlx_distributed_group {
        self.0
    }

    fn require_populated(&self, operation: &'static str) -> Result<(), MlxCError> {
        if self.0.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty distributed group handle".to_owned(),
            });
        }
        Ok(())
    }
}

impl Drop for MlxDistributedGroup {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live group exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_distributed_group_free(self.0);
        }
    }
}

impl MlxArray {
    /// Gathers every rank's copy of the array in the group.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the collective fails.
    pub fn all_gather(
        &self,
        group: &MlxDistributedGroup,
        stream: &MlxStream,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "run the MLX distributed all-gather";
        let mut output = Self::empty();
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_distributed_all_gather(output.raw_mut(), self.raw(), group.raw(), stream.raw())
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Reduces the array across the group with a maximum.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the collective fails.
    pub fn all_max(
        &self,
        group: &MlxDistributedGroup,
        stream: &MlxStream,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "run the MLX distributed all-max";
        let mut output = Self::empty();
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_distributed_all_max(output.raw_mut(), self.raw(), group.raw(), stream.raw())
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Reduces the array across the group with a minimum.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the collective fails.
    pub fn all_min(
        &self,
        group: &MlxDistributedGroup,
        stream: &MlxStream,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "run the MLX distributed all-min";
        let mut output = Self::empty();
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_distributed_all_min(output.raw_mut(), self.raw(), group.raw(), stream.raw())
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Reduces the array across the group with a sum.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the collective fails.
    pub fn all_sum(
        &self,
        group: &MlxDistributedGroup,
        stream: &MlxStream,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "run the MLX distributed all-sum";
        let mut output = Self::empty();
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_distributed_all_sum(output.raw_mut(), self.raw(), group.raw(), stream.raw())
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Scatters the array across the group with a sum.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the collective fails.
    pub fn sum_scatter(
        &self,
        group: &MlxDistributedGroup,
        stream: &MlxStream,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "run the MLX distributed sum-scatter";
        let mut output = Self::empty();
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_distributed_sum_scatter(
                output.raw_mut(),
                self.raw(),
                group.raw(),
                stream.raw(),
            )
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Sends this array to the destination rank.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the send fails.
    pub fn send_to_rank(
        &self,
        destination: i32,
        group: &MlxDistributedGroup,
        stream: &MlxStream,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "run the MLX distributed send";
        let mut output = Self::empty();
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_distributed_send(
                output.raw_mut(),
                self.raw(),
                destination,
                group.raw(),
                stream.raw(),
            )
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Receives an array of the given shape from the source rank.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the receive fails.
    pub fn receive_from_rank(
        shape: &[i32],
        dtype: MlxDtype,
        source: i32,
        group: &MlxDistributedGroup,
        stream: &MlxStream,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "run the MLX distributed receive";
        let mut output = Self::empty();
        // SAFETY: All handles are live, the shape slice remains valid for the
        // call, and the output pointer is valid writable storage.
        let status = unsafe {
            raw::mlx_distributed_recv(
                output.raw_mut(),
                shape.as_ptr(),
                shape.len(),
                dtype.to_raw(),
                source,
                group.raw(),
                stream.raw(),
            )
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Receives an array matching this array's shape from the source rank.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the receive fails.
    pub fn receive_like(
        &self,
        source: i32,
        group: &MlxDistributedGroup,
        stream: &MlxStream,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "run the MLX distributed receive-like";
        let mut output = Self::empty();
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_distributed_recv_like(
                output.raw_mut(),
                self.raw(),
                source,
                group.raw(),
                stream.raw(),
            )
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }
}
