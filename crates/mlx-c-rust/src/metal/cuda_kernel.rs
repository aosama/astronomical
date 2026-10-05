//! The MLX-C CUDA fast-kernel family.
//!
//! The CUDA backend's symbols exist in the linked native image but the
//! backend reports itself unavailable on Apple Silicon; the wrappers keep the
//! surface callable and typed so builds that target CUDA-capable hosts need
//! no bridge work.

use std::ffi::CString;

use crate::error::check_status;
use crate::raw;
use crate::{MlxArrayVector, MlxBindingsContext, MlxCError, MlxDtype, MlxStream};

/// Owned MLX-C CUDA kernel configuration handle.
#[derive(Debug)]
pub struct MlxCudaKernelConfig(raw::mlx_fast_cuda_kernel_config);

impl MlxCudaKernelConfig {
    /// Creates an empty CUDA kernel configuration.
    pub fn new() -> Result<Self, MlxCError> {
        const OPERATION: &str = "configure an MLX custom CUDA kernel";
        // SAFETY: MLX returns one owned config handle that enters RAII
        // ownership immediately.
        let raw_config = unsafe { raw::mlx_fast_cuda_kernel_config_new() };
        if raw_config.ctx.is_null() {
            return Err(MlxCError {
                operation: OPERATION,
                description: "MLX returned an empty custom CUDA kernel config".to_owned(),
            });
        }
        Ok(Self(raw_config))
    }

    /// Adds one typed output with a static shape.
    pub fn add_output(&mut self, shape: &[i32], dtype: MlxDtype) -> Result<(), MlxCError> {
        const OPERATION: &str = "configure an MLX custom CUDA kernel output";
        // SAFETY: The shape slice remains live for this copying call.
        let status = unsafe {
            raw::mlx_fast_cuda_kernel_config_add_output_arg(
                self.raw(),
                shape.as_ptr(),
                shape.len(),
                dtype.to_raw(),
            )
        };
        check_status(status, OPERATION)
    }

    /// Sets the launch grid dimensions.
    pub fn set_grid(&mut self, grid: [i32; 3]) -> Result<(), MlxCError> {
        const OPERATION: &str = "configure an MLX custom CUDA kernel grid";
        // SAFETY: The config handle is live and the values are plain.
        let status = unsafe {
            raw::mlx_fast_cuda_kernel_config_set_grid(self.raw(), grid[0], grid[1], grid[2])
        };
        check_status(status, OPERATION)
    }

    /// Sets the thread-group dimensions.
    pub fn set_thread_group(&mut self, thread_group: [i32; 3]) -> Result<(), MlxCError> {
        const OPERATION: &str = "configure an MLX custom CUDA kernel thread group";
        // SAFETY: The config handle is live and the values are plain.
        let status = unsafe {
            raw::mlx_fast_cuda_kernel_config_set_thread_group(
                self.raw(),
                thread_group[0],
                thread_group[1],
                thread_group[2],
            )
        };
        check_status(status, OPERATION)
    }

    /// Sets the scalar initialization value for output buffers.
    pub fn set_init_value(&mut self, initial_value: f32) -> Result<(), MlxCError> {
        const OPERATION: &str = "initialize an MLX custom CUDA kernel output";
        // SAFETY: The config handle is live and the value is copied by MLX.
        let status =
            unsafe { raw::mlx_fast_cuda_kernel_config_set_init_value(self.raw(), initial_value) };
        check_status(status, OPERATION)
    }

    /// Toggles MLX's verbose logging for this kernel launch.
    pub fn set_verbose(&mut self, verbose: bool) -> Result<(), MlxCError> {
        const OPERATION: &str = "configure MLX custom CUDA kernel logging";
        // SAFETY: The config handle is live and the flag is a plain value.
        let status = unsafe { raw::mlx_fast_cuda_kernel_config_set_verbose(self.raw(), verbose) };
        check_status(status, OPERATION)
    }

    /// Adds a dtype template argument.
    pub fn add_dtype_template_argument(
        &mut self,
        name: &str,
        dtype: MlxDtype,
    ) -> Result<(), MlxCError> {
        const OPERATION: &str = "add an MLX custom CUDA kernel dtype template argument";
        let name_argument = CString::new(name).map_err(|_| MlxCError {
            operation: OPERATION,
            description: "template argument name contains an interior null byte".to_owned(),
        })?;
        // SAFETY: The config handle is live and the name outlives the call.
        let status = unsafe {
            raw::mlx_fast_cuda_kernel_config_add_template_arg_dtype(
                self.raw(),
                name_argument.as_ptr(),
                dtype.to_raw(),
            )
        };
        check_status(status, OPERATION)
    }

    /// Adds an integer template argument.
    pub fn add_int_template_argument(
        &mut self,
        name: &str,
        argument: i32,
    ) -> Result<(), MlxCError> {
        const OPERATION: &str = "add an MLX custom CUDA kernel integer template argument";
        let name_argument = CString::new(name).map_err(|_| MlxCError {
            operation: OPERATION,
            description: "template argument name contains an interior null byte".to_owned(),
        })?;
        // SAFETY: The config handle is live and the name outlives the call.
        let status = unsafe {
            raw::mlx_fast_cuda_kernel_config_add_template_arg_int(
                self.raw(),
                name_argument.as_ptr(),
                argument,
            )
        };
        check_status(status, OPERATION)
    }

    /// Adds a boolean template argument.
    pub fn add_bool_template_argument(
        &mut self,
        name: &str,
        argument: bool,
    ) -> Result<(), MlxCError> {
        const OPERATION: &str = "add an MLX custom CUDA kernel boolean template argument";
        let name_argument = CString::new(name).map_err(|_| MlxCError {
            operation: OPERATION,
            description: "template argument name contains an interior null byte".to_owned(),
        })?;
        // SAFETY: The config handle is live and the name outlives the call.
        let status = unsafe {
            raw::mlx_fast_cuda_kernel_config_add_template_arg_bool(
                self.raw(),
                name_argument.as_ptr(),
                argument,
            )
        };
        check_status(status, OPERATION)
    }

    pub(crate) const fn raw(&self) -> raw::mlx_fast_cuda_kernel_config {
        self.0
    }
}

impl Drop for MlxCudaKernelConfig {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live config exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_fast_cuda_kernel_config_free(self.0);
        }
    }
}

/// Owned MLX-C CUDA kernel handle.
#[derive(Debug)]
pub struct MlxCudaKernel(raw::mlx_fast_cuda_kernel);

impl MlxCudaKernel {
    /// Compiles a CUDA kernel from source with named inputs and outputs.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when compilation fails.
    pub fn new(
        kernel_name: &str,
        input_names: &[&str],
        output_names: &[&str],
        kernel_header: &str,
        kernel_source: &str,
        ensure_row_contiguous: bool,
        shared_memory_bytes: i32,
    ) -> Result<Self, MlxCError> {
        const OPERATION: &str = "compile an MLX custom CUDA kernel";
        let kernel_name_argument = CString::new(kernel_name).map_err(|_| MlxCError {
            operation: OPERATION,
            description: "kernel name contains an interior null byte".to_owned(),
        })?;
        let input_names_vector = crate::types::vectors::MlxVectorString::from_strings(input_names)?;
        let output_names_vector =
            crate::types::vectors::MlxVectorString::from_strings(output_names)?;
        let header_argument = CString::new(kernel_header).map_err(|_| MlxCError {
            operation: OPERATION,
            description: "kernel header contains an interior null byte".to_owned(),
        })?;
        let source_argument = CString::new(kernel_source).map_err(|_| MlxCError {
            operation: OPERATION,
            description: "kernel source contains an interior null byte".to_owned(),
        })?;
        // SAFETY: Every C string and vector handle remains live for this
        // compiling constructor and the returned handle enters RAII
        // ownership.
        let raw_kernel = unsafe {
            raw::mlx_fast_cuda_kernel_new(
                kernel_name_argument.as_ptr(),
                input_names_vector.raw(),
                output_names_vector.raw(),
                source_argument.as_ptr(),
                header_argument.as_ptr(),
                ensure_row_contiguous,
                shared_memory_bytes,
            )
        };
        if raw_kernel.ctx.is_null() {
            return Err(MlxCError {
                operation: OPERATION,
                description: "MLX returned an empty custom CUDA kernel handle".to_owned(),
            });
        }
        Ok(Self(raw_kernel))
    }

    /// Applies the kernel to the inputs on the given stream.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the launch fails.
    pub fn apply(
        &self,
        context: &MlxBindingsContext,
        inputs: &MlxArrayVector,
        config: &MlxCudaKernelConfig,
    ) -> Result<MlxArrayVector, MlxCError> {
        const OPERATION: &str = "apply an MLX custom CUDA kernel";
        let mut outputs = MlxArrayVector::empty(OPERATION)?;
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_fast_cuda_kernel_apply(
                outputs.raw_mut(),
                self.0,
                inputs.raw(),
                config.raw(),
                context.gpu_stream().raw(),
            )
        };
        check_status(status, OPERATION)?;
        Ok(outputs)
    }
}

impl Drop for MlxCudaKernel {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live kernel exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_fast_cuda_kernel_free(self.0);
        }
    }
}

impl MlxStream {
    /// Applies the CUDA kernel with an explicit stream argument.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the launch fails.
    pub fn apply_cuda_kernel(
        &self,
        kernel: &MlxCudaKernel,
        inputs: &MlxArrayVector,
        config: &MlxCudaKernelConfig,
    ) -> Result<MlxArrayVector, MlxCError> {
        const OPERATION: &str = "apply an MLX custom CUDA kernel on a stream";
        let mut outputs = MlxArrayVector::empty(OPERATION)?;
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_fast_cuda_kernel_apply(
                outputs.raw_mut(),
                kernel.0,
                inputs.raw(),
                config.raw(),
                self.raw(),
            )
        };
        check_status(status, OPERATION)?;
        Ok(outputs)
    }
}
