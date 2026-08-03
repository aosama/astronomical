use std::ffi::{CString, NulError};

use crate::{
    MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, mlx_array_vector::MlxArrayVector,
    mlx_runtime::check_status, raw,
};

/// Output shape and dtype requested from one custom Metal kernel launch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxMetalKernelOutput {
    shape: Vec<i32>,
    dtype: MlxDtype,
}

impl MlxMetalKernelOutput {
    #[must_use]
    pub const fn new(shape: Vec<i32>, dtype: MlxDtype) -> Self {
        Self { shape, dtype }
    }
}

/// Compile-time template argument passed to an MLX custom Metal kernel launch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MlxMetalKernelTemplateArgument {
    Dtype {
        name: &'static str,
        dtype: MlxDtype,
    },
    Integer {
        name: &'static str,
        integer_template_argument: i32,
    },
    Boolean {
        name: &'static str,
        boolean_template_argument: bool,
    },
}

/// Owned MLX custom Metal kernel function compiled from a source string.
#[derive(Debug)]
pub struct MlxMetalKernel {
    raw_kernel: raw::mlx_fast_metal_kernel,
}

impl MlxMetalKernel {
    pub fn new(
        kernel_name: &str,
        input_names: &[&str],
        output_names: &[&str],
        kernel_source: &str,
    ) -> Result<Self, MlxRuntimeError> {
        const OPERATION: &str = "create an MLX custom Metal kernel";
        if input_names.is_empty() || output_names.is_empty() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: OPERATION,
                description: "custom Metal kernels must declare at least one input and output"
                    .to_owned(),
            });
        }
        let kernel_name = c_string(kernel_name, OPERATION, "kernel name")?;
        let kernel_source = c_string(kernel_source, OPERATION, "kernel source")?;
        let empty_header = c_string("", OPERATION, "kernel header")?;
        let input_name_vector = MlxStringVector::new(input_names, OPERATION)?;
        let output_name_vector = MlxStringVector::new(output_names, OPERATION)?;
        // SAFETY: C strings and vectors stay live through this copying constructor;
        // Row-contiguous inputs and non-atomic outputs match this kernel contract.
        let raw_kernel = unsafe {
            raw::mlx_fast_metal_kernel_new(
                kernel_name.as_ptr(),
                input_name_vector.raw(),
                output_name_vector.raw(),
                kernel_source.as_ptr(),
                empty_header.as_ptr(),
                true,
                false,
            )
        };
        if raw_kernel.ctx.is_null() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: OPERATION,
                description: "MLX returned an empty custom Metal kernel handle".to_owned(),
            });
        }
        Ok(Self { raw_kernel })
    }

    const fn raw(&self) -> raw::mlx_fast_metal_kernel {
        self.raw_kernel
    }
}

impl Drop for MlxMetalKernel {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live kernel handle exactly once.
        unsafe {
            raw::mlx_fast_metal_kernel_free(self.raw_kernel);
        }
    }
}

impl MlxRuntime {
    /// Applies one owned custom Metal kernel on the runtime's GPU stream.
    pub fn apply_metal_kernel(
        &self,
        kernel: &MlxMetalKernel,
        input_arrays: &[&MlxArray],
        output_specs: &[MlxMetalKernelOutput],
        grid: [i32; 3],
        thread_group: [i32; 3],
        template_arguments: &[MlxMetalKernelTemplateArgument],
    ) -> Result<Vec<MlxArray>, MlxRuntimeError> {
        const OPERATION: &str = "apply an MLX custom Metal kernel";
        validate_kernel_launch(input_arrays, output_specs, grid, thread_group)?;
        let input_vector = MlxArrayVector::new(input_arrays)?;
        let kernel_config = MlxMetalKernelConfig::new(output_specs, grid, thread_group)?;
        kernel_config.add_template_arguments(template_arguments)?;
        let mut output_vector = MlxArrayVector::empty(OPERATION)?;
        // SAFETY: Kernel, input vector, config, stream, and output vector are live;
        // MLX populates the output vector synchronously as a lazy graph result.
        let status = unsafe {
            raw::mlx_fast_metal_kernel_apply(
                output_vector.raw_mut(),
                kernel.raw(),
                input_vector.raw(),
                kernel_config.raw(),
                self.gpu_stream().raw(),
            )
        };
        check_status(status, OPERATION)?;
        let output_count = output_vector.len();
        if output_count != output_specs.len() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: OPERATION,
                description: format!(
                    "custom Metal kernel returned {output_count} outputs but {} were requested",
                    output_specs.len()
                ),
            });
        }
        (0..output_count)
            .map(|output_index| output_vector.array_at(output_index, OPERATION))
            .collect()
    }
}

#[derive(Debug)]
struct MlxMetalKernelConfig {
    raw_config: raw::mlx_fast_metal_kernel_config,
}

impl MlxMetalKernelConfig {
    fn new(
        output_specs: &[MlxMetalKernelOutput],
        grid: [i32; 3],
        thread_group: [i32; 3],
    ) -> Result<Self, MlxRuntimeError> {
        const OPERATION: &str = "configure an MLX custom Metal kernel";
        // SAFETY: The runtime error handler is installed before model loading,
        // and MLX returns one owned config handle.
        let raw_config = unsafe { raw::mlx_fast_metal_kernel_config_new() };
        if raw_config.ctx.is_null() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: OPERATION,
                description: "MLX returned an empty custom Metal kernel config".to_owned(),
            });
        }
        let kernel_config = Self { raw_config };
        for output_spec in output_specs {
            // SAFETY: The shape slice remains live for this copying call.
            let status = unsafe {
                raw::mlx_fast_metal_kernel_config_add_output_arg(
                    kernel_config.raw(),
                    output_spec.shape.as_ptr(),
                    output_spec.shape.len(),
                    output_spec.dtype.to_raw(),
                )
            };
            check_status(status, OPERATION)?;
        }
        // SAFETY: Scalar launch dimensions are validated before construction.
        let status = unsafe {
            raw::mlx_fast_metal_kernel_config_set_grid(
                kernel_config.raw(),
                grid[0],
                grid[1],
                grid[2],
            )
        };
        check_status(status, OPERATION)?;
        // SAFETY: Scalar launch dimensions are validated before construction.
        let status = unsafe {
            raw::mlx_fast_metal_kernel_config_set_thread_group(
                kernel_config.raw(),
                thread_group[0],
                thread_group[1],
                thread_group[2],
            )
        };
        check_status(status, OPERATION)?;
        Ok(kernel_config)
    }

    const fn raw(&self) -> raw::mlx_fast_metal_kernel_config {
        self.raw_config
    }

    fn add_template_arguments(
        &self,
        template_arguments: &[MlxMetalKernelTemplateArgument],
    ) -> Result<(), MlxRuntimeError> {
        const OPERATION: &str = "configure an MLX custom Metal kernel";
        for template_argument in template_arguments {
            match *template_argument {
                MlxMetalKernelTemplateArgument::Dtype { name, dtype } => {
                    let argument_name = c_string(name, OPERATION, "template dtype name")?;
                    // SAFETY: Name is live for this copying call; dtype is validated by MLX.
                    let status = unsafe {
                        raw::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            self.raw(),
                            argument_name.as_ptr(),
                            dtype.to_raw(),
                        )
                    };
                    check_status(status, OPERATION)?;
                }
                MlxMetalKernelTemplateArgument::Integer {
                    name,
                    integer_template_argument,
                } => {
                    let argument_name = c_string(name, OPERATION, "template integer name")?;
                    // SAFETY: Name is live for this copying call; the scalar argument is copied.
                    let status = unsafe {
                        raw::mlx_fast_metal_kernel_config_add_template_arg_int(
                            self.raw(),
                            argument_name.as_ptr(),
                            integer_template_argument,
                        )
                    };
                    check_status(status, OPERATION)?;
                }
                MlxMetalKernelTemplateArgument::Boolean {
                    name,
                    boolean_template_argument,
                } => {
                    let argument_name = c_string(name, OPERATION, "template boolean name")?;
                    // SAFETY: Name is live for this copying call; the scalar argument is copied.
                    let status = unsafe {
                        raw::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            self.raw(),
                            argument_name.as_ptr(),
                            boolean_template_argument,
                        )
                    };
                    check_status(status, OPERATION)?;
                }
            }
        }
        Ok(())
    }
}

impl Drop for MlxMetalKernelConfig {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live config handle exactly once.
        unsafe {
            raw::mlx_fast_metal_kernel_config_free(self.raw_config);
        }
    }
}

#[derive(Debug)]
struct MlxStringVector {
    raw_vector: raw::mlx_vector_string,
}

impl MlxStringVector {
    fn new(strings: &[&str], operation: &'static str) -> Result<Self, MlxRuntimeError> {
        let c_strings = strings
            .iter()
            .map(|source_string| c_string(source_string, operation, "string vector entry"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut string_pointers = c_strings
            .iter()
            .map(|c_string| c_string.as_ptr())
            .collect::<Vec<_>>();
        // SAFETY: Pointers remain live for this copying constructor.
        let raw_vector = unsafe {
            raw::mlx_vector_string_new_data(string_pointers.as_mut_ptr(), string_pointers.len())
        };
        if raw_vector.ctx.is_null() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation,
                description: "MLX returned an empty string vector handle".to_owned(),
            });
        }
        Ok(Self { raw_vector })
    }

    const fn raw(&self) -> raw::mlx_vector_string {
        self.raw_vector
    }
}

impl Drop for MlxStringVector {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live string vector handle exactly once.
        unsafe {
            raw::mlx_vector_string_free(self.raw_vector);
        }
    }
}

fn validate_kernel_launch(
    input_arrays: &[&MlxArray],
    output_specs: &[MlxMetalKernelOutput],
    grid: [i32; 3],
    thread_group: [i32; 3],
) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "apply an MLX custom Metal kernel";
    if input_arrays.is_empty() || output_specs.is_empty() {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "custom Metal kernel launches require at least one input and output"
                .to_owned(),
        });
    }
    if grid.iter().any(|dimension| *dimension <= 0)
        || thread_group.iter().any(|dimension| *dimension <= 0)
    {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "custom Metal kernel grid and threadgroup dimensions must be positive"
                .to_owned(),
        });
    }
    Ok(())
}

fn c_string(
    source_text: &str,
    operation: &'static str,
    value_description: &'static str,
) -> Result<CString, MlxRuntimeError> {
    CString::new(source_text).map_err(|source| nul_error(operation, value_description, source))
}

fn nul_error(
    operation: &'static str,
    value_description: &'static str,
    source: NulError,
) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: format!("{value_description} contains an interior NUL byte: {source}"),
    }
}
