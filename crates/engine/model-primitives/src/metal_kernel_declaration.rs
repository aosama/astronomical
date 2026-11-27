//! The one place a Metal kernel is declared to MLX (issue #2712).
//!
//! Every launcher in every crate hands its generated
//! [`MetalKernelSource`] here instead of calling `MlxMetalKernel::new`
//! itself, so the names MLX binds buffers by, their order and the source
//! text all come from the kernel's `.metal` file. `cargo xtask
//! check-metal-kernels` refuses an `MlxMetalKernel::new` anywhere else
//! outside the contracts that test MLX's own kernel mechanism.

use metal_kernel_declarations::{MetalKernelDefine, MetalKernelSource};
use mlx_bridge::MlxMetalKernel;
use runtime_integration::MlxRuntimeError;

/// Declares `kernel` to MLX with the launcher's run-time `defines`. MLX
/// compiles the source on the kernel's first evaluation, not here.
///
/// # Errors
/// Fails when the defines do not match the kernel's declared ones, or when
/// MLX refuses the declaration.
pub fn declare_metal_kernel(
    kernel: &MetalKernelSource,
    defines: &[MetalKernelDefine],
) -> Result<MlxMetalKernel, MlxRuntimeError> {
    let source_text =
        kernel
            .source_text(defines)
            .map_err(|define_error| MlxRuntimeError::RuntimeOperation {
                operation: "declare a Metal kernel",
                description: define_error.to_string(),
            })?;
    MlxMetalKernel::new_with_options(
        kernel.name,
        kernel.input_names,
        kernel.output_names,
        &kernel.header_text(),
        &source_text,
        kernel.has_atomic_outputs,
    )
    .map_err(MlxRuntimeError::from)
}

/// A float baked into a kernel's source, written as the launchers always
/// wrote it: nine digits after the point in scientific notation, which
/// round-trips every `f32`, with Metal's `f` suffix.
pub fn float_define(name: &'static str, value: f32) -> MetalKernelDefine {
    MetalKernelDefine {
        name,
        value: format!("{value:.9e}f"),
    }
}

#[cfg(test)]
#[path = "metal_kernel_declaration_tests.rs"]
mod metal_kernel_declaration_tests;
