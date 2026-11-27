//! Astronomical's own Metal kernels as first-class files (issue #2712).
//!
//! Every kernel Astronomical hands MLX lives in a `.metal` file under a
//! `metal/` folder of the crate that launches it, named
//! `astro_<scope>_[<owner>_]<operation>[_<variant>]`, and opens with a
//! signature block: the inputs, outputs, template parameters and run-time
//! defines MLX's wrapper is built from, and the template values to compile
//! it with offline. This crate is the one reader of those files:
//!
//! - [`signature_block`] parses the block.
//! - [`kernel_naming`] states the naming convention and the scope a crate's
//!   location gives its kernels.
//! - [`kernel_files`] finds a crate's kernel files and resolves the headers
//!   and bodies a kernel names.
//! - [`generated_declarations`] writes, from a build script, the Rust
//!   declaration each launcher passes to MLX, so a launcher's input and
//!   output names can never drift from its kernel's file.
//! - [`kernel_instantiation`] assembles one `@compile-with` instantiation as
//!   a launcher hands it to MLX.
//! - [`mlx_wrapper`] writes the source MLX compiles around a body, line for
//!   line as MLX's `write_signature` does, for the offline compile.
//! - [`kernel_source`] is the declaration the generated code holds.

pub mod generated_declarations;
pub mod kernel_files;
pub mod kernel_instantiation;
pub mod kernel_naming;
pub mod kernel_source;
pub mod mlx_wrapper;
pub mod signature_block;

#[cfg(test)]
mod generated_declarations_tests;
#[cfg(test)]
mod kernel_files_tests;
#[cfg(test)]
mod kernel_instantiation_tests;
#[cfg(test)]
mod kernel_naming_tests;
#[cfg(test)]
mod kernel_source_tests;
#[cfg(test)]
mod mlx_wrapper_tests;
#[cfg(test)]
mod signature_block_tests;

pub use generated_declarations::{
    generate_kernel_declarations_for_build, run_kernel_declarations_build_script,
};
pub use kernel_source::{MetalKernelDefine, MetalKernelDefineError, MetalKernelSource};
