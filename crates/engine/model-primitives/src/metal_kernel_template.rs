//! The integer template arguments the custom Metal kernels take, shared by
//! every kernel family in this crate.

use mlx_bridge::MlxMetalKernelTemplateArgument;

/// The template integer `name` = `value` of a custom kernel launch.
pub fn integer(name: &'static str, value: i32) -> MlxMetalKernelTemplateArgument {
    MlxMetalKernelTemplateArgument::Integer {
        name,
        integer_template_argument: value,
    }
}
