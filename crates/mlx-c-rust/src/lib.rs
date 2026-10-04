//! Safe ownership boundary around the official MLX C API.
//!
//! This crate owns exactly one responsibility: the MLX-C C ABI translated
//! into idiomatic Rust. It holds the bindgen-generated raw declarations, the
//! captured-error machinery, and the owned handle types (`MlxArray`,
//! `MlxDtype`, `MlxArrayVector`, `MlxStream`) whose validity rules are
//! inseparable from the C contract they wrap. It contains no Astronomical
//! runtime policy: memory limits, the metallib path selection, and every
//! operation wrapper live in `astronomical-runtime-integration`, which links
//! this crate against the pinned native image and converts `MlxCError`
//! values into its own typed runtime errors at its boundary.
//!
//! # Why this crate ships no standalone test binaries
//!
//! Every object file in this crate references MLX-C extern symbols. Those
//! symbols are satisfied by the static native image that
//! `astronomical-runtime-integration`'s build script links into final
//! binaries. A standalone test binary for this crate would therefore fail to
//! link without pulling in the whole native build, and wiring this crate as a
//! dev-dependency of `astronomical-runtime-integration` would create a
//! dependency cycle. Exercising tests for the behavior moved here stay in
//! `astronomical-runtime-integration`'s and `astronomical-model-serving`'s
//! test trees, which link the native image already.

mod mlx_array;
mod mlx_array_vector;
mod mlx_stream;

pub mod error;
pub mod raw;

pub use error::{
    MlxCError, check_status, clear_captured_mlx_error, install_non_terminating_error_handler,
    take_captured_mlx_error,
};
pub use mlx_array::{MlxArray, MlxDtype};
pub use mlx_array_vector::MlxArrayVector;
pub use mlx_stream::MlxStream;
