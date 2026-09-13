//! Generated-variant support for `qwen4_exp` family tests.
//!
//! `shape_spec` declares a variant, `writer` materializes it into a temporary
//! artifact directory, and `variants` names the published packaging spread.
//! Test-support only: production must never reach this module.

pub mod shape_spec;
pub mod variants;
pub mod writer;

pub use shape_spec::{NgramStorageForm, VariantQuantization};
pub use variants::variant_matrix;
pub use writer::generate_variant;
