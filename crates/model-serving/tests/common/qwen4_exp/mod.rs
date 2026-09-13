//! Generated-variant support for `qwen4_exp` family tests.
//!
//! `shape_spec` declares a variant, `writer` materializes it into a temporary
//! artifact directory, and `variants` names the published packaging spread.
//! Test-support only: production must never reach this module.

#[allow(dead_code)]
pub mod shape_spec;
#[allow(dead_code)]
pub mod variants;
#[allow(dead_code)]
pub mod writer;

#[allow(unused_imports)]
pub use shape_spec::{NgramStorageForm, VariantQuantization};
#[allow(unused_imports)]
pub use variants::variant_matrix;
#[allow(unused_imports)]
pub use writer::generate_variant;
