//! Core ML model documents written directly from Rust, each one a function from
//! weights to bytes, all in the ML Program format. Serving uses `projection` and
//! `mlp`; `rotated_mlp` belongs to the ternary research.

pub mod contraction_split;
mod mil_operations;
pub mod mlp;
pub mod mlp_stages;
pub mod model_interface;
pub mod package_layout;
pub mod projection;
pub mod rotated_mlp;
