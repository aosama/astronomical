//! Core ML model documents written directly from Rust, each one a function from
//! weights to bytes. Serving uses `projection` and `mlp`; `rotated_mlp` belongs to
//! the ternary research and `neural_network` to the earliest placement evidence.

mod mil_operations;
pub mod mlp;
pub mod mlp_stages;
pub mod model_interface;
pub mod neural_network;
pub mod package_layout;
pub mod projection;
pub mod rotated_mlp;
