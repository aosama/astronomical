//! Serialized direct-MLX qualification for FLUX component boundaries.

#[path = "../../flux2_klein_hermetic/engine/direct_mlx.rs"]
mod engine;

#[allow(dead_code)]
#[path = "../../../src/flux2_klein/text_conditioning/error.rs"]
mod error;

#[path = "../../flux2_klein_hermetic/text_conditioning/direct_mlx_oracles.rs"]
mod text_conditioning;

#[path = "../../flux2_klein_hermetic/transformer/execution.rs"]
mod transformer;
