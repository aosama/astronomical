//! Serialized direct-MLX acceptance for FLUX component boundaries.

#[path = "../../../../../../crates/model-serving/tests/flux2_klein_offline/engine/direct_mlx.rs"]
mod engine;

#[allow(dead_code)]
#[path = "../../../../../../crates/model-serving/src/flux2_klein/text_conditioning/error.rs"]
mod error;

#[path = "../../../../../../crates/model-serving/tests/flux2_klein_offline/text_conditioning/direct_mlx_oracles.rs"]
mod text_conditioning;

#[path = "../../../../../../crates/model-serving/tests/flux2_klein_offline/transformer/execution.rs"]
mod transformer;
