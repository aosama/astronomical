//! Serialized direct-MLX acceptance for FLUX component boundaries.

#[path = "../../../../../../crates/model-serving/tests/families/flux2_klein/engine/direct_mlx.rs"]
mod engine;

#[allow(dead_code)]
#[path = "../../../../../../crates/model-serving/src/flux2_klein/text_conditioning/error.rs"]
mod error;

// The text-conditioning direct-MLX oracles moved beside-source in the model-serving
// test-layout campaign; they run in its direct-mlx lib lane now.

#[path = "../../../../../../crates/model-serving/tests/families/flux2_klein/transformer/execution.rs"]
mod transformer;
