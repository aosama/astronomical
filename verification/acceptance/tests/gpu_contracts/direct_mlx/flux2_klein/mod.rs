//! Serialized direct-MLX acceptance for FLUX component boundaries.

mod engine;

#[allow(dead_code)]
#[path = "../../../../../../crates/model-serving/src/flux2_klein/text_conditioning/error.rs"]
mod error;

// The text-conditioning direct-MLX oracles moved beside-source in the model-serving
// test-layout campaign; they run in its direct-mlx lib lane now.

mod transformer;
