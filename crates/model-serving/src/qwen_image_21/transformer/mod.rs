//! Native MLX execution for the Qwen-Image-2.1 denoising transformer.
//!
//! Port of the diffusers `QwenImage21Transformer2DModel` (single-stream, block-causal,
//! `causal_condition` modulation, one shared modulation projection for all 32 blocks). Every
//! projection is the artifact's 4-bit affine quantization, loaded directly into MLX's quantized
//! matmul layout. The high-risk index logic — 3-axis RoPE, block-causal segmentation, target
//! masks, the timestep sinusoid — reuses the family's hermetically-tested pure modules; this
//! module owns the MLX binding and the block equations.

mod blocks;
mod execution;
mod preparation;
mod weights;

pub use execution::QwenImage21Transformer;
pub use preparation::QwenImage21TransformerRequest;
