//! Native MLX execution for the Qwen3-VL text encoder (text-only conditioning path).
//!
//! Port of the Qwen3-VL language model as used by the Qwen-Image-2.1 pipeline: hidden states
//! per token, no logits, no vision tower, no KV cache — the prompt is encoded once per
//! generation. The heavy precision rules (FP32 norms and rope, BF16 activations) mirror the
//! family's shared MLX primitives.

mod blocks;
mod execution;
mod weights;

pub use execution::QwenImage21TextEncoder;
