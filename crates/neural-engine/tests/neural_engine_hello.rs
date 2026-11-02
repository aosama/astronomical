//! "Hello, Neural Engine": one real prefill projection on real hardware.
//!
//! The operation is the `qwen3_5` MoE mixer output projection (DeltaNet
//! `out_proj` / attention `o_proj`) for one 2,048-token prefill chunk:
//! 4,096 → 2,048 channels, 34.4 GFLOP. Each journey loads it through public
//! Core ML with a different set of allowed compute units and records three
//! pieces of evidence: where `MLComputePlan` placed it, how far its numbers
//! sit from the Accelerate fp32 reference, and how long one prediction takes.
//!
//! Run with `cargo test-neural-engine-hello` (optimized build, serial).

mod hello;

#[cfg(feature = "mlx-baseline")]
mod mlx_baseline;
