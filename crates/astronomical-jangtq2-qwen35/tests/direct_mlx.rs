//! Graphics-processor contract and measurement tests for the jangtq2 Metal
//! kernels: the codebook decoder and Hadamard-32 activation rotation, the
//! fused gathered projection, and the tier-0 prefill gathered-dequant
//! prototype. Every test is `#[serial]` because MLX graphics-processor tests
//! never run in parallel.

#[path = "direct_mlx/mod.rs"]
mod direct_mlx_contracts;
