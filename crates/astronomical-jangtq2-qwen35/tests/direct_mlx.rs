//! Graphics-processor contract and measurement tests for the jangtq2 Metal
//! kernels: the codebook decoder and Hadamard-32 activation rotation, the
//! fused gathered projection, and the tier-0 prefill gathered-dequant
//! prototype. The correctness contracts run in parallel on small synthetic
//! tensors; the throughput measurements stay `#[serial]` per the
//! repository's performance-test rule.

#[path = "direct_mlx/mod.rs"]
mod direct_mlx_contracts;
