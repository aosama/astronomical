//! GPU contract tests for the MLX encoder and the shard-wise requantizer.
//! Every test is `#[serial]`: MLX tests never run in parallel.

#[path = "direct_mlx/mod.rs"]
mod direct_mlx_contracts;
