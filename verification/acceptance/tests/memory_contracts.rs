//! The MLX memory contracts, one binary: allocator model-swap behavior at
//! the model-serving boundary (`direct-mlx`) and the native/MLX-C boundary
//! transitions (`mlx-memory-contract-probe`).

#[cfg(feature = "direct-mlx")]
#[path = "gpu_contracts/common/mod.rs"]
mod common;

#[cfg(feature = "direct-mlx")]
#[path = "memory_contracts/model_swap_allocator.rs"]
mod model_swap_allocator;

#[cfg(feature = "mlx-memory-contract-probe")]
#[path = "gpu_contracts/runtime_mlx/common/mod.rs"]
mod runtime_common;

#[cfg(feature = "mlx-memory-contract-probe")]
#[path = "memory_contracts/mlx_c_memory.rs"]
mod mlx_c_memory;

#[cfg(feature = "mlx-memory-contract-probe")]
#[path = "memory_contracts/native_probe.rs"]
// Measures the Metal allocator, which exists only on macOS.
#[cfg(target_os = "macos")]
mod native_probe;

#[cfg(feature = "mlx-memory-contract-probe")]
#[path = "memory_contracts/memory_fill_speed_probe.rs"]
// Measures the Metal allocator, which exists only on macOS.
#[cfg(target_os = "macos")]
mod memory_fill_speed_probe;
