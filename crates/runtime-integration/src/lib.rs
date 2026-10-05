//! Narrow unsafe ownership boundary around the official MLX C API.

#[cfg(feature = "experimental-aligned-expert-packs")]
mod experimental;
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
mod mlx_bounded_safetensors_read_concurrency;
#[cfg(feature = "mlx")]
mod mlx_bounded_safetensors_reader;
#[cfg(feature = "mlx")]
mod mlx_descriptor_file_reader;
mod mlx_metallib_path;
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
mod mlx_runtime;
#[cfg(feature = "mlx")]
mod mlx_runtime_device_info;
mod mlx_runtime_types;
#[cfg(feature = "mlx")]
mod mlx_safetensors;
#[cfg(feature = "mlx")]
mod mlx_safetensors_memory_writer;
#[cfg(feature = "mlx")]
mod mlx_safetensors_writer;
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
mod positional_file_read_metrics;

#[cfg(feature = "experimental-aligned-expert-packs")]
pub use experimental::{
    MlxMetalExpertPackLoad, MlxMetalExpertPackLoadMetrics,
    MlxMetalExpertPackLoadMetricsAccumulator, MlxMetalExpertPackLoadMetricsSnapshot,
    MlxMetalExpertPackLoadRange, MlxMetalExpertPackOutputTensor,
};
pub use mlx_metallib_path::resolve_mlx_metallib_path;
#[cfg(feature = "mlx")]
pub use mlx_runtime::{
    MlxRuntime, classify_mlx_error, compiled_metallib_path, validate_metallib_path,
};
#[cfg(feature = "mlx")]
pub use mlx_runtime_device_info::maximum_recommended_gpu_working_set_size_bytes;
pub use mlx_runtime_types::{
    ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES, MlxMemoryLimits, MlxMemorySnapshot, MlxRuntimeError,
    allocator_cache_exceeds_reclaim_threshold,
};
#[cfg(feature = "mlx")]
pub use mlx_safetensors::{BoundedReadInterval, MlxSafetensors, SafetensorsLoadResult};
#[cfg(feature = "mlx")]
pub use mlx_safetensors_writer::{MlxSafetensorsWriteOutcome, MlxSafetensorsWriterError};
#[cfg(feature = "mlx")]
#[cfg(feature = "mlx")]
pub use positional_file_read_metrics::{
    PositionalFileReadMetrics, PositionalFileReadMetricsSnapshot,
};
