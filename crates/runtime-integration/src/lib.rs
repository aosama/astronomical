//! Narrow unsafe ownership boundary around the official MLX C API.

#[cfg(feature = "experimental-aligned-expert-packs")]
mod experimental;
#[cfg(feature = "mlx")]
mod mlx_activation_operations;
#[cfg(feature = "mlx")]
mod mlx_array;
#[cfg(feature = "mlx")]
mod mlx_array_vector;
#[cfg(feature = "mlx")]
mod mlx_attention_operations;
#[cfg(feature = "mlx")]
mod mlx_bounded_safetensors_reader;
#[cfg(feature = "mlx")]
mod mlx_compiled_attention_output_gate;
#[cfg(feature = "mlx")]
mod mlx_compiled_elementwise_graphs;
#[cfg(feature = "mlx")]
mod mlx_compiled_graph;
#[cfg(feature = "mlx")]
mod mlx_compiled_sparse_shared_expert_combination;
#[cfg(feature = "mlx")]
mod mlx_compiled_swiglu;
#[cfg(feature = "mlx")]
mod mlx_convolution_operations;
#[cfg(feature = "mlx")]
mod mlx_creation_operations;
#[cfg(feature = "mlx")]
mod mlx_descriptor_file_reader;
#[cfg(feature = "mlx")]
mod mlx_metal_kernel;
#[cfg(feature = "mlx")]
mod mlx_operations;
#[cfg(feature = "mlx")]
mod mlx_quantized_operations;
#[cfg(feature = "mlx")]
mod mlx_random_operations;
#[cfg(feature = "mlx")]
mod mlx_rope_operations;
#[cfg(feature = "mlx")]
mod mlx_runtime;
#[cfg(feature = "mlx")]
mod mlx_runtime_device_info;
#[cfg(feature = "mlx")]
mod mlx_runtime_types;
#[cfg(feature = "mlx")]
mod mlx_safetensors;
#[cfg(feature = "mlx")]
mod mlx_safetensors_memory_writer;
#[cfg(feature = "mlx")]
mod mlx_safetensors_writer;
#[cfg(feature = "mlx")]
mod mlx_shape_operations;
#[cfg(feature = "mlx")]
mod mlx_stream;
#[cfg(feature = "mlx")]
mod positional_file_read_metrics;
#[cfg(feature = "mlx")]
mod raw;

#[cfg(feature = "experimental-aligned-expert-packs")]
pub use experimental::{
    MlxMetalExpertPackLoad, MlxMetalExpertPackLoadMetrics,
    MlxMetalExpertPackLoadMetricsAccumulator, MlxMetalExpertPackLoadMetricsSnapshot,
    MlxMetalExpertPackLoadRange, MlxMetalExpertPackOutputTensor,
};
#[cfg(feature = "mlx")]
pub use mlx_array::{MlxArray, MlxDtype};
#[cfg(feature = "mlx")]
pub use mlx_compiled_elementwise_graphs::MlxCompiledElementwiseGraphs;
#[cfg(feature = "mlx")]
pub use mlx_compiled_swiglu::MlxCompiledSwiGlu;
#[cfg(feature = "mlx")]
pub use mlx_metal_kernel::{MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument};
#[cfg(feature = "mlx")]
pub use mlx_runtime::{MlxRuntime, compiled_metallib_path, validate_metallib_path};
#[cfg(feature = "mlx")]
pub use mlx_runtime_device_info::maximum_recommended_gpu_working_set_size_bytes;
#[cfg(feature = "mlx")]
pub use mlx_runtime_types::{MlxMemoryLimits, MlxMemorySnapshot, MlxRuntimeError};
#[cfg(feature = "mlx")]
pub use mlx_safetensors::{BoundedReadInterval, MlxSafetensors, SafetensorsLoadResult};
#[cfg(feature = "mlx")]
pub use mlx_safetensors_writer::{MlxSafetensorsWriteOutcome, MlxSafetensorsWriterError};
#[cfg(feature = "mlx")]
pub use positional_file_read_metrics::{
    PositionalFileReadMetrics, PositionalFileReadMetricsSnapshot,
};
