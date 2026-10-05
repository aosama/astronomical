//! Safe ownership boundary around the official MLX C API.
//!
//! This crate owns exactly one responsibility: the MLX-C C ABI translated
//! into idiomatic Rust. It holds the bindgen-generated raw declarations, the
//! captured-error machinery, the owned handle types (`MlxArray`, `MlxDtype`,
//! `MlxArrayVector`, `MlxStream`), the per-worker bindings context
//! (`MlxBindingsContext`: GPU stream and linked version), and every operation
//! wrapper family: creation, shape, padding, elementwise math, activation,
//! normalization, convolution, random, rope, quantized operations,
//! quantization construction, NVFP4, attention, the compiled-graph family,
//! the Metal kernels and capture, and the general operation set. It contains
//! no Astronomical runtime policy: memory limits, the metallib path
//! selection and verification, the safetensors vtable policies, and the
//! process I/O accounting stay in `astronomical-runtime-integration` and its
//! consumers, which link this crate against the pinned native image and
//! convert `MlxCError` values into their own typed runtime errors at their
//! boundaries.
//!
//! # API-surface completeness
//!
//! The bridge is complete, not selective: every public function and type the
//! pinned MLX-C headers declare is carried here. The coverage contract in
//! `astronomical-runtime-integration` proves that continuously by parsing
//! the provisioned pinned headers and comparing them against the compiled
//! bridge inventory — any unbridged symbol, stale bridge entry, or
//! unjustified exclusion fails the contract. The current state is recorded
//! in this crate's `COVERAGE_INVENTORY.md`; regenerate it after a deliberate
//! surface change with `scripts/generate-mlx-c-coverage-inventory.sh`.
//!
//! # Attribution
//!
//! The fallible-closure error channel — a payload trampoline parking its
//! typed failure in a thread-local slot that the caller's status check
//! prefers over MLX-C's own status error — adopts the pattern used by
//! mlx-rs (MIT OR Apache-2.0). The pattern is adopted; no mlx-rs code is
//! included.
//!
//! # Why this crate ships no standalone test binaries
//!
//! Every object file in this crate references MLX-C extern symbols. Those
//! symbols are satisfied by the static native image that
//! `astronomical-runtime-integration`'s build script links into final
//! binaries. A standalone test binary for this crate would therefore fail to
//! link without pulling in the whole native build, and wiring this crate as a
//! dev-dependency of `astronomical-runtime-integration` would create a
//! dependency cycle. Exercising tests for the behavior moved here stay in
//! `astronomical-runtime-integration`'s and `astronomical-model-serving`'s
//! test trees, which link the native image already.

mod mlx_activation_operations;
mod mlx_array;
mod mlx_array_accessors;
mod mlx_array_creation_operations;
mod mlx_array_shape_operations;
mod mlx_array_vector;
mod mlx_attention_operations;
mod mlx_bindings_context;
mod mlx_bindings_support;
mod mlx_closure_custom_types;
mod mlx_closure_types;
mod mlx_compile_operations;
mod mlx_compiled_attention_output_gate;
mod mlx_compiled_elementwise_graphs;
mod mlx_compiled_graph;
mod mlx_compiled_multi_output_graph;
mod mlx_compiled_sparse_shared_expert_combination;
mod mlx_compiled_swiglu;
mod mlx_compiled_verify_window_geometry;
mod mlx_compiled_verify_window_graph;
mod mlx_compiled_verify_window_ops;
mod mlx_compiled_vision_rope;
mod mlx_convolution_general_operations;
mod mlx_convolution_operations;
mod mlx_creation_operations;
mod mlx_cumulative_operations;
mod mlx_device;
mod mlx_elementwise_math_operations;
mod mlx_export_domain;
mod mlx_fast_cuda_kernel;
mod mlx_fft_operations;
mod mlx_indexing_operations;
mod mlx_io_domain;
mod mlx_linear_algebra_operations;
mod mlx_map_types;
mod mlx_math_binary_operations;
mod mlx_math_unary_operations;
mod mlx_memory_controls;
mod mlx_metal;
mod mlx_metal_capture;
mod mlx_metal_kernel;
mod mlx_normalization_operations;
mod mlx_nvfp4_operations;
mod mlx_operations;
mod mlx_padding_operations;
mod mlx_platform_domain;
mod mlx_quantization_construction;
mod mlx_quantized_array_operations;
mod mlx_quantized_operations;
mod mlx_random_operations;
mod mlx_random_sampling_operations;
mod mlx_reduction_operations;
mod mlx_rope_operations;
mod mlx_shape_operations;
mod mlx_stream;
mod mlx_stream_domain;
mod mlx_string;
mod mlx_vector_types;

pub mod error;
pub mod raw;

pub use error::{
    MlxCError, check_status, clear_captured_mlx_error, install_non_terminating_error_handler,
    take_captured_mlx_error,
};
pub use mlx_array::{MlxArray, MlxDtype};
pub use mlx_array_vector::MlxArrayVector;
pub use mlx_bindings_context::MlxBindingsContext;
pub use mlx_closure_custom_types::{MlxClosureCustom, MlxClosureCustomJvp, MlxClosureCustomVmap};
pub use mlx_closure_types::{MlxClosure, MlxClosureKwargs, MlxClosureValueAndGrad};
pub use mlx_compile_operations::{
    MlxCompileCache, disable_compile, enable_compile, jvp_of_closure, set_compile_mode,
    value_and_grad_of_closure, vjp_of_closure, vmap_replace, vmap_trace,
};
pub use mlx_compiled_elementwise_graphs::MlxCompiledElementwiseGraphs;
pub use mlx_compiled_graph::{
    MlxGraphBuilder, array_from_vector, graph_output_array, set_graph_output,
    set_graph_output_vector,
};
pub use mlx_compiled_multi_output_graph::MlxCompiledMultiOutputGraph;
pub use mlx_compiled_swiglu::MlxCompiledSwiGlu;
pub use mlx_compiled_verify_window_geometry::{
    VerifyWindowAffineSlot, VerifyWindowFeedForwardWeightSlot,
    VerifyWindowFullAttentionQuantization, VerifyWindowFullAttentionWeightSlot,
    VerifyWindowGatedDeltaQuantization, VerifyWindowGatedDeltaWeightSlot, VerifyWindowGeometry,
    VerifyWindowInputSlot, VerifyWindowLayerKind, VerifyWindowLayerQuantization,
    VerifyWindowLayerWeightSlot, VerifyWindowQuantizationPair, VerifyWindowTrunkQuantization,
    VerifyWindowTrunkWeightSlot, verify_window_input_slots,
};
pub use mlx_compiled_verify_window_graph::{
    MlxCompiledVerifyWindowGraph, VerifyWindowGdnKernelSet,
};
pub use mlx_device::{MlxDeviceHandle, MlxDeviceInfo};
pub use mlx_export_domain::{
    GraphOutputFile, MlxFunctionExporter, MlxImportedFunction, MlxNodeNamer, export_function,
    export_function_with_keywords, export_graph_to_dot, print_graph,
};
pub use mlx_fast_cuda_kernel::{MlxCudaKernel, MlxCudaKernelConfig};
pub use mlx_io_domain::{
    MlxIoGguf, MlxIoReader, MlxIoWriter, load_array_from_reader, load_safetensors,
    load_safetensors_from_reader, save_array_to_writer, save_safetensors,
    save_safetensors_to_writer,
};
pub use mlx_map_types::{
    MlxMapStringToArray, MlxMapStringToArrayIterator, MlxMapStringToString,
    MlxMapStringToStringIterator,
};
pub use mlx_memory_controls::{
    active_memory_bytes, cache_memory_bytes, clear_allocator_cache, memory_limit_bytes,
    peak_memory_bytes, reset_peak_memory, set_cache_limit, set_memory_limit, synchronize,
};
pub use mlx_metal::set_metallib_path;
pub use mlx_metal_kernel::{
    MlxMetalKernel, MlxMetalKernelConfig, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
    apply_metal_kernel_in_graph,
};
pub use mlx_platform_domain::{
    MlxDistributedGroup, cuda_is_available, distributed_is_available, initialize_distributed,
    metal_is_available, metallib_path, set_wired_limit,
};
pub use mlx_stream::MlxStream;
pub use mlx_stream_domain::{
    MlxStreamThreadLocal, clear_streams, set_default_stream, synchronize_all_defaults,
};
pub use mlx_string::MlxString;
pub use mlx_vector_types::{MlxVectorInt, MlxVectorStream, MlxVectorString, MlxVectorVectorArray};
