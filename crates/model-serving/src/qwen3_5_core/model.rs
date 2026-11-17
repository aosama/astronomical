//! The shared Qwen3.5 model base type and its paging-free execution pieces.

#[cfg(feature = "direct-mlx")]
pub(crate) mod base;
#[cfg(feature = "direct-mlx")]
pub(crate) mod bound_weight_dtype_flow;
#[cfg(feature = "direct-mlx")]
pub(crate) mod decoder_layer_attention;
#[cfg(feature = "direct-mlx")]
pub(crate) mod differentiable_attention;
#[cfg(feature = "direct-mlx")]
pub(crate) mod disk_embedding_rows;
#[cfg(all(test, feature = "direct-mlx"))]
#[path = "model/disk_embedding_rows_tests.rs"]
mod disk_embedding_rows_tests;
#[cfg(feature = "direct-mlx")]
pub(crate) mod embedding_row_reads;
#[cfg(all(test, feature = "direct-mlx"))]
#[path = "model/embedding_row_reads_tests.rs"]
mod embedding_row_reads_tests;
#[cfg(feature = "direct-mlx")]
pub(crate) mod empty_state_forward;
#[cfg(feature = "direct-mlx")]
pub(crate) mod full_attention;
#[cfg(feature = "direct-mlx")]
pub(crate) mod gated_delta;
#[cfg(feature = "direct-mlx")]
pub(crate) mod layer_additions;
#[cfg(feature = "direct-mlx")]
pub(crate) mod loading_pipeline;
#[cfg(feature = "direct-mlx")]
pub(crate) mod memory_breakdown;
#[cfg(feature = "direct-mlx")]
pub(crate) mod model_chunking_configuration;
#[cfg(feature = "direct-mlx")]
pub(crate) mod output_gate_engine;
#[cfg(feature = "direct-mlx")]
pub(crate) mod probe_token_ids;
#[cfg(feature = "direct-mlx")]
pub(crate) mod register_ordered_projections;
#[cfg(feature = "direct-mlx")]
pub(crate) mod tree_attention_measurement;
#[cfg(feature = "direct-mlx")]
pub(crate) mod tree_layer_attention;

#[cfg(feature = "direct-mlx")]
pub(crate) use base::Qwen3_5ModelBase;
#[cfg(feature = "direct-mlx")]
pub(crate) use decoder_layer_attention::Qwen3_5DecoderLayerAttentionOutput;
