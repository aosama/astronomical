//! The shared Qwen3.5 model base type and its paging-free execution pieces.

#[cfg(feature = "direct-mlx")]
pub mod base;
#[cfg(feature = "direct-mlx")]
pub mod bound_weight_dtype_flow;
#[cfg(feature = "direct-mlx")]
pub mod decoder_layer_attention;
#[cfg(feature = "direct-mlx")]
pub mod differentiable_attention;
#[cfg(feature = "direct-mlx")]
pub mod disk_embedding_rows;
#[cfg(all(test, feature = "direct-mlx"))]
#[path = "model/disk_embedding_rows_tests.rs"]
mod disk_embedding_rows_tests;
#[cfg(feature = "direct-mlx")]
pub mod embedding_row_reads;
#[cfg(all(test, feature = "direct-mlx"))]
#[path = "model/embedding_row_reads_tests.rs"]
mod embedding_row_reads_tests;
#[cfg(feature = "direct-mlx")]
pub mod empty_state_forward;
#[cfg(feature = "direct-mlx")]
pub mod full_attention;
#[cfg(feature = "direct-mlx")]
pub mod gated_delta;
#[cfg(feature = "direct-mlx")]
pub mod layer_additions;
#[cfg(feature = "direct-mlx")]
pub mod loading_pipeline;
#[cfg(feature = "direct-mlx")]
pub mod memory_breakdown;
#[cfg(feature = "direct-mlx")]
pub mod model_chunking_configuration;
#[cfg(feature = "direct-mlx")]
pub mod output_gate_engine;
#[cfg(feature = "direct-mlx")]
pub mod prefill_forward;
#[cfg(feature = "direct-mlx")]
pub mod probe_token_ids;
#[cfg(feature = "direct-mlx")]
pub mod register_ordered_projections;
#[cfg(feature = "direct-mlx")]
pub mod routed_expert_selection;
#[cfg(feature = "direct-mlx")]
pub mod serving_model;
#[cfg(feature = "direct-mlx")]
pub mod target_forward_graph;
#[cfg(feature = "direct-mlx")]
pub mod tree_attention_measurement;
#[cfg(feature = "direct-mlx")]
pub mod tree_layer_attention;

#[cfg(feature = "direct-mlx")]
pub use base::Qwen3_5ModelBase;
#[cfg(feature = "direct-mlx")]
pub use decoder_layer_attention::Qwen3_5DecoderLayerAttentionOutput;
#[cfg(feature = "direct-mlx")]
pub use serving_model::Qwen3_5ServingModel;
