#[cfg(feature = "direct-mlx")]
mod astro_expert_cache_plus_streamed_page_route;
#[cfg(feature = "direct-mlx")]
mod deferred_expert_demand;
#[cfg(feature = "direct-mlx")]
mod dense_layer_counts;
#[cfg(feature = "direct-mlx")]
mod dense_layer_measurement;
#[cfg(feature = "direct-mlx")]
pub mod dense_layer_mlx_working_buffer_probe;
#[cfg(feature = "direct-mlx")]
mod dense_layer_payload;
#[cfg(feature = "direct-mlx")]
pub mod dense_layer_reads;
#[cfg(all(test, feature = "direct-mlx"))]
mod dense_layer_reads_tests;
#[cfg(feature = "direct-mlx")]
mod dense_layer_residency;
#[cfg(feature = "direct-mlx")]
pub mod dense_layer_streamer;
#[cfg(feature = "direct-mlx")]
mod diagnostic_paging;
#[cfg(feature = "direct-mlx")]
pub mod drafted_rows_verification;
#[cfg(feature = "direct-mlx")]
pub mod expert_layer_mlx_working_buffer_probe;
#[cfg(feature = "direct-mlx")]
mod expert_memory_mode;
#[cfg(feature = "direct-mlx")]
pub mod expert_reuse;
#[cfg(feature = "direct-mlx")]
mod expert_reuse_union_offer;
#[cfg(feature = "direct-mlx")]
mod expert_storage_streaming;
mod expert_streaming_stats;
#[cfg(feature = "direct-mlx")]
mod forward_layer_weights;
mod lane_mixing_decoder_layer;
#[cfg(feature = "direct-mlx")]
mod layer_major_checkpoint_parity;
mod layer_major_image_parity;
#[cfg(feature = "direct-mlx")]
pub mod layer_major_prefill;
#[cfg(feature = "direct-mlx")]
mod layer_major_prefill_run;
mod layer_major_span_embedding;
mod layer_major_span_redo;
mod layer_major_span_state;
#[cfg(feature = "direct-mlx")]
pub mod layer_probe_shape;
#[cfg(feature = "direct-mlx")]
pub mod layer_submission;
pub mod memory_governance;
mod mlx_buffer_cache_bound;
pub mod mtp_drafting;
pub mod mtp_head;
// Holds one streamed layer for every span of a layer-at-a-time prompt.
pub mod held_prompt_expert_layer;
// Reads the next streamed layer while the graphics processor computes.
pub mod prefill_complete_layer_prefetch;
#[cfg(test)]
mod prefill_complete_layer_prefetch_tests;
mod prefill_partial_layer_read;
#[cfg(test)]
mod prefill_partial_layer_read_tests;
#[cfg(feature = "direct-mlx")]
mod prefill_partial_page;
#[cfg(feature = "direct-mlx")]
mod request_expert_storage_read_tally;
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::model_math::{feed_forward_weights, output_combination, routing};
pub mod direct_expert_fetch;
pub mod direct_expert_fetch_tally;
#[cfg(feature = "direct-mlx")]
mod forward;
mod miss_route_dispatch;
#[cfg(feature = "direct-mlx")]
mod mixed_decode_execution;
pub mod reply_stage_tally;
// The one module of in-graph serving with unsafe code: its host step
// reaches the model from MLX's thread. The module says why that is sound.
#[cfg(feature = "direct-mlx")]
#[allow(unsafe_code)]
mod in_graph_route_lookup;
#[cfg(feature = "direct-mlx")]
mod in_graph_serving_record;
#[cfg(feature = "direct-mlx")]
pub mod in_graph_table_serving;
#[cfg(feature = "direct-mlx")]
mod optimistic_table_serving;
mod paged_execution;
#[cfg(feature = "direct-mlx")]
mod paged_execution_mode;
#[cfg(feature = "direct-mlx")]
mod paged_route_resolution;
#[cfg(feature = "direct-mlx")]
pub(super) mod phase_aware_expert_residency;
#[cfg(feature = "direct-mlx")]
mod read_through_residency;
#[cfg(feature = "direct-mlx")]
mod resident_layer_accounting;
#[cfg(feature = "direct-mlx")]
mod route_id_materialization;
#[cfg(feature = "direct-mlx")]
pub mod route_observation;
#[cfg(feature = "direct-mlx")]
pub mod routed_reply_recording;
#[cfg(feature = "direct-mlx")]
mod seat_planned_complete_layers;
pub mod streaming_model;
mod system_wired_governance;

pub use crate::qwen3_5_core::model::Qwen3_5ModelBase;
pub use crate::qwen3_5_core::model_math::{forward_contract, weights};
pub use crate::qwen3_5_streaming::model::streaming_model::Qwen3_5StreamingModel as Qwen3_5Model;
#[cfg(feature = "direct-mlx")]
pub use layer_probe_shape::LayerProbeShape;
pub use text_generation::execution_error as error;

#[cfg(all(test, feature = "direct-mlx"))]
mod paged_execution_tests;

#[cfg(feature = "direct-mlx")]
mod artifact_loading;
#[cfg(feature = "direct-mlx")]
mod decoder_layer_forward;
#[cfg(feature = "direct-mlx")]
mod evaluation;
#[cfg(feature = "direct-mlx")]
mod forward_attribution;
#[cfg(feature = "direct-mlx")]
mod forward_attribution_generation;
#[cfg(feature = "direct-mlx")]
mod forward_graph;
#[cfg(feature = "direct-mlx")]
mod governed_leases;
#[cfg(feature = "direct-mlx")]
mod live_memory_limit;
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::model::target_forward_graph::Qwen3_5TargetForwardOutput;
#[cfg(feature = "direct-mlx")]
pub use error::TextExecutionError;
#[cfg(feature = "direct-mlx")]
pub use text_generation::model::model_chunking_configuration::TextModelChunkingConfiguration;
#[cfg(feature = "direct-mlx")]
pub use weights::Qwen3_5Weights;

#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::artifacts::{Qwen3_5ShardIndex, ValidatedQwen3_5Artifact};
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::configuration::Qwen3_5FeedForwardArchitecture;
#[cfg(feature = "direct-mlx")]
pub use crate::qwen3_5_core::vision::{Qwen3_5VisionModel, visual_embedding_injection};
#[cfg(feature = "direct-mlx")]
pub use text_generation::RequestDecoderStateStack;

#[cfg(feature = "direct-mlx")]
pub use astro_expert_cache_plus_streamed_page_route::Qwen3_5MoEAstroExpertCachePlusStreamedPageRoute;
pub use mixed_decode_execution::qwen3_5_moe_combine_partial_route_outputs_for_tests;
#[cfg(feature = "direct-mlx")]
pub use output_combination::qwen3_5_moe_combine_experts;
#[cfg(feature = "direct-mlx")]
pub use paged_execution_mode::Qwen3_5MoEPagedExecutionMode;
#[cfg(feature = "direct-mlx")]
pub use paged_route_resolution::{PagedForwardMissingRouteCollector, PagedRouteValidationOutcome};
#[cfg(feature = "direct-mlx")]
pub use routing::{
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel, qwen3_5_moe_unsorted_expert_weighted_sum,
};
