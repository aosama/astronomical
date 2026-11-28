//! A worker chunking configuration for unit tests: the product's defaults
//! around the two prompt chunk sizes a test cares about.

use crate::WorkerChunkingConfiguration;

/// The chunking a worker would receive with `in_memory_chunk_tokens` (`None`
/// is Automatic, not yet settled) and `ssd_streaming_chunk_tokens`.
#[must_use]
pub const fn worker_chunking_with_prompt_chunks(
    in_memory_chunk_tokens: Option<u32>,
    ssd_streaming_chunk_tokens: u32,
) -> WorkerChunkingConfiguration {
    WorkerChunkingConfiguration {
        prompt_processing_chunk_size_tokens: in_memory_chunk_tokens,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: ssd_streaming_chunk_tokens,
        full_attention_key_value_growth_tokens: 256,
        prefill_graph_submission_layer_interval: 0,
        experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
        ssd_prompt_cache_block_tokens: None,
        ssd_prompt_cache_common_prefix_stride_blocks: 4,
        experimental_in_memory_prefix_adoption_enabled: false,
        experimental_neural_engine_prefill_enabled: false,
        multi_token_prediction_draft_tokens: None,
    }
}
