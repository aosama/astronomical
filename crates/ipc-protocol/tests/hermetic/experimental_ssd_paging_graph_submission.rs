use astronomical_ipc_protocol::{
    WorkerChunkingConfiguration, WorkerPromptProcessingChunkSizingPolicy,
    experimental_ssd_paging_graph_submission_layer_interval,
};

/// Builds the experimental solid-state-drive paging intervals used by the user journey.
fn experimental_ssd_paging_chunking(
    experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
) -> WorkerChunkingConfiguration {
    WorkerChunkingConfiguration {
        prompt_processing_chunk_sizing_policy:
            WorkerPromptProcessingChunkSizingPolicy::Optimized {
            prompt_processing_chunk_size_optimizer_candidate_token_counts: vec![1_024, 2_048],
        },
        full_attention_key_value_growth_tokens: 256,
        speculative_prefill_draft_forward_tokens: 2_048,
        experimental_ssd_paging_prefill_graph_submission_layer_interval,
        experimental_ssd_paging_generation_graph_submission_layer_interval,
        prompt_processing_chunk_size_optimizer_maximum_retained_measurements_per_candidate_and_context: 5,
        prompt_processing_chunk_size_optimizer_position_range_size_tokens: 32_768,
        prompt_cache_block_tokens: None,
        prompt_cache_common_prefix_stride_blocks: 4,
    }
}

#[test]
fn should_ignore_experimental_ssd_paging_intervals_when_experts_are_memory_resident() {
    let chunking = experimental_ssd_paging_chunking(1, 3);

    assert_eq!(
        chunking.experimental_ssd_paging_graph_submission_layer_interval(2_048, false),
        0
    );
    assert_eq!(
        chunking.experimental_ssd_paging_graph_submission_layer_interval(1, false),
        0
    );
    assert_eq!(
        experimental_ssd_paging_graph_submission_layer_interval(2_048, false, 1, 3),
        0
    );
}

#[test]
fn should_apply_experimental_ssd_paging_intervals_only_while_experts_stream_from_storage() {
    let chunking = experimental_ssd_paging_chunking(1, 3);

    assert_eq!(
        chunking.experimental_ssd_paging_graph_submission_layer_interval(2_048, true),
        1
    );
    assert_eq!(
        chunking.experimental_ssd_paging_graph_submission_layer_interval(1, true),
        3
    );
}

#[test]
fn should_keep_one_lazy_tape_when_experimental_ssd_paging_intervals_are_zero() {
    let chunking = experimental_ssd_paging_chunking(0, 0);

    assert_eq!(
        chunking.experimental_ssd_paging_graph_submission_layer_interval(2_048, true),
        0
    );
    assert_eq!(
        chunking.experimental_ssd_paging_graph_submission_layer_interval(1, true),
        0
    );
}
