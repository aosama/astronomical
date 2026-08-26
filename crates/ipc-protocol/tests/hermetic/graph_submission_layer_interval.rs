use astronomical_ipc_protocol::{WorkerChunkingConfiguration, graph_submission_layer_interval};

/// Builds resident and SSD-paged intervals used by the user journey.
fn graph_submission_chunking(
    prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
) -> WorkerChunkingConfiguration {
    WorkerChunkingConfiguration {
        fixed_prompt_processing_chunk_size_tokens: 2_048,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 2_048,
        full_attention_key_value_growth_tokens: 256,
        speculative_prefill_draft_forward_tokens: 2_048,
        prefill_graph_submission_layer_interval,
        experimental_ssd_paging_prefill_graph_submission_layer_interval,
        experimental_ssd_paging_generation_graph_submission_layer_interval,
        prompt_cache_block_tokens: None,
        prompt_cache_common_prefix_stride_blocks: 4,
    }
}

#[test]
fn should_use_resident_prefill_interval_and_keep_decode_on_one_tape() {
    let chunking = graph_submission_chunking(0, 1, 3);

    assert_eq!(chunking.graph_submission_layer_interval(2_048, false), 0);
    assert_eq!(chunking.graph_submission_layer_interval(1, false), 0);
    assert_eq!(graph_submission_layer_interval(2_048, false, 0, 1, 3), 0);
}

#[test]
fn should_honor_a_positive_resident_prefill_interval_without_changing_ssd_paging() {
    let chunking = graph_submission_chunking(2, 1, 3);

    assert_eq!(chunking.graph_submission_layer_interval(2_048, false), 2);
    assert_eq!(chunking.graph_submission_layer_interval(1, false), 0);
    assert_eq!(chunking.graph_submission_layer_interval(2_048, true), 1);
    assert_eq!(chunking.graph_submission_layer_interval(1, true), 3);
}

#[test]
fn should_apply_paged_prefill_and_generation_intervals_while_experts_stream_from_storage() {
    let chunking = graph_submission_chunking(0, 1, 3);

    assert_eq!(chunking.graph_submission_layer_interval(2_048, true), 1);
    assert_eq!(chunking.graph_submission_layer_interval(1, true), 3);
}

#[test]
fn should_keep_one_lazy_tape_when_ssd_paging_intervals_are_zero() {
    let chunking = graph_submission_chunking(0, 0, 0);

    assert_eq!(chunking.graph_submission_layer_interval(2_048, true), 0);
    assert_eq!(chunking.graph_submission_layer_interval(1, true), 0);
}
