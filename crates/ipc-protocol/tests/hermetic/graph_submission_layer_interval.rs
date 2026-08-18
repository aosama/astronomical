use astronomical_ipc_protocol::{
    WorkerChunkingConfiguration, decode_graph_submission_layer_interval,
    graph_submission_layer_interval,
};

/// Builds prefill and paged-generation intervals used by the user journey.
fn graph_submission_chunking(
    prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
) -> WorkerChunkingConfiguration {
    WorkerChunkingConfiguration {
        fixed_prompt_processing_chunk_size_tokens: 2_048,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: None,
        full_attention_key_value_growth_tokens: 256,
        speculative_prefill_draft_forward_tokens: 2_048,
        prefill_graph_submission_layer_interval,
        experimental_ssd_paging_generation_graph_submission_layer_interval,
        prompt_cache_block_tokens: None,
        prompt_cache_common_prefix_stride_blocks: 4,
    }
}

#[test]
fn should_apply_prefill_submission_interval_when_experts_are_memory_resident() {
    let chunking = graph_submission_chunking(1, 3);

    assert_eq!(chunking.graph_submission_layer_interval(2_048, false), 1);
    assert_eq!(chunking.graph_submission_layer_interval(1, false), 0);
    assert_eq!(chunking.graph_submission_layer_interval(4, false), 0);
    assert_eq!(chunking.graph_submission_layer_interval(5, false), 1);
    assert_eq!(graph_submission_layer_interval(2_048, false, 1, 3), 1);
    assert_eq!(graph_submission_layer_interval(3, false, 1, 3), 0);
}

#[test]
fn should_apply_paged_generation_interval_only_while_experts_stream_from_storage() {
    let chunking = graph_submission_chunking(1, 3);

    assert_eq!(chunking.graph_submission_layer_interval(2_048, true), 1);
    assert_eq!(chunking.graph_submission_layer_interval(1, true), 3);
}

#[test]
fn should_keep_one_lazy_tape_when_graph_submission_intervals_are_zero() {
    let chunking = graph_submission_chunking(0, 0);

    assert_eq!(chunking.graph_submission_layer_interval(2_048, true), 0);
    assert_eq!(chunking.graph_submission_layer_interval(1, true), 0);
}

#[test]
fn should_keep_resident_decode_windows_on_one_lazy_tape() {
    let chunking = graph_submission_chunking(1, 3);

    assert_eq!(chunking.decode_graph_submission_layer_interval(false), 0);
    assert_eq!(decode_graph_submission_layer_interval(false, 3), 0);
}

#[test]
fn should_reuse_paged_generation_interval_for_decode_time_verification_windows() {
    let chunking = graph_submission_chunking(1, 3);

    assert_eq!(chunking.decode_graph_submission_layer_interval(true), 3);
    assert_eq!(decode_graph_submission_layer_interval(true, 3), 3);
}
