use crate::WorkerChunkingConfiguration;

/// Builds the in-memory prefill interval and the streamed generation interval.
fn graph_submission_chunking(
    prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
) -> WorkerChunkingConfiguration {
    WorkerChunkingConfiguration {
        prompt_processing_chunk_size_tokens: Some(2_048),
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 2_048,
        full_attention_key_value_growth_tokens: 256,
        prefill_graph_submission_layer_interval,
        experimental_ssd_paging_generation_graph_submission_layer_interval,
        ssd_prompt_cache_block_tokens: None,
        ssd_prompt_cache_common_prefix_stride_blocks: 4,
        experimental_in_memory_prefix_adoption_enabled: false,
        experimental_neural_engine_prefill_enabled: false,
        multi_token_prediction_draft_tokens: None,
    }
}

#[test]
fn should_use_resident_prefill_interval_and_keep_decode_on_one_tape() {
    let chunking = graph_submission_chunking(0, 3);

    assert_eq!(chunking.graph_submission_layer_interval(2_048, false), 0);
    assert_eq!(chunking.graph_submission_layer_interval(1, false), 0);
    assert_eq!(
        crate::graph_submission_layer_interval(2_048, false, 0, 3),
        0
    );
}

#[test]
fn should_honor_a_positive_resident_prefill_interval_without_changing_ssd_paging() {
    let chunking = graph_submission_chunking(2, 3);

    assert_eq!(chunking.graph_submission_layer_interval(2_048, false), 2);
    assert_eq!(chunking.graph_submission_layer_interval(1, false), 0);
    assert_eq!(chunking.graph_submission_layer_interval(1, true), 3);
}

#[test]
fn should_apply_no_interval_to_a_streamed_multi_token_forward_whatever_is_configured() {
    let chunking = graph_submission_chunking(2, 3);

    assert_eq!(chunking.graph_submission_layer_interval(2_048, true), 0);
    assert_eq!(chunking.graph_submission_layer_interval(2, true), 0);
}

#[test]
fn should_keep_one_lazy_tape_when_the_streamed_generation_interval_is_zero() {
    let chunking = graph_submission_chunking(0, 0);

    assert_eq!(chunking.graph_submission_layer_interval(1, true), 0);
}
