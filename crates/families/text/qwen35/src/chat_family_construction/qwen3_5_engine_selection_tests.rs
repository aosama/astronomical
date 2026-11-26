use ipc_protocol::WorkerChunkingConfiguration;

use crate::EnginePlacement;

use super::{StreamingStart, prompt_chunk_tokens_for, streaming_engine_choice};

/// A model that fits whole keeps its placement; only a generation that
/// always streamed keeps the streaming chunk size.
#[test]
fn should_report_a_model_that_fits_whole_as_fitting_whatever_its_chunk_size() {
    for (keeps_the_streaming_chunk_size, holds_whole_model) in [(false, true), (true, false)] {
        let (selected_engine, engine_placement) = streaming_engine_choice(
            EnginePlacement::CompleteResidencyFits,
            keeps_the_streaming_chunk_size,
        );

        assert_eq!(selected_engine, StreamingStart { holds_whole_model });
        assert_eq!(engine_placement, EnginePlacement::CompleteResidencyFits);
    }
}

#[test]
fn should_keep_a_models_shortfall_when_it_does_not_fit() {
    let short_by_a_gigabyte = EnginePlacement::CompleteResidencyDoesNotFit {
        shortfall_bytes: 1_073_741_824,
    };
    for keeps_the_streaming_chunk_size in [false, true] {
        assert_eq!(
            streaming_engine_choice(short_by_a_gigabyte, keeps_the_streaming_chunk_size),
            (
                StreamingStart {
                    holds_whole_model: false
                },
                short_by_a_gigabyte
            )
        );
    }
}

#[test]
fn should_read_prompts_in_the_in_memory_chunk_only_when_the_whole_model_is_held() {
    // The two chunk sizes come first because they are what is tested; the
    // rest are the worker's defaults.
    let chunking = WorkerChunkingConfiguration {
        fixed_prompt_processing_chunk_size_tokens: 2_048,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 4_096,
        multi_token_prediction_draft_tokens: None,
        experimental_neural_engine_prefill_enabled: false,
        experimental_in_memory_prefix_adoption_enabled: false,
        ssd_prompt_cache_common_prefix_stride_blocks: 4,
        ssd_prompt_cache_block_tokens: None,
        experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
        prefill_graph_submission_layer_interval: 0,
        full_attention_key_value_growth_tokens: 256,
    };

    assert_eq!(prompt_chunk_tokens_for(&chunking, true), 2_048);
    assert_eq!(prompt_chunk_tokens_for(&chunking, false), 4_096);
}
