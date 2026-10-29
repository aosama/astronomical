//! Beside-source coverage for the pure resident-loading units: the chunking
//! configuration validation. The load pipeline itself runs under the serial
//! engine lane against real artifacts.

use super::resolve_model_chunking_configuration;
use astronomical_ipc_protocol::WorkerChunkingConfiguration;

fn worker_chunking_configuration_with(growth_tokens: u32) -> WorkerChunkingConfiguration {
    WorkerChunkingConfiguration {
        fixed_prompt_processing_chunk_size_tokens: 512,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 512,
        full_attention_key_value_growth_tokens: growth_tokens,
        prefill_graph_submission_layer_interval: 4,
        experimental_ssd_paging_prefill_graph_submission_layer_interval: 4,
        experimental_ssd_paging_generation_graph_submission_layer_interval: 0,
        prompt_cache_block_tokens: None,
        prompt_cache_common_prefix_stride_blocks: 2,
        experimental_decode_stage_attribution_enabled: false,
        experimental_prefill_stage_attribution_enabled: false,
        experimental_in_memory_prefix_adoption_enabled: false,
        experimental_quantized_kv_cache_enabled: false,
        experimental_fused_moe_decode_enabled: false,
    }
}

#[test]
fn should_accept_a_positive_chunking_configuration() {
    let outcome = resolve_model_chunking_configuration(&worker_chunking_configuration_with(64));

    assert!(outcome.is_ok(), "a positive growth budget must validate");
}

#[test]
fn should_reject_a_zero_growth_token_configuration_with_the_loading_message() {
    let outcome = resolve_model_chunking_configuration(&worker_chunking_configuration_with(0));
    let loading_error = outcome.expect_err("zero growth tokens must fail validation");

    assert!(
        format!("{loading_error}").ends_with(
            "failed to validate model chunking configuration: \
             invalid Qwen3.5 model input: full-attention key/value growth tokens must be positive"
        ),
        "the loading-time wrap must keep the underlying validation message: {loading_error}"
    );
}

#[test]
fn should_reject_a_growth_token_count_beyond_the_int32_range() {
    let outcome =
        resolve_model_chunking_configuration(&worker_chunking_configuration_with(u32::MAX));
    let loading_error = outcome.expect_err("a u32 growth budget above Int32 must fail");

    assert!(
        format!("{loading_error}").ends_with(
            "failed to validate model chunking configuration: \
             invalid Qwen3.5 model input: full-attention key/value growth tokens exceed the Int32 range"
        ),
        "the loading-time wrap must keep the range message: {loading_error}"
    );
}
