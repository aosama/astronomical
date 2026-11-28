//! Beside-source coverage for the pure model-loading steps: the chunking
//! configuration validation. The load pipeline itself runs under the serial
//! engine lane against real artifacts.

use super::resolve_model_chunking_configuration;
use ipc_protocol::WorkerChunkingConfiguration;
use ipc_protocol::worker_chunking_test_support::worker_chunking_with_prompt_chunks;

fn worker_chunking_configuration_with(growth_tokens: u32) -> WorkerChunkingConfiguration {
    WorkerChunkingConfiguration {
        full_attention_key_value_growth_tokens: growth_tokens,
        prefill_graph_submission_layer_interval: 4,
        experimental_ssd_paging_generation_graph_submission_layer_interval: 0,
        ssd_prompt_cache_common_prefix_stride_blocks: 2,
        ..worker_chunking_with_prompt_chunks(Some(512), 512)
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
             invalid model input: full-attention key/value growth tokens must be positive"
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
             invalid model input: full-attention key/value growth tokens exceed the Int32 range"
        ),
        "the loading-time wrap must keep the range message: {loading_error}"
    );
}

fn resolved_draft_tokens(configured_draft_tokens: Option<u32>) -> Option<u32> {
    let mut worker_chunking = worker_chunking_configuration_with(64);
    worker_chunking.multi_token_prediction_draft_tokens = configured_draft_tokens;
    resolve_model_chunking_configuration(&worker_chunking)
        .expect("the chunking configuration validates")
        .multi_token_prediction_draft_tokens
}

#[test]
fn should_leave_the_count_to_the_head_when_the_configuration_leaves_it_open() {
    assert_eq!(resolved_draft_tokens(None), None);
}

#[test]
fn should_draft_nothing_when_the_configuration_says_zero() {
    assert_eq!(resolved_draft_tokens(Some(0)), Some(0));
}

#[test]
fn should_draft_the_configured_count_when_one_is_given() {
    assert_eq!(resolved_draft_tokens(Some(3)), Some(3));
}
