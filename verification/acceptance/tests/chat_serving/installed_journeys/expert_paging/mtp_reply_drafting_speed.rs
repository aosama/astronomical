//! Reply speed with and without drafted tokens on a streamed model that
//! ships a multi-token-prediction head (issue #2533).
//!
//! Each case loads the sparse model under the configured MLX limit through
//! the real streaming engine, sets how many tokens a round drafts before
//! anything is served (as the app's setting would be known at load), serves
//! a 500/50 warmup and then one 10,000-token Romeo and Juliet summary. The
//! engine's own reply loop runs every token, so the figures are what the
//! app would serve. Compare the cases' tokens per second from their
//! `[performance-attribution] status=success phase=mtp_drafts_N` lines:
//! zero drafted tokens is the reply written one token a forward.
//!
//! Run with `cargo measure-mtp-reply-drafting --configuration PATH`, whose
//! `model_directory` names the model and whose
//! `paging_experiment.maximum_mlx_memory_gigabytes` sets the limit. About
//! seventy seconds a case.

use std::time::Duration;

use ipc_protocol::RequestId;
use model_serving::Qwen3_5StreamingEngineTestControls;
use serial_test::serial;
use test_harness::test_run::TestRunContext;
use tokio::time::timeout;

use crate::serving_acceptance::support::performance_attribution::{
    GenerationSampling, PagingExperimentEngine, RequestAttribution, load_paging_experiment_engine,
    run_generation_with_request_attribution,
};

const WARMUP_REQUEST_ID: RequestId = RequestId::new(98_501);
const MEASURED_REQUEST_ID: RequestId = RequestId::new(98_502);
const WARMUP_PROMPT_TOKEN_COUNT: usize = 500;
const WARMUP_OUTPUT_TOKEN_COUNT: u16 = 50;
const MEASURED_PROMPT_TOKEN_COUNT: usize = 10_000;
const MEASURED_OUTPUT_TOKEN_COUNT: u16 = 500;
const CASE_TIMEOUT: Duration = Duration::from_secs(150);

#[tokio::test]
#[ignore = "measures essay reply speed one token a forward on an installed streamed model"]
#[serial]
async fn should_measure_essay_reply_speed_without_drafted_tokens() {
    run_case(GenerationSampling::Seeded, 0).await;
}

#[tokio::test]
#[ignore = "measures essay reply speed with one drafted token a round on an installed streamed model"]
#[serial]
async fn should_measure_essay_reply_speed_with_one_drafted_token() {
    run_case(GenerationSampling::Seeded, 1).await;
}

#[tokio::test]
#[ignore = "measures essay reply speed with two drafted tokens a round on an installed streamed model"]
#[serial]
async fn should_measure_essay_reply_speed_with_two_drafted_tokens() {
    run_case(GenerationSampling::Seeded, 2).await;
}

#[tokio::test]
#[ignore = "measures essay reply speed with three drafted tokens a round on an installed streamed model"]
#[serial]
async fn should_measure_essay_reply_speed_with_three_drafted_tokens() {
    run_case(GenerationSampling::Seeded, 3).await;
}

#[tokio::test]
#[ignore = "measures code-temperature reply speed one token a forward on an installed streamed model"]
#[serial]
async fn should_measure_code_temperature_reply_speed_without_drafted_tokens() {
    run_case(GenerationSampling::SeededForCode, 0).await;
}

#[tokio::test]
#[ignore = "measures code-temperature reply speed with one drafted token a round on an installed streamed model"]
#[serial]
async fn should_measure_code_temperature_reply_speed_with_one_drafted_token() {
    run_case(GenerationSampling::SeededForCode, 1).await;
}

#[tokio::test]
#[ignore = "attributes an essay reply's time one token a forward on an installed streamed model"]
#[serial]
async fn should_attribute_essay_reply_time_without_drafted_tokens() {
    run_attributed_case(0).await;
}

#[tokio::test]
#[ignore = "attributes an essay reply's time with one drafted token a round on an installed streamed model"]
#[serial]
async fn should_attribute_essay_reply_time_with_one_drafted_token() {
    run_attributed_case(1).await;
}

/// Serves the measured reply with performance attribution on and keeps its
/// report in `evidence_directory`; attribution slows the reply, so only the
/// shares of its time are to be read, not its speed.
async fn run_attributed_case(draft_token_count: u32) {
    timeout(
        CASE_TIMEOUT,
        measure_reply(
            GenerationSampling::Seeded,
            draft_token_count,
            RequestAttribution::Enabled,
        ),
    )
    .await
    .expect("the attributed drafting case must finish within its budget");
}

async fn run_case(generation_sampling: GenerationSampling, draft_token_count: u32) {
    timeout(
        CASE_TIMEOUT,
        measure_reply(
            generation_sampling,
            draft_token_count,
            RequestAttribution::Disabled,
        ),
    )
    .await
    .expect("the drafting speed case must finish within its budget");
}

async fn measure_reply(
    generation_sampling: GenerationSampling,
    draft_token_count: u32,
    request_attribution: RequestAttribution,
) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let PagingExperimentEngine {
        mut engine,
        end_of_sequence_token_ids,
        model_directory,
        attribution_directory,
    } = load_paging_experiment_engine("mtp_drafting_model_load").await;
    engine
        .set_multi_token_prediction_draft_tokens_for_tests(draft_token_count)
        .await
        .expect("the engine takes the draft count");
    let model_id = model_directory
        .file_name()
        .and_then(|name| name.to_str())
        .expect("the model directory has a UTF-8 name")
        .to_owned();
    let warmup_prompt_token_ids =
        crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
            &model_directory,
            &model_id,
            WARMUP_REQUEST_ID,
            WARMUP_PROMPT_TOKEN_COUNT,
            WARMUP_OUTPUT_TOKEN_COUNT,
        );
    run_generation_with_request_attribution(
        &mut engine,
        WARMUP_REQUEST_ID,
        &warmup_prompt_token_ids,
        "mtp_drafting_warmup",
        WARMUP_OUTPUT_TOKEN_COUNT,
        &end_of_sequence_token_ids,
        generation_sampling,
        RequestAttribution::Disabled,
    )
    .await;
    let prompt_token_ids =
        crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
            &model_directory,
            &model_id,
            MEASURED_REQUEST_ID,
            MEASURED_PROMPT_TOKEN_COUNT,
            MEASURED_OUTPUT_TOKEN_COUNT,
        );
    eprintln!(
        "[mtp-drafting] draft_tokens={draft_token_count} temperature_thousandths={}",
        generation_sampling.temperature_thousandths()
    );
    let reply_token_ids = run_generation_with_request_attribution(
        &mut engine,
        MEASURED_REQUEST_ID,
        &prompt_token_ids,
        &format!("mtp_drafts_{draft_token_count}"),
        MEASURED_OUTPUT_TOKEN_COUNT,
        &end_of_sequence_token_ids,
        generation_sampling,
        request_attribution,
    )
    .await;
    assert!(
        !reply_token_ids.is_empty(),
        "a reply with {draft_token_count} drafted tokens a round must produce tokens"
    );
    if let (RequestAttribution::Enabled, Some(evidence_directory)) = (
        request_attribution,
        TestRunContext::current().configuration.evidence_directory,
    ) {
        std::fs::create_dir_all(&evidence_directory).expect("the evidence directory is writable");
        std::fs::copy(
            attribution_directory.path().join("attribution.jsonl"),
            evidence_directory.join(format!("mtp_drafts_{draft_token_count}.jsonl")),
        )
        .expect("the attribution report copies to the evidence directory");
    }
}
