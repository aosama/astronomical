//! A streamed reply written in draft tree rounds by a hosted DFlash drafter,
//! against the same reply written one token a forward (issue #2330).
//!
//! The first journey compares: one loaded model writes one greedy reply
//! twice, with the drafter silent and with it proposing. Every token of
//! either reply is the target's own pick, but a tree pass is several rows
//! and MLX rounds a several-row kernel differently from a one-row one, so
//! the two replies part where two tokens are nearly tied (on a 35B-A3B the
//! logits move by up to 1.6 between row counts; `cargo
//! test-streaming-draft-tree-verification` holds each tree row to that). On
//! a dense model the one stage left that does so is full attention above
//! five rows a call (issue #2599).
//! The journey reports how far they agree and fails only when they part at
//! the first token, which rounding does not explain. The drafted reply
//! must have run rounds and the plain
//! one none, or the comparison proves nothing. The others are the
//! measurement, one process a case so neither starts from tables the other
//! warmed: a 500/50 warmup, then a 10,000-token Romeo and Juliet summary
//! and 500 reply tokens through the engine's own reply loop, served once
//! unmeasured so the model has timed its plain tokens and its rounds
//! (`draft_round_payoff`) and then measured, at temperature
//! 1.0 and at 0.7, and the same lengths of real Python source continued
//! at 0.7. Compare the cases' `[performance-attribution]
//! status=success` lines; the drafter stays loaded in both, so the plain
//! case is what the same memory serves without rounds.
//!
//! Run with `cargo test-hosted-drafter-reply --configuration PATH` and
//! `cargo measure-hosted-drafter-reply --configuration PATH`, whose
//! `model_directory` names a model packaged with a DFlash drafter and whose
//! `paging_experiment.maximum_mlx_memory_gigabytes` sets the limit. About
//! forty seconds for the proof and seventy a measured case.

use ipc_protocol::RequestId;
use model_serving::Qwen3_5StreamingEngineTestControls;
use serial_test::serial;
use tokio::time::{Duration, timeout};

use super::reply_speed_cases as reply_cases;
use crate::serving_acceptance::support::performance_attribution as engine_support;
use engine_support::{GenerationSampling, PagingExperimentEngine};
use reply_cases::{ReplyCase, run_reply_case};

const PLAIN_REQUEST_ID: RequestId = RequestId::new(98_701);
const DRAFTED_REQUEST_ID: RequestId = RequestId::new(98_702);
const PROOF_TIMEOUT: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "compares one greedy reply drafted and plain on an installed streamed model with a DFlash drafter"]
#[serial]
async fn should_begin_a_drafted_greedy_reply_as_the_plain_reply_begins() {
    timeout(PROOF_TIMEOUT, compare_greedy_replies())
        .await
        .expect("the hosted drafter proof must finish within its budget");
}

#[tokio::test]
#[ignore = "measures essay reply speed with a hosted drafter silent"]
#[serial]
async fn should_measure_essay_reply_speed_with_the_hosted_drafter_silent() {
    run_reply_case(speed_case(GenerationSampling::Seeded, false)).await;
}

#[tokio::test]
#[ignore = "measures essay reply speed with a hosted drafter proposing"]
#[serial]
async fn should_measure_essay_reply_speed_with_the_hosted_drafter_proposing() {
    run_reply_case(speed_case(GenerationSampling::Seeded, true)).await;
}

#[tokio::test]
#[ignore = "measures code-temperature reply speed with a hosted drafter silent"]
#[serial]
async fn should_measure_code_temperature_reply_speed_with_the_hosted_drafter_silent() {
    run_reply_case(speed_case(GenerationSampling::SeededForCode, false)).await;
}

#[tokio::test]
#[ignore = "measures code-temperature reply speed with a hosted drafter proposing"]
#[serial]
async fn should_measure_code_temperature_reply_speed_with_the_hosted_drafter_proposing() {
    run_reply_case(speed_case(GenerationSampling::SeededForCode, true)).await;
}

#[tokio::test]
#[ignore = "measures code-writing reply speed with a hosted drafter silent"]
#[serial]
async fn should_measure_code_writing_reply_speed_with_the_hosted_drafter_silent() {
    run_reply_case(code_case(false)).await;
}

#[tokio::test]
#[ignore = "measures code-writing reply speed with a hosted drafter proposing"]
#[serial]
async fn should_measure_code_writing_reply_speed_with_the_hosted_drafter_proposing() {
    run_reply_case(code_case(true)).await;
}

/// Real source continued at the temperature code is written at.
fn code_case(hosted_drafter_writes_replies: bool) -> ReplyCase {
    ReplyCase {
        reply_task: reply_cases::ReplyTask::Code,
        ..speed_case(
            GenerationSampling::SeededForCode,
            hosted_drafter_writes_replies,
        )
    }
}

fn phase_name(hosted_drafter_writes_replies: bool) -> String {
    if hosted_drafter_writes_replies {
        "hosted_drafter_proposing".to_owned()
    } else {
        "hosted_drafter_silent".to_owned()
    }
}

fn speed_case(
    generation_sampling: GenerationSampling,
    hosted_drafter_writes_replies: bool,
) -> ReplyCase {
    ReplyCase {
        reply_task: reply_cases::ReplyTask::Essay,
        serves_the_prompt_once_before_measuring: true,
        generation_sampling,
        draft_token_count: 0,
        route_lookup: reply_cases::RouteLookup::InsideTheGraph,
        hosted_drafter_writes_replies,
        request_attribution: engine_support::RequestAttribution::Disabled,
        phase_name: phase_name(hosted_drafter_writes_replies),
    }
}

async fn compare_greedy_replies() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut paging_engine =
        engine_support::load_paging_experiment_engine("hosted_drafter_model_load").await;
    let (plain_reply, plain_round_count) =
        write_greedy_reply(&mut paging_engine, PLAIN_REQUEST_ID, false).await;
    let (drafted_reply, drafted_round_count) =
        write_greedy_reply(&mut paging_engine, DRAFTED_REQUEST_ID, true).await;
    eprintln!(
        "[hosted-drafter-proof] reply_tokens={} plain_rounds={plain_round_count} drafted_rounds={drafted_round_count}",
        plain_reply.len()
    );
    assert_eq!(
        plain_round_count, 0,
        "the plain reply must be written without a round"
    );
    assert!(
        drafted_round_count > 0,
        "the drafted reply must run rounds for the comparison to prove anything"
    );
    assert!(
        plain_reply.len() > 1,
        "the compared reply must be more than one token"
    );
    let mut agreeing_token_count = 0;
    while agreeing_token_count < plain_reply.len().min(drafted_reply.len())
        && plain_reply[agreeing_token_count] == drafted_reply[agreeing_token_count]
    {
        agreeing_token_count += 1;
    }
    eprintln!(
        "[hosted-drafter-proof] agreeing_tokens={agreeing_token_count} of {}",
        plain_reply.len()
    );
    assert!(
        agreeing_token_count > 0,
        "a drafted reply must begin as the plain reply does"
    );
}

/// Writes the greedy reply with the drafter proposing or silent; returns
/// its tokens and the draft rounds it ran.
async fn write_greedy_reply(
    paging_engine: &mut PagingExperimentEngine,
    request_id: RequestId,
    hosted_drafter_writes_replies: bool,
) -> (Vec<u32>, u64) {
    paging_engine
        .engine
        .set_hosted_drafter_writes_replies_for_tests(hosted_drafter_writes_replies)
        .await
        .expect("the engine takes the drafter's part");
    let reply_token_ids = reply_cases::write_compared_greedy_reply(
        paging_engine,
        request_id,
        &phase_name(hosted_drafter_writes_replies),
    )
    .await;
    let draft_round_count = paging_engine
        .engine
        .last_reply_stage_counts_for_tests()
        .await
        .expect("the engine keeps the last reply's stages")
        .map_or(0, |stages| stages.draft_round_count);
    (reply_token_ids, draft_round_count)
}
