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

use serial_test::serial;

use super::reply_speed_cases::{ReplyCase, ReplyTask, RouteLookup, run_reply_case};
use crate::serving_acceptance::support::performance_attribution::{
    GenerationSampling, RequestAttribution,
};

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
    run_reply_case(ReplyCase {
        generation_sampling: GenerationSampling::Seeded,
        draft_token_count,
        route_lookup: RouteLookup::InsideTheGraph,
        hosted_drafter_writes_replies: true,
        reply_task: ReplyTask::Essay,
        serves_the_prompt_once_before_measuring: false,
        request_attribution: RequestAttribution::Enabled,
        phase_name: format!("mtp_drafts_{draft_token_count}"),
    })
    .await;
}

async fn run_case(generation_sampling: GenerationSampling, draft_token_count: u32) {
    run_reply_case(ReplyCase {
        generation_sampling,
        draft_token_count,
        route_lookup: RouteLookup::InsideTheGraph,
        hosted_drafter_writes_replies: true,
        reply_task: ReplyTask::Essay,
        serves_the_prompt_once_before_measuring: false,
        request_attribution: RequestAttribution::Disabled,
        phase_name: format!("mtp_drafts_{draft_token_count}"),
    })
    .await;
}
