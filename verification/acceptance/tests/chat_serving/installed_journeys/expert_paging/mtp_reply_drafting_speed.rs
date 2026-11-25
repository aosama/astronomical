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
    run_reply_case(ReplyCase {
        generation_sampling: GenerationSampling::Seeded,
        draft_token_count,
        route_lookup: RouteLookup::InsideTheGraph,
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
        request_attribution: RequestAttribution::Disabled,
        phase_name: format!("mtp_drafts_{draft_token_count}"),
    })
    .await;
}

/// How a reply token finds the table slots of its routed experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RouteLookup {
    /// A lookup inside the token's graph, as the engine serves.
    InsideTheGraph,
    /// Each layer's route copied to the host, as the engine served before.
    OnTheHost,
}

/// One measured reply: how its tokens are sampled, drafted and routed.
pub(super) struct ReplyCase {
    pub(super) generation_sampling: GenerationSampling,
    pub(super) draft_token_count: u32,
    pub(super) route_lookup: RouteLookup,
    pub(super) request_attribution: RequestAttribution,
    /// Names the case in its `[performance-attribution] phase=` line and its
    /// evidence file.
    pub(super) phase_name: String,
}

pub(super) async fn run_reply_case(reply_case: ReplyCase) {
    timeout(CASE_TIMEOUT, measure_reply(reply_case))
        .await
        .expect("the reply speed case must finish within its budget");
}

async fn measure_reply(reply_case: ReplyCase) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let PagingExperimentEngine {
        mut engine,
        end_of_sequence_token_ids,
        model_directory,
        attribution_directory,
    } = load_paging_experiment_engine("mtp_drafting_model_load").await;
    engine
        .set_multi_token_prediction_draft_tokens_for_tests(reply_case.draft_token_count)
        .await
        .expect("the engine takes the draft count");
    engine
        .set_routes_looked_up_in_the_graph_for_tests(
            reply_case.route_lookup == RouteLookup::InsideTheGraph,
        )
        .await
        .expect("the engine takes the route lookup");
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
        reply_case.generation_sampling,
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
        "[reply-speed] phase={} draft_tokens={} route_lookup={:?} temperature_thousandths={}",
        reply_case.phase_name,
        reply_case.draft_token_count,
        reply_case.route_lookup,
        reply_case.generation_sampling.temperature_thousandths()
    );
    let reply_token_ids = run_generation_with_request_attribution(
        &mut engine,
        MEASURED_REQUEST_ID,
        &prompt_token_ids,
        &reply_case.phase_name,
        MEASURED_OUTPUT_TOKEN_COUNT,
        &end_of_sequence_token_ids,
        reply_case.generation_sampling,
        reply_case.request_attribution,
    )
    .await;
    assert!(
        !reply_token_ids.is_empty(),
        "the {} reply must produce tokens",
        reply_case.phase_name
    );
    eprintln!(
        "[reply-speed] phase={} in_graph_counts={:?} direct_fetch_counts={:?}",
        reply_case.phase_name,
        engine
            .in_graph_serving_counts_for_tests()
            .await
            .expect("the engine counts its lookups"),
        engine
            .take_direct_expert_fetch_counts_for_tests()
            .await
            .expect("the engine counts its fetches")
    );
    report_reply_stages(&reply_case.phase_name, &engine).await;
    if let (RequestAttribution::Enabled, Some(evidence_directory)) = (
        reply_case.request_attribution,
        TestRunContext::current().configuration.evidence_directory,
    ) {
        std::fs::create_dir_all(&evidence_directory).expect("the evidence directory is writable");
        std::fs::copy(
            attribution_directory.path().join("attribution.jsonl"),
            evidence_directory.join(format!("{}.jsonl", reply_case.phase_name)),
        )
        .expect("the attribution report copies to the evidence directory");
    }
}

/// Prints where the measured reply's time went inside the engine, a
/// forward: the standing stage tally, so no case needs timers of its own.
async fn report_reply_stages(phase_name: &str, engine: &impl Qwen3_5StreamingEngineTestControls) {
    let Some(stages) = engine
        .last_reply_stage_counts_for_tests()
        .await
        .expect("the engine keeps the last reply's stages")
    else {
        return;
    };
    let forward_count = stages.reply_forward_count.max(1) as f64;
    let round_count = stages.draft_round_count.max(1) as f64;
    let milliseconds = |nanoseconds: u64, count: f64| nanoseconds as f64 / 1_000_000.0 / count;
    eprintln!(
        "[reply-stages] phase={phase_name} forwards={} rows={} lookups_per_forward={:.1} lookup_read_ms_per_forward={:.3} lookup_bookkeeping_ms_per_forward={:.3} lookup_bookkeeping_microseconds_per_lookup={:.1} blocking_evaluations_per_forward={:.2} asynchronous_evaluations_per_forward={:.2} waits_for_submitted_per_forward={:.2} cross_stream_waits_per_forward={:.1}",
        stages.reply_forward_count,
        stages.reply_row_count,
        stages.lookup_count as f64 / forward_count,
        milliseconds(stages.lookup_storage_read_nanoseconds, forward_count),
        milliseconds(stages.lookup_bookkeeping_nanoseconds(), forward_count),
        stages.lookup_bookkeeping_nanoseconds() as f64
            / 1_000.0
            / (stages.lookup_count.max(1) as f64),
        stages.hand_offs.blocking_evaluations as f64 / forward_count,
        stages.hand_offs.asynchronous_evaluations as f64 / forward_count,
        stages.hand_offs.waits_for_submitted_arrays as f64 / forward_count,
        stages.hand_offs.cross_stream_waits as f64 / forward_count,
    );
    if stages.draft_round_count > 0 {
        eprintln!(
            "[reply-stages] phase={phase_name} draft_rounds={} draft_ms_per_round={:.2} verify_ms_per_round={:.2} pick_ms_per_round={:.2} keep_ms_per_round={:.2}",
            stages.draft_round_count,
            milliseconds(stages.draft_nanoseconds, round_count),
            milliseconds(stages.verify_nanoseconds, round_count),
            milliseconds(stages.pick_nanoseconds, round_count),
            milliseconds(stages.keep_nanoseconds, round_count),
        );
    }
}
