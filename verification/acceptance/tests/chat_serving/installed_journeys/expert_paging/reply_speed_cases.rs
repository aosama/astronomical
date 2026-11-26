//! The reply cases every streamed reply-speed journey measures, and the
//! serving they share: a 500/50 warm-up, one unmeasured reply when the case
//! asks for it, then the measured 10,000-token reply, read on the engine's
//! own reply loop. The multi-token-prediction, hosted-drafter, in-graph
//! lookup and resident journeys all ask their engine the same thing through
//! `serve_warmup_learning_and_measured_replies` (issue #2585).

use std::path::Path;
use std::time::Duration;

use ipc_protocol::RequestId;
use model_serving::{InferenceEngine, Qwen3_5StreamingEngineTestControls, TextInferenceRequest};
use test_harness::test_run::TestRunContext;
use tokio::time::timeout;

use crate::serving_acceptance::support::performance_attribution::{
    GenerationSampling, PagingExperimentEngine, RequestAttribution, load_paging_experiment_engine,
    run_generation_with_request_attribution,
};

const WARMUP_REQUEST_ID: RequestId = RequestId::new(98_501);
const MEASURED_REQUEST_ID: RequestId = RequestId::new(98_502);
const LEARNING_REQUEST_ID: RequestId = RequestId::new(98_503);
const WARMUP_PROMPT_TOKEN_COUNT: usize = 500;
const WARMUP_OUTPUT_TOKEN_COUNT: u16 = 50;
const MEASURED_PROMPT_TOKEN_COUNT: usize = 10_000;
/// A dense model reads a prompt layer by layer at a few hundred tokens a
/// second (the 27B: 250 to 300), so 10,000 tokens cost over half a minute a
/// reply for nothing a shorter prompt does not show.
const DENSE_MEASURED_PROMPT_TOKEN_COUNT: usize = 2_000;

/// The measured prompt's size for the configured model: shorter for a model
/// whose configuration names no experts.
fn measured_prompt_token_count(model_directory: &Path) -> usize {
    let has_experts = std::fs::read_to_string(model_directory.join("config.json"))
        .is_ok_and(|config_text| config_text.contains("\"num_experts\""));
    if has_experts {
        MEASURED_PROMPT_TOKEN_COUNT
    } else {
        DENSE_MEASURED_PROMPT_TOKEN_COUNT
    }
}
const MEASURED_OUTPUT_TOKEN_COUNT: u16 = 500;
/// A dense model that streams its layers writes about ten tokens a second
/// and reads a 10,000-token prompt slowly; a case that serves its prompt
/// twice needs about six minutes there.
const CASE_TIMEOUT: Duration = Duration::from_secs(480);

/// A short reply two ways of serving are compared on: past the
/// sparse-attention budget of a lane-mixing model, and short enough to
/// read twice in a comparison's budget.
pub(super) const COMPARED_PROMPT_TOKEN_COUNT: usize = 2_600;
pub(super) const COMPARED_OUTPUT_TOKEN_COUNT: u16 = 96;

/// Writes the greedy reply to the compared Romeo and Juliet summary on the
/// engine as it is set up, and returns its tokens.
pub(super) async fn write_compared_greedy_reply(
    paging_engine: &mut PagingExperimentEngine,
    request_id: RequestId,
    phase_name: &str,
) -> Vec<u32> {
    let model_id = paging_engine
        .model_directory
        .file_name()
        .and_then(|name| name.to_str())
        .expect("the model directory has a UTF-8 name")
        .to_owned();
    let prompt_token_ids =
        crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
            &paging_engine.model_directory,
            &model_id,
            request_id,
            COMPARED_PROMPT_TOKEN_COUNT,
            COMPARED_OUTPUT_TOKEN_COUNT,
        );
    run_generation_with_request_attribution(
        &mut paging_engine.engine,
        request_id,
        &prompt_token_ids,
        phase_name,
        COMPARED_OUTPUT_TOKEN_COUNT,
        &paging_engine.end_of_sequence_token_ids,
        GenerationSampling::Greedy,
        RequestAttribution::Disabled,
    )
    .await
}

/// How a reply token finds the table slots of its routed experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RouteLookup {
    /// A lookup inside the token's graph, as the engine serves.
    InsideTheGraph,
    /// Each layer's route copied to the host, as the engine served before.
    OnTheHost,
}

/// What the measured reply is asked to write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReplyTask {
    /// A three-paragraph summary of Romeo and Juliet.
    Essay,
    /// The continuation of real Python source.
    Code,
}

impl ReplyTask {
    fn prompt_token_ids(
        self,
        model_directory: &std::path::Path,
        model_id: &str,
        request_id: RequestId,
        prompt_token_count: usize,
        output_token_count: u16,
    ) -> Vec<u32> {
        use crate::serving_acceptance::support::{romeo_and_juliet, source_code_prompt};
        let instructed_source = match self {
            Self::Essay => romeo_and_juliet::ROMEO_AND_JULIET_SUMMARY,
            Self::Code => source_code_prompt::SOURCE_CODE_CONTINUATION,
        };
        romeo_and_juliet::prepare_instructed_source_prompt(
            instructed_source,
            model_directory,
            model_id,
            request_id,
            prompt_token_count,
            output_token_count,
        )
    }
}

/// One measured reply: how its tokens are sampled, drafted and routed.
pub(super) struct ReplyCase {
    pub(super) generation_sampling: GenerationSampling,
    pub(super) draft_token_count: u32,
    pub(super) route_lookup: RouteLookup,
    /// Off keeps a hosted DFlash drafter loaded and silent, for the plain
    /// reply its rounds are compared with; nothing to a model without one.
    pub(super) hosted_drafter_writes_replies: bool,
    pub(super) reply_task: ReplyTask,
    /// Serves the measured prompt once before it is measured, so a model
    /// with a hosted drafter has timed its plain tokens and rounds and the
    /// measured reply is decided as every reply after a first one is.
    pub(super) serves_the_prompt_once_before_measuring: bool,
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

/// Sets the engine up as the case says, before anything is served.
async fn set_engine_up_for(
    engine: &impl Qwen3_5StreamingEngineTestControls,
    reply_case: &ReplyCase,
) {
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
    engine
        .set_hosted_drafter_writes_replies_for_tests(reply_case.hosted_drafter_writes_replies)
        .await
        .expect("the engine takes the drafter's part");
}

async fn measure_reply(reply_case: ReplyCase) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let PagingExperimentEngine {
        mut engine,
        end_of_sequence_token_ids,
        model_directory,
        attribution_directory,
    } = load_paging_experiment_engine("mtp_drafting_model_load").await;
    set_engine_up_for(&engine, &reply_case).await;
    let reply_token_ids = serve_warmup_learning_and_measured_replies(
        &mut engine,
        &reply_case,
        &model_directory,
        &end_of_sequence_token_ids,
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

/// Serves the warm-up, the unmeasured reply when the case asks for one, and
/// the measured reply on any text engine, so the streaming and the resident
/// engine are asked exactly the same thing (issue #2585).
pub(super) async fn serve_warmup_learning_and_measured_replies(
    engine: &mut impl InferenceEngine<Request = TextInferenceRequest>,
    reply_case: &ReplyCase,
    model_directory: &Path,
    end_of_sequence_token_ids: &[u32],
) -> Vec<u32> {
    let model_id = model_directory
        .file_name()
        .and_then(|name| name.to_str())
        .expect("the model directory has a UTF-8 name")
        .to_owned();
    let warmup_prompt_token_ids = reply_case.reply_task.prompt_token_ids(
        model_directory,
        &model_id,
        WARMUP_REQUEST_ID,
        WARMUP_PROMPT_TOKEN_COUNT,
        WARMUP_OUTPUT_TOKEN_COUNT,
    );
    run_generation_with_request_attribution(
        engine,
        WARMUP_REQUEST_ID,
        &warmup_prompt_token_ids,
        "reply_speed_warmup",
        WARMUP_OUTPUT_TOKEN_COUNT,
        end_of_sequence_token_ids,
        reply_case.generation_sampling,
        RequestAttribution::Disabled,
    )
    .await;
    let prompt_token_ids = reply_case.reply_task.prompt_token_ids(
        model_directory,
        &model_id,
        MEASURED_REQUEST_ID,
        measured_prompt_token_count(model_directory),
        MEASURED_OUTPUT_TOKEN_COUNT,
    );
    if reply_case.serves_the_prompt_once_before_measuring {
        run_generation_with_request_attribution(
            engine,
            LEARNING_REQUEST_ID,
            &prompt_token_ids,
            "reply_before_the_measured_one",
            MEASURED_OUTPUT_TOKEN_COUNT,
            end_of_sequence_token_ids,
            reply_case.generation_sampling,
            RequestAttribution::Disabled,
        )
        .await;
    }
    eprintln!(
        "[reply-speed] phase={} draft_tokens={} route_lookup={:?} temperature_thousandths={}",
        reply_case.phase_name,
        reply_case.draft_token_count,
        reply_case.route_lookup,
        reply_case.generation_sampling.temperature_thousandths()
    );
    let measured_reply_token_ids = run_generation_with_request_attribution(
        engine,
        MEASURED_REQUEST_ID,
        &prompt_token_ids,
        &reply_case.phase_name,
        MEASURED_OUTPUT_TOKEN_COUNT,
        end_of_sequence_token_ids,
        reply_case.generation_sampling,
        reply_case.request_attribution,
    )
    .await;
    report_memory_after_the_reply(&reply_case.phase_name);
    measured_reply_token_ids
}

/// MLX's active and peak memory once the measured reply is written, in GB of
/// 2^30 bytes, so the same reply can be compared for memory between engines
/// as well as for speed (issue #2585). The peak is MLX's since an engine last
/// reset it, so it bounds the reply from above.
fn report_memory_after_the_reply(phase_name: &str) {
    const BYTES_PER_GIGABYTE: f64 = 1_073_741_824.0;
    let active_bytes = mlx_bridge::active_memory_bytes().unwrap_or(0);
    let peak_bytes = mlx_bridge::peak_memory_bytes().unwrap_or(0);
    eprintln!(
        "[reply-memory] phase={phase_name} active_gb={:.2} peak_gb={:.2}",
        active_bytes as f64 / BYTES_PER_GIGABYTE,
        peak_bytes as f64 / BYTES_PER_GIGABYTE
    );
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
        "[reply-stages] phase={phase_name} forwards={} rows={} lookups_per_forward={:.1} lookup_read_ms_per_forward={:.3} lookup_bookkeeping_ms_per_forward={:.3} lookup_bookkeeping_microseconds_per_lookup={:.1} blocking_evaluations_per_forward={:.2} asynchronous_evaluations_per_forward={:.2} waits_for_submitted_per_forward={:.2} cross_stream_waits_per_forward={:.1} gpu_primitives_per_forward={:.0} cpu_primitives_per_forward={:.0} gpu_command_buffers_per_forward={:.1} gpu_busy_ms_per_forward={:.2}",
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
        stages.hand_offs.gpu_primitives as f64 / forward_count,
        stages.hand_offs.cpu_primitives as f64 / forward_count,
        stages.hand_offs.gpu_command_buffers as f64 / forward_count,
        milliseconds(stages.hand_offs.gpu_busy_nanoseconds, forward_count),
    );
    if stages.draft_round_count > 0 {
        eprintln!(
            "[reply-stages] phase={phase_name} draft_rounds={} draft_ms_per_round={:.2} verify_ms_per_round={:.2} pick_ms_per_round={:.2} keep_ms_per_round={:.2} blocking_evaluations_per_round_draft_verify_pick_keep={:.2}/{:.2}/{:.2}/{:.2}",
            stages.draft_round_count,
            milliseconds(stages.draft_nanoseconds, round_count),
            milliseconds(stages.verify_nanoseconds, round_count),
            milliseconds(stages.pick_nanoseconds, round_count),
            milliseconds(stages.keep_nanoseconds, round_count),
            stages.draft_blocking_evaluations as f64 / round_count,
            stages.verify_blocking_evaluations as f64 / round_count,
            stages.pick_blocking_evaluations as f64 / round_count,
            stages.keep_blocking_evaluations as f64 / round_count,
        );
    }
}
