//! How often a reply's tokens miss the `astro_expert_cache`, and the routes
//! behind it.
//!
//! A miss is one routed expert a reply token needed that the
//! `astro_expert_cache` did not hold when its layer ran, so it was read from
//! storage. Each case loads a sparse model under the configured MLX limit,
//! keeps its files out of the `macos_file_cache` (`cold_macos_file_cache`),
//! serves a 500/50 warmup and then one measured reply, and prints that reply's
//! misses, hits, bytes read and read wait per token from the engine's
//! always-on counters. Per-token figures divide by every reply token,
//! including the first, which comes out of the prompt's last step and reads
//! nothing; that understates them by one part in the reply's length.
//!
//! The recording cases serve the measured reply with performance attribution
//! on, which is what makes the engine copy each decode token's route to the
//! host, and save the reply's routes and the `astro_expert_cache`'s state when
//! it began to `evidence_directory`, for the replay lab in
//! `astronomical-memory-policy`. Attribution slows decode, so their speed is
//! not a measurement; the speed case serves the Shakespeare reply with
//! attribution off and is the one to read tokens per second from.
//!
//! Run the recordings with `cargo measure-expert-misses` and the speed cases
//! (the `macos_file_cache` evicted, then warm) with
//! `cargo measure-expert-misses-speed`, each with `--configuration PATH`
//! whose `model_directory`, `paging_experiment.maximum_mlx_memory_gigabytes`
//! and optional `evidence_directory` set the case.

use std::path::Path;
use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_memory_policy::astro_expert_cache_model::RoutedReply;
use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5StreamingEngine, Qwen3_5Tokenizer, RoutedReplyRecording,
    SessionExpertRouteSample,
};
use astronomical_runtime_integration::MlxMemoryLimits;
use astronomical_test_harness::test_run::TestRunContext;
use serial_test::serial;
use tokio::time::timeout;

use super::cold_macos_file_cache::ColdMacosFileCache;
use crate::serving_acceptance::support::performance_attribution::{
    GenerationSampling, RequestAttribution, create_attributed_engine, load_engine_with_progress,
    run_generation_with_request_attribution,
};

const WARMUP_REQUEST_ID: RequestId = RequestId::new(97_501);
const MEASURED_REQUEST_ID: RequestId = RequestId::new(97_502);
const WARMUP_PROMPT_TOKEN_COUNT: usize = 500;
const WARMUP_OUTPUT_TOKEN_COUNT: u16 = 50;
const MEASURED_OUTPUT_TOKEN_COUNT: u16 = 500;
const SHAKESPEARE_PROMPT_TOKEN_COUNT: usize = 10_000;
const CASE_TIMEOUT: Duration = Duration::from_secs(120);

/// The C++ expert record reader: real code a person might ask about.
const CODING_SOURCE_TEXT: &str =
    include_str!("../../../../../../crates/expert-record-reader/native/expert_record_reader.cpp");

/// One measured reply's text and sampling.
#[derive(Clone, Copy, Debug)]
enum MeasuredWorkload {
    ShakespeareSummary,
    CodeReview,
    MultiTurnChat,
}

impl MeasuredWorkload {
    const fn name(self) -> &'static str {
        match self {
            Self::ShakespeareSummary => "shakespeare_summary",
            Self::CodeReview => "code_review",
            Self::MultiTurnChat => "multi_turn_chat",
        }
    }

    const fn sampling(self) -> GenerationSampling {
        match self {
            Self::CodeReview => GenerationSampling::SeededForCode,
            Self::ShakespeareSummary | Self::MultiTurnChat => GenerationSampling::Seeded,
        }
    }

    fn prompt_token_ids(self, model_directory: &Path) -> Vec<u32> {
        match self {
            Self::ShakespeareSummary => {
                crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
                    model_directory,
                    &model_id_of(model_directory),
                    MEASURED_REQUEST_ID,
                    SHAKESPEARE_PROMPT_TOKEN_COUNT,
                    MEASURED_OUTPUT_TOKEN_COUNT,
                )
            }
            Self::CodeReview => chat_prompt_token_ids(model_directory, code_review_messages()),
            Self::MultiTurnChat => chat_prompt_token_ids(model_directory, multi_turn_chat_messages()),
        }
    }
}

#[tokio::test]
#[ignore = "records a 10,000-token Shakespeare summary's expert routes and misses on an installed sparse model with a cold macOS file cache"]
#[serial]
async fn should_count_expert_misses_while_recording_a_shakespeare_summary() {
    run_case(
        MeasuredWorkload::ShakespeareSummary,
        RequestAttribution::Enabled,
    )
    .await;
}

#[tokio::test]
#[ignore = "records a code review's expert routes and misses on an installed sparse model with a cold macOS file cache"]
#[serial]
async fn should_count_expert_misses_while_recording_a_code_review() {
    run_case(MeasuredWorkload::CodeReview, RequestAttribution::Enabled).await;
}

#[tokio::test]
#[ignore = "records a multi-turn chat reply's expert routes and misses on an installed sparse model with a cold macOS file cache"]
#[serial]
async fn should_count_expert_misses_while_recording_a_multi_turn_chat() {
    run_case(MeasuredWorkload::MultiTurnChat, RequestAttribution::Enabled).await;
}

#[tokio::test]
#[ignore = "serves a 10,000-token Shakespeare summary without attribution on an installed sparse model with a cold macOS file cache and counts its expert misses"]
#[serial]
async fn should_count_expert_misses_and_decode_speed_without_attribution() {
    run_case(
        MeasuredWorkload::ShakespeareSummary,
        RequestAttribution::Disabled,
    )
    .await;
}

#[tokio::test]
#[ignore = "serves a 10,000-token Shakespeare summary without attribution on an installed sparse model with a warm macOS file cache and counts its expert misses"]
#[serial]
async fn should_count_expert_misses_and_decode_speed_with_a_warm_macos_file_cache() {
    timeout(
        CASE_TIMEOUT,
        count_expert_misses_with(
            MeasuredWorkload::ShakespeareSummary,
            RequestAttribution::Disabled,
            MacosFileCacheState::Warm,
        ),
    )
    .await
    .expect("the expert miss case must finish within 120 seconds");
}

/// Whether the model's files are kept out of the `macos_file_cache` while
/// the case serves.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MacosFileCacheState {
    Cold,
    Warm,
}

async fn run_case(measured_workload: MeasuredWorkload, request_attribution: RequestAttribution) {
    timeout(
        CASE_TIMEOUT,
        count_expert_misses_with(
            measured_workload,
            request_attribution,
            MacosFileCacheState::Cold,
        ),
    )
    .await
    .expect("the expert miss case must finish within 120 seconds");
}

async fn count_expert_misses_with(
    measured_workload: MeasuredWorkload,
    request_attribution: RequestAttribution,
    macos_file_cache_state: MacosFileCacheState,
) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_run_configuration = TestRunContext::current().configuration;
    let model_directory = test_run_configuration
        .model_directory
        .expect("model_directory names the sparse model");
    let paging_experiment = test_run_configuration.paging_experiment;
    let mlx_limit_bytes = usize::try_from(
        paging_experiment
            .maximum_mlx_memory_bytes()
            .expect("paging_experiment sets a valid MLX limit"),
    )
    .expect("the MLX limit should fit usize");
    let attribution_directory =
        tempfile::tempdir().expect("the case should create an attribution directory");
    let mlx_memory_limits = MlxMemoryLimits::new(mlx_limit_bytes, mlx_limit_bytes)
        .expect("the configured limit should be a valid MLX limit");
    let (mut engine, end_of_sequence_token_ids) = create_attributed_engine(
        &model_directory,
        &attribution_directory.path().join("unused.jsonl"),
        &mlx_memory_limits,
        paging_experiment.streaming_prompt_processing_chunk_tokens,
    );
    load_engine_with_progress(&mut engine, "expert_miss_model_load").await;
    let cold_macos_file_cache = if macos_file_cache_state == MacosFileCacheState::Cold {
        Some(ColdMacosFileCache::start(&model_directory))
    } else {
        None
    };
    eprintln!("[expert-misses] macos_file_cache={macos_file_cache_state:?}");
    let warmup_prompt_token_ids =
        crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
            &model_directory,
            &model_id_of(&model_directory),
            WARMUP_REQUEST_ID,
            WARMUP_PROMPT_TOKEN_COUNT,
            WARMUP_OUTPUT_TOKEN_COUNT,
        );
    run_generation_with_request_attribution(
        &mut engine,
        WARMUP_REQUEST_ID,
        &warmup_prompt_token_ids,
        "expert_miss_warmup",
        WARMUP_OUTPUT_TOKEN_COUNT,
        &end_of_sequence_token_ids,
        GenerationSampling::Seeded,
        RequestAttribution::Disabled,
    )
    .await;
    // The warmup is unattributed, so it leaves no recording; this clears any
    // snapshot an earlier reply left behind.
    let _ = engine.take_routed_reply_recording_for_tests().await;
    let measured_reply = serve_measured_reply(
        &mut engine,
        measured_workload,
        request_attribution,
        &model_directory,
        &end_of_sequence_token_ids,
    )
    .await;
    report_measured_reply(measured_workload, request_attribution, &measured_reply);
    if let Some(cold_macos_file_cache) = cold_macos_file_cache {
        eprintln!(
            "[expert-misses] macos_file_cache_sweeps={}",
            cold_macos_file_cache.finish()
        );
    }
    if let (Some(recording), Some(evidence_directory)) = (
        engine
            .take_routed_reply_recording_for_tests()
            .await
            .expect("the engine hands over its reply recording"),
        test_run_configuration.evidence_directory,
    ) {
        // The first reply token comes out of the prompt's last step, so the
        // routes cover every token after it; anything else means the ring
        // dropped routes and the replay would not stand for this reply.
        assert_eq!(
            recording.token_routes.len() + 1,
            measured_reply.reply_token_count,
            "the recording must hold one route per reply token after the first"
        );
        save_recording(measured_workload, &recording, &evidence_directory);
    }
}

/// What the measured reply did, from the engine's always-on counters.
struct MeasuredReply {
    reply_token_count: usize,
    route_change: SessionExpertRouteSample,
}

async fn serve_measured_reply(
    engine: &mut Qwen3_5StreamingEngine,
    measured_workload: MeasuredWorkload,
    request_attribution: RequestAttribution,
    model_directory: &Path,
    end_of_sequence_token_ids: &[u32],
) -> MeasuredReply {
    let prompt_token_ids = measured_workload.prompt_token_ids(model_directory);
    eprintln!(
        "[expert-misses] begin workload={} prompt_tokens={}",
        measured_workload.name(),
        prompt_token_ids.len()
    );
    let route_sample_before = engine
        .session_expert_route_sample_for_tests()
        .await
        .expect("the engine samples its expert counters");
    let generated_token_ids = run_generation_with_request_attribution(
        engine,
        MEASURED_REQUEST_ID,
        &prompt_token_ids,
        measured_workload.name(),
        MEASURED_OUTPUT_TOKEN_COUNT,
        end_of_sequence_token_ids,
        measured_workload.sampling(),
        request_attribution,
    )
    .await;
    let route_sample_after = engine
        .session_expert_route_sample_for_tests()
        .await
        .expect("the engine samples its expert counters");
    assert!(
        !generated_token_ids.is_empty(),
        "the measured reply must generate"
    );
    MeasuredReply {
        reply_token_count: generated_token_ids.len(),
        route_change: route_sample_after.change_since(&route_sample_before),
    }
}

fn report_measured_reply(
    measured_workload: MeasuredWorkload,
    request_attribution: RequestAttribution,
    measured_reply: &MeasuredReply,
) {
    let reply_token_count = measured_reply.reply_token_count as f64;
    let route_change = measured_reply.route_change;
    eprintln!(
        "[expert-misses] workload={} attribution={} reply_tokens={} expert_misses_per_token={:.2} expert_hits_per_token={:.2} storage_read_megabytes_per_token={:.2} storage_read_wait_milliseconds_per_token={:.2}",
        measured_workload.name(),
        matches!(request_attribution, RequestAttribution::Enabled),
        measured_reply.reply_token_count,
        route_change.storage_read_expert_count as f64 / reply_token_count,
        route_change.memory_served_expert_count as f64 / reply_token_count,
        route_change.decode_storage_read_bytes as f64 / reply_token_count / 1_048_576.0,
        route_change.decode_storage_read_wait_nanoseconds as f64 / reply_token_count / 1_000_000.0,
    );
}

/// Writes `<workload>.routes` (the routes, in the replay lab's format) and
/// `<workload>.case.json` (what the `astro_expert_cache` held when the reply
/// began, its capacity and each layer's expert size).
fn save_recording(
    measured_workload: MeasuredWorkload,
    recording: &RoutedReplyRecording,
    evidence_directory: &Path,
) {
    let routed_reply = RoutedReply::from_token_routes(
        u16::try_from(recording.experts_per_layer).expect("experts per layer fit u16"),
        &recording.token_routes,
    )
    .expect("the engine's routes form a recording");
    std::fs::create_dir_all(evidence_directory).expect("the evidence directory is writable");
    let routes_path = evidence_directory.join(format!("{}.routes", measured_workload.name()));
    std::fs::write(&routes_path, routed_reply.to_recording_bytes()).expect("the routes write");
    let case_document = serde_json::json!({
        "workload": measured_workload.name(),
        "experts_per_layer": recording.experts_per_layer,
        "kept_expert_capacity_bytes": recording.kept_expert_capacity_bytes,
        "warm_slot_count": recording.warm_slot_count,
        "expert_payload_bytes_by_layer": recording.expert_payload_bytes_by_layer,
        "held_expert_ids_by_layer_at_reply_start": recording.held_expert_ids_by_layer_at_reply_start,
        "held_expert_ids_by_layer_at_reply_end": recording.held_expert_ids_by_layer_at_reply_end,
        "prompt_demand_by_layer_at_reply_start": recording.prompt_demand_by_layer_at_reply_start,
        "reply_demand_by_layer_at_reply_start": recording.reply_demand_by_layer_at_reply_start,
    });
    let case_path = evidence_directory.join(format!("{}.case.json", measured_workload.name()));
    std::fs::write(
        &case_path,
        serde_json::to_vec_pretty(&case_document).expect("the case document serializes"),
    )
    .expect("the case document writes");
    eprintln!(
        "[expert-misses] recorded tokens={} layers={} routes={} case={}",
        routed_reply.token_count(),
        routed_reply.layer_count(),
        routes_path.display(),
        case_path.display()
    );
}

fn code_review_messages() -> Vec<ChatMessage> {
    vec![ChatMessage::User {
        content: format!(
            "Here is a C++ file from a Mac app that reads model weights from an SSD.\n\n```cpp\n{CODING_SOURCE_TEXT}\n```\n\nExplain how a batch of reads is shared between the calling thread and the helper threads, point out one race or edge case worth testing, and write a unit test for it in C++."
        ),
        images: Vec::new(),
    }]
}

fn multi_turn_chat_messages() -> Vec<ChatMessage> {
    vec![
        ChatMessage::User {
            content: "I'm planning a week in Japan in April with my parents, who don't walk far. Where should we base ourselves?".to_owned(),
            images: Vec::new(),
        },
        ChatMessage::Assistant {
            content: Some("For a first week with limited walking, two bases work well: Kyoto for four nights, using taxis between temples and the Philosopher's Path when the cherry blossoms are out, and Tokyo for three nights near a major station such as Tokyo Station or Shinjuku, so day trips start without transfers.".to_owned()),
            reasoning_content: None,
            tool_calls: Vec::new(),
        },
        ChatMessage::User {
            content: "That sounds good. My mother is vegetarian and my father loves trains. Can you plan the Kyoto days in detail, with food stops and one train experience he'll remember?".to_owned(),
            images: Vec::new(),
        },
    ]
}

fn chat_prompt_token_ids(model_directory: &Path, messages: Vec<ChatMessage>) -> Vec<u32> {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, u32::from(MEASURED_OUTPUT_TOKEN_COUNT))
        .expect("the configured model should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the configured tokenizer should load");
    tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id: MEASURED_REQUEST_ID,
                model: model_id_of(model_directory),
                messages,
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                settings: ChatGenerationSettings {
                    max_output_tokens: MEASURED_OUTPUT_TOKEN_COUNT,
                    temperature_thousandths: None,
                    top_p_thousandths: None,
                    seed: None,
                    thinking_budget: None,
                    repetition_penalty_thousandths: None,
                },
                structured_generation: None,
            },
            false,
        )
        .expect("the chat prompt should prepare")
        .input_token_ids()
        .to_vec()
}

fn model_id_of(model_directory: &Path) -> String {
    model_directory
        .file_name()
        .and_then(|name| name.to_str())
        .expect("the model directory has a UTF-8 name")
        .to_owned()
}
