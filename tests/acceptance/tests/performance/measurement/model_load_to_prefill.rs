//! Measures `load_to_prefill_elapsed_millis`: the wall-clock interval from the
//! first chat request's acceptance on an idle worker to the exact moment
//! prompt prefill starts. Nothing else is measured or reported.
//!
//! Boundaries, and why they are the right ones:
//! - Start: the instant the journey submits its `ChatGenerationCommand`. On an
//!   idle worker that request triggers `SwapModel`, so this instant is the
//!   observable start of model loading (weight binding, tokenizer, any
//!   per-family startup work) plus tokenization and prompt-cache lookup.
//! - End: the arrival of the first `PrefillProgress` stream event carrying
//!   `processed_tokens == 0, elapsed_millis == 0`. The worker emits that event
//!   from `PromptProcessingPhaseStarted` — a confirmed prompt phase is about to
//!   begin, before any blocking prefill model work — so the boundary lands
//!   exactly on "prefill starts" and excludes prefill execution entirely.
//!
//! Method: one real worker process in an isolated temporary home (no SSD
//! prompt cache, fixed 2,048-token prefill chunks per the repo standard), one
//! ~500-token Romeo and Juliet warmup-fixture request, and a one-field JSON
//! report on stderr. The request runs to completion only so the journey can
//! verify it succeeded; no other metric is collected. Laptop-only: it loads
//! real weights into wired GPU memory, so it is `#[ignore]`d and not wired
//! into CI.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use astronomical_daemon::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationPerformanceLog,
    ResolvedRuntimeConfigResolver, WorkerHandle, WorkerHealthStatus,
};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use serde_json::json;
use serial_test::serial;
use tokio::time::{Instant, MissedTickBehavior, interval, sleep, timeout};

/// The whole journey (worker launch, idle-ready wait, one load-plus-request
/// pass) must finish within this budget; it stays under the repo's 120-second
/// test ceiling like the sibling prompt-cache warmup lane.
const BENCHMARK_TIMEOUT: Duration = Duration::from_secs(115);

/// The worker's launch handshake gets its own bounded wait, mirroring the
/// sibling measurement lanes.
const WORKER_LAUNCH_TIMEOUT: Duration = Duration::from_secs(60);

const READY_ATTEMPT_LIMIT: u8 = 70;

/// The repo-standard prefill chunk size, fixed so chunk boundaries can never
/// shift the measured prefill-start boundary between runs.
const FIXED_PROMPT_PROCESSING_CHUNK_TOKENS: u32 = 2_048;

/// The Romeo and Juliet warmup fixture (~500 tokens): large enough to pass
/// through tokenization and full phase-start machinery, small enough that
/// prefill execution stays negligible next to the load interval.
const WARMUP_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_warmup_romeo_and_juliet.txt");

const WARMUP_INPUT_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story above.";

/// A short decode tail keeps the journey bounded; the headline measurement is
/// complete the moment prefill starts. 32 clears the worker's minimum
/// output-token contract even with the thinking channel closed: transition
/// reserve plus at least one visible answer token (see the thinking-controls
/// contract: a disable maps to worker budget 0).
const MAXIMUM_OUTPUT_TOKENS: u16 = 32;

const PROGRESS_LOG_INTERVAL: Duration = Duration::from_secs(10);

#[test]
fn should_configure_model_directory_and_fixed_chunk_for_load_measurement_worker() {
    let (isolated_worker_home, _worker_executable_path) =
        isolated_load_measurement_worker_launcher(Path::new("/bin/true"), Path::new("/tmp/model"));
    let configuration_document: serde_json::Value = serde_json::from_slice(
        &fs::read(
            isolated_worker_home
                .path()
                .join(".astronomical-dev/config.json"),
        )
        .expect("the load-measurement worker config should be readable"),
    )
    .expect("the load-measurement worker config should be JSON");

    assert_eq!(
        configuration_document["chunking"]["fixed_prompt_processing_chunk_size_tokens"],
        FIXED_PROMPT_PROCESSING_CHUNK_TOKENS,
        "load-to-prefill measurement must pin the repo-standard prefill chunk size",
    );
    assert_eq!(
        configuration_document["model_directories"]
            .as_array()
            .expect("model directories should be an array")
            .len(),
        1,
        "the load-measurement worker should configure exactly one model directory",
    );
    assert!(
        configuration_document
            .get("prompt_cache_max_size_gb")
            .is_none(),
        "the load-measurement worker must not enable the SSD prompt cache",
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the resident sparse-MoE model and measures load-to-prefill-start latency over one request"]
#[serial]
async fn should_measure_model_load_to_prefill_start_latency() {
    timeout(
        BENCHMARK_TIMEOUT,
        run_load_to_prefill_measurement(BENCHMARK_TIMEOUT),
    )
    .await
    .expect("the load-to-prefill measurement must finish within 115 seconds");
}

struct LoadToPrefillMeasurement {
    load_to_prefill_elapsed_millis: u128,
    prompt_token_count: u32,
    cached_token_count: u32,
}

async fn run_load_to_prefill_measurement(journey_timeout: Duration) {
    let model_id = crate::support::resident_sparse_moe_model_id();
    let configured_model_directory =
        crate::support::configured_installed_model_directory_by_id(model_id);
    let production_worker_executable_path =
        astronomical_test_harness::workspace_binary("astronomical-inference-worker");
    let (isolated_worker_home, worker_executable_path) = isolated_load_measurement_worker_launcher(
        Path::new(&production_worker_executable_path),
        &configured_model_directory,
    );

    eprintln!("[load-to-prefill] launching the idle {model_id} worker");
    let performance_log_directory = isolated_worker_home.path().join("logs");
    fs::create_dir_all(&performance_log_directory)
        .expect("the load-measurement performance log directory should be created");
    let worker_runtime_config = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_worker_home.path().to_path_buf(),
        PathBuf::from(&production_worker_executable_path),
    )
    .load()
    .expect("the isolated load-measurement worker configuration should resolve");
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        WORKER_LAUNCH_TIMEOUT,
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the load-measurement performance log should open"),
        worker_runtime_config.model_policy_catalog.clone(),
        worker_runtime_config.worker_startup_configuration(),
    )
    .await
    .expect("the daemon should launch the load-measurement worker");
    wait_until_idle(&worker_handle).await;

    let measurement = run_single_measured_request(&worker_handle, model_id, journey_timeout).await;

    worker_handle
        .shutdown()
        .await
        .expect("the measured worker should terminate and be reaped");

    assert!(
        measurement.load_to_prefill_elapsed_millis > 0,
        "the load-to-prefill interval must be positive",
    );
    assert!(
        measurement.prompt_token_count > 0,
        "the measured request must have carried prompt tokens",
    );
    assert_eq!(
        measurement.cached_token_count, 0,
        "the isolated cold worker must not restore SSD-cached tokens",
    );

    let measurement_report = json!({
        "model": model_id,
        "measurement": "load_to_prefill",
        "load_to_prefill_elapsed_millis": measurement.load_to_prefill_elapsed_millis,
    });
    eprintln!("[load-to-prefill] {measurement_report}");
}

async fn run_single_measured_request(
    worker_handle: &WorkerHandle,
    model_id: &str,
    journey_timeout: Duration,
) -> LoadToPrefillMeasurement {
    let request_accepted_at = Instant::now();
    let mut stream_receiver = worker_handle
        .start_chat_generation(ChatGenerationCommand {
            request_id: RequestId::new(1),
            model: model_id.to_owned(),
            messages: vec![ChatMessage::User {
                content: format!("{WARMUP_INPUT_INSTRUCTION}\n\n{WARMUP_ROMEO_AND_JULIET_SOURCE}"),
                images: Vec::new(),
            }],
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: ChatGenerationSettings {
                max_output_tokens: MAXIMUM_OUTPUT_TOKENS,
                temperature_thousandths: None,
                top_p_thousandths: None,
                repetition_penalty_thousandths: None,
                seed: None,
                thinking_budget: Some(0),
            },
            structured_generation: None,
        })
        .await
        .expect("the worker should start the load-measurement request");
    eprintln!("[load-to-prefill] request accepted; the idle worker now loads the model");

    let mut prefill_start_arrival_at: Option<Instant> = None;
    let mut completed_token_counts: Option<(u32, u32)> = None;

    let mut progress_log_interval = interval(PROGRESS_LOG_INTERVAL);
    progress_log_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    progress_log_interval.tick().await;

    loop {
        let stream_event = tokio::select! {
            stream_event = stream_receiver.recv() => stream_event,
            _ = progress_log_interval.tick() => {
                let elapsed = request_accepted_at.elapsed();
                let remaining = journey_timeout.saturating_sub(elapsed);
                eprintln!(
                    "[load-to-prefill] waiting; elapsed={}s, deadline <= {}s",
                    elapsed.as_secs(),
                    remaining.as_secs(),
                );
                continue;
            }
        };
        let Some(stream_event) = stream_event else {
            break;
        };
        match stream_event {
            ChatGenerationStreamEvent::PrefillProgress {
                processed_tokens,
                elapsed_millis,
                ..
            } => {
                if prefill_start_arrival_at.is_none() {
                    assert_eq!(
                        processed_tokens, 0,
                        "the first prefill event must be the phase-start boundary",
                    );
                    assert_eq!(
                        elapsed_millis, 0,
                        "the prefill-start boundary event must carry zero prefill time",
                    );
                    let boundary_arrival_at = Instant::now();
                    prefill_start_arrival_at = Some(boundary_arrival_at);
                    eprintln!(
                        "[load-to-prefill] prefill started; load_to_prefill_elapsed_millis={}",
                        boundary_arrival_at
                            .duration_since(request_accepted_at)
                            .as_millis(),
                    );
                }
            }
            ChatGenerationStreamEvent::ReasoningFragment(_)
            | ChatGenerationStreamEvent::TextFragment(_)
            | ChatGenerationStreamEvent::ToolCall { .. } => {}
            ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                cached_token_count,
                ..
            } => {
                completed_token_counts = Some((prompt_token_count, cached_token_count));
                break;
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                panic!("the worker failed the load-measurement request: {reason:?}");
            }
            ChatGenerationStreamEvent::Error(error_code) => {
                panic!("the worker stream reported an error: {error_code:?}");
            }
        }
    }

    let prefill_start_arrival_at = prefill_start_arrival_at
        .expect("the journey must observe the prefill-start boundary event");
    let (prompt_token_count, cached_token_count) =
        completed_token_counts.expect("the terminal worker event should report the token counts");
    LoadToPrefillMeasurement {
        load_to_prefill_elapsed_millis: prefill_start_arrival_at
            .duration_since(request_accepted_at)
            .as_millis(),
        prompt_token_count,
        cached_token_count,
    }
}

fn isolated_load_measurement_worker_launcher(
    production_worker_executable_path: &Path,
    model_directory: &Path,
) -> (tempfile::TempDir, PathBuf) {
    let isolated_worker_home =
        tempfile::tempdir().expect("the load-measurement worker home should be created");
    let configuration_directory = isolated_worker_home.path().join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the load-measurement configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": FIXED_PROMPT_PROCESSING_CHUNK_TOKENS,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the load-measurement configuration should serialize"),
    )
    .expect("the load-measurement configuration should be written");

    (
        isolated_worker_home,
        production_worker_executable_path.to_path_buf(),
    )
}

async fn wait_until_idle(worker_handle: &WorkerHandle) {
    for readiness_attempt in 1..=READY_ATTEMPT_LIMIT {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready
            && worker_health_snapshot.ready_model_id.is_none()
        {
            eprintln!("[load-to-prefill] idle worker ready after {readiness_attempt} attempts");
            return;
        }
        let remaining_seconds = u16::from(READY_ATTEMPT_LIMIT - readiness_attempt);
        eprintln!(
            "[load-to-prefill] startup attempt {readiness_attempt}/{READY_ATTEMPT_LIMIT}, ETA <= {remaining_seconds}s"
        );
        sleep(Duration::from_secs(1)).await;
    }
    panic!("the real worker did not become idle before the load-measurement deadline");
}
