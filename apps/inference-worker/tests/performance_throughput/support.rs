//! IPC worker driver that measures serving throughput for one model.
//!
//! The driver launches the production inference-worker process over the
//! supervisor IPC boundary, runs one short warmup completion to spin up first-use
//! JIT kernels, then one measured completion over the real prompt, and reads the
//! worker's authoritative per-request throughput (`prefill_tok_per_second` and
//! `generation_tok_per_second`) from the generation performance log. The warmup
//! is discarded so JIT compilation never inflates the measured prefill; the
//! measured completion's values feed the reported record. The test case (warmup
//! prompt and output cap, measured prompt and output cap, temperature) is defined
//! by the test that calls this driver, so the test case is readable where the
//! test is.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatImageInput, ChatMessage, ChatToolChoice,
    RequestId, WorkerStartupConfiguration,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationPerformanceLog,
    ResolvedRuntimeConfigResolver, RuntimeModelPolicy, WorkerHandle, WorkerHealthStatus,
};
use serde::Deserialize;
use tokio::time::sleep;

use crate::performance_throughput::historical_record::{
    ThroughputJourneyKind, ThroughputRecord, append_throughput_history, current_unix_epoch_millis,
    format_utc_timestamp, history_log_path, recorded_git_commit, throughput_record_json,
};
use crate::performance_throughput::machine_specs::MachineSpecs;

pub(crate) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(60);
const READY_ATTEMPT_LIMIT: u8 = 70;
const WORKER_LOG_DUMP_LINE_LIMIT: usize = 60;
const DIAGNOSTICS_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_THROUGHPUT_DIAGNOSTICS";

/// Diagnostic runs turn on worker attribution and info logging to explain where
/// time goes; their throughput is not production-faithful and is never recorded.
fn diagnostics_enabled() -> bool {
    std::env::var(DIAGNOSTICS_ENVIRONMENT_VARIABLE).is_ok_and(|value| value == "1")
}

/// The test case for one throughput journey: the journey family recorded in
/// the durable history line, the short warmup completion (its prompt, images,
/// and output cap), the measured completion (its prompt, images, and output
/// cap), and the sampling temperature. The test that defines these values
/// lives beside this driver so the test case is readable where the test is.
#[derive(Clone, Debug)]
pub(crate) struct ThroughputJourney {
    pub(crate) journey_kind: ThroughputJourneyKind,
    /// The MTP draft depth the journey's isolated worker configuration
    /// engages; `None` measures the MTP-off baseline.
    pub(crate) mtp_draft_depth: Option<u8>,
    pub(crate) warmup_input_prompt: String,
    pub(crate) warmup_images: Vec<ChatImageInput>,
    pub(crate) warmup_output_tokens: u16,
    pub(crate) measured_input_prompt: String,
    pub(crate) measured_images: Vec<ChatImageInput>,
    pub(crate) measured_output_tokens: u16,
    pub(crate) temperature_thousandths: u16,
}

/// Values from the single measured completion, after the warmup is discarded.
#[derive(Debug)]
struct ThroughputMeasurement {
    model_id: String,
    total_input_tokens: u32,
    total_output_tokens: u16,
    cached_tokens: u32,
    prefill_time_seconds: f64,
    decode_time_seconds: f64,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
}

/// One line of the worker performance log, re-parsed for the fields the report needs.
#[derive(Debug, Deserialize)]
struct ThroughputSample {
    prompt_token_count: u32,
    cached_token_count: u32,
    generated_token_count: u16,
    prefill_tok_per_second: Option<f64>,
    generation_tok_per_second: Option<f64>,
    prefill_elapsed_millis: u64,
    generation_elapsed_millis: u64,
}

/// Builds an isolated Development home pinned to the measured model and resolves
/// the supervisor-owned bootstrap settings for the worker. The worker's logging
/// directory is pointed at a persistent per-model directory so its log lines
/// survive the isolated home's panic-unwind cleanup and stay readable after a
/// failed journey.
/// The directory to advertise as a `model_directories` entry for `model_directory`.
///
/// For a HuggingFace-cache snapshot (`.../models--org--repo/snapshots/<hash>`), returns
/// the `models--org--repo` entry root so discovery derives the decoded `org/repo`
/// identity. For any other layout the directory is already named by its model id, so it
/// is returned unchanged.
fn discovery_root_for_model_directory(model_directory: &Path) -> &Path {
    model_directory
        .ancestors()
        .find(|ancestor| {
            ancestor
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("models--"))
        })
        .unwrap_or(model_directory)
}

pub(crate) fn perf_worker_environment(
    model_id: &str,
    model_directory: &Path,
    mtp_draft_depth: Option<u8>,
) -> (
    tempfile::TempDir,
    PathBuf,
    PathBuf,
    Arc<HashMap<String, RuntimeModelPolicy>>,
    WorkerStartupConfiguration,
) {
    let production_worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the production inference-worker executable path"),
    );
    let isolated_worker_home =
        tempfile::tempdir().expect("the throughput worker home should be created");
    let configuration_directory = isolated_worker_home.path().join(".astronomical-dev");
    fs::create_dir_all(&configuration_directory)
        .expect("the throughput worker configuration directory should be created");
    // A HuggingFace-cache model must be advertised through its `models--org--repo`
    // entry root, not the raw `snapshots/<hash>` leaf. Discovery only decodes the
    // `org/repo` identity from a `models--` directory; pointed at the snapshot
    // leaf it names the model by the hash, so the requested model id never
    // resolves and the worker rejects the request as unavailable.
    let discovery_root = discovery_root_for_model_directory(model_directory);
    let mut configuration_document = serde_json::json!({
        "model_directories": [discovery_root],
        "persistent_prompt_cache_enabled": false,
    });
    if diagnostics_enabled() {
        // Attribution adds host synchronization to every multi-token forward
        // and info logging adds I/O, so a diagnostic run explains where time
        // goes but its throughput is distorted. Production-faithful measured
        // runs leave both off.
        configuration_document["logging"] = serde_json::json!({ "level": "info" });
        configuration_document["performance_attribution_enabled"] = serde_json::Value::Bool(true);
    }
    if let Some(mtp_draft_depth) = mtp_draft_depth {
        configuration_document["mtp_enabled"] = serde_json::Value::Bool(true);
        configuration_document["mtp_draft_depth"] = mtp_draft_depth.into();
    }
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the throughput worker configuration should serialize"),
    )
    .expect("the throughput worker configuration should be written");
    let resolved_configuration = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_worker_home.path().to_path_buf(),
        PathBuf::from(&production_worker_executable_path),
    )
    .load()
    .expect("the throughput worker configuration should resolve");
    let persistent_logging_directory = std::env::temp_dir()
        .join("astronomical-throughput-logs")
        .join(model_id);
    if persistent_logging_directory.exists() {
        fs::remove_dir_all(&persistent_logging_directory)
            .expect("the stale throughput logging directory should be removed");
    }
    let mut worker_startup_configuration = resolved_configuration.worker_startup_configuration();
    worker_startup_configuration.logging_directory = persistent_logging_directory.clone();
    (
        isolated_worker_home,
        production_worker_executable_path,
        persistent_logging_directory,
        resolved_configuration.model_policy_catalog.clone(),
        worker_startup_configuration,
    )
}

/// Drives one completion to its terminal event, returning when the worker reports
/// completion so its performance record is written.
async fn drive_completion(
    worker_handle: &WorkerHandle,
    request_id: RequestId,
    model_id: &str,
    prompt: &str,
    images: Vec<ChatImageInput>,
    output_tokens: u16,
    temperature_thousandths: u16,
) -> Result<(), String> {
    let command = ChatGenerationCommand {
        request_id,
        model: model_id.to_owned(),
        messages: vec![ChatMessage::User {
            content: prompt.to_owned(),
            images,
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: output_tokens,
            temperature_thousandths: Some(temperature_thousandths),
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
        structured_generation: None,
    };
    let mut receiver = match worker_handle.start_chat_generation(command).await {
        Ok(receiver) => receiver,
        Err(error) => {
            return Err(format!(
                "the worker rejected the completion request: {error:?}"
            ));
        }
    };
    while let Some(event) = receiver.recv().await {
        match event {
            ChatGenerationStreamEvent::Completed { .. } => return Ok(()),
            ChatGenerationStreamEvent::Failed { reason } => {
                return Err(format!(
                    "the worker failed the completion request: {reason:?}"
                ));
            }
            ChatGenerationStreamEvent::Error(error_code) => {
                return Err(format!(
                    "the worker stream reported a completion error: {error_code:?}"
                ));
            }
            _ => {}
        }
    }
    Err("the worker stream closed without a terminal completion event".to_owned())
}

/// Drives a completion and, when it fails, prints the worker's own log lines
/// from the persistent logging directory so a dead worker's last words are
/// never lost inside a bare WorkerUnavailable panic.
async fn drive_completion_with_exit_diagnostics(
    worker_handle: &WorkerHandle,
    request_id: RequestId,
    model_id: &str,
    prompt: &str,
    images: Vec<ChatImageInput>,
    output_tokens: u16,
    temperature_thousandths: u16,
    logging_directory: &Path,
) {
    if let Err(error) = drive_completion(
        worker_handle,
        request_id,
        model_id,
        prompt,
        images,
        output_tokens,
        temperature_thousandths,
    )
    .await
    {
        dump_worker_logging_directory(logging_directory);
        panic!("the throughput completion for {model_id} failed: {error}");
    }
}

/// Prints the tail of every file in the worker's persistent logging directory
/// so the last lines a failed worker wrote stay in the journey output.
fn dump_worker_logging_directory(logging_directory: &Path) {
    let Ok(entries) = fs::read_dir(logging_directory) else {
        eprintln!(
            "[performance-throughput] no worker logging directory at {}",
            logging_directory.display()
        );
        return;
    };
    for entry in entries.flatten().filter(|entry| entry.path().is_file()) {
        let log_file_path = entry.path();
        let Ok(contents) = fs::read_to_string(&log_file_path) else {
            continue;
        };
        eprintln!(
            "[performance-throughput] worker log {} (tail):",
            log_file_path.display()
        );
        for log_line in contents.lines().rev().take(WORKER_LOG_DUMP_LINE_LIMIT) {
            eprintln!("[performance-throughput]   {log_line}");
        }
    }
}

async fn wait_until_idle(worker_handle: &WorkerHandle) {
    for _ in 1..=READY_ATTEMPT_LIMIT {
        let snapshot = worker_handle.worker_health_snapshot();
        if snapshot.status == WorkerHealthStatus::Ready && snapshot.ready_model_id.is_none() {
            return;
        }
        sleep(Duration::from_secs(1)).await;
    }
    panic!("the real worker did not become idle before the throughput deadline");
}

fn read_throughput_samples(log_path: &Path) -> Vec<ThroughputSample> {
    let text = fs::read_to_string(log_path).unwrap_or_default();
    text.lines()
        .filter_map(|line| serde_json::from_str::<ThroughputSample>(line).ok())
        .collect()
}

/// Summarizes the measured completion, which is the last performance sample the
/// worker wrote (the warmup sample precedes it), so JIT warmup never enters the
/// reported values.
fn summarize(model_id: &str, samples: &[ThroughputSample]) -> ThroughputMeasurement {
    let measured = samples
        .last()
        .expect("the measured completion should produce a performance sample");
    ThroughputMeasurement {
        model_id: model_id.to_owned(),
        total_input_tokens: measured.prompt_token_count,
        total_output_tokens: measured.generated_token_count,
        cached_tokens: measured.cached_token_count,
        prefill_time_seconds: measured.prefill_elapsed_millis as f64 / 1_000.0,
        decode_time_seconds: measured.generation_elapsed_millis as f64 / 1_000.0,
        prefill_tokens_per_second: measured.prefill_tok_per_second.unwrap_or(0.0),
        decode_tokens_per_second: measured.generation_tok_per_second.unwrap_or(0.0),
    }
}

/// Runs one journey on a dedicated multi-thread runtime and enforces the
/// built-in timeout so a wedged worker can never hang the test process.
pub(crate) fn run_journey_with_timeout(model_id: &'static str, journey: ThroughputJourney) {
    let (sender, receiver) = std::sync::mpsc::channel::<()>();
    let worker = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("the throughput worker runtime should build");
        runtime.block_on(run_throughput_journey(model_id, &journey));
        sender.send(()).ok();
    });
    match receiver.recv_timeout(JOURNEY_TIMEOUT) {
        Ok(()) => {}
        Err(RecvTimeoutError::Timeout) => panic!(
            "the throughput journey for {model_id} exceeded the {JOURNEY_TIMEOUT:?} deadline"
        ),
        Err(RecvTimeoutError::Disconnected) => {
            panic!("the throughput worker thread terminated before completing")
        }
    }
    worker
        .join()
        .expect("the throughput worker thread should join cleanly");
}

/// Runs one full throughput journey and persists a durable historical record.
pub(crate) async fn run_throughput_journey(model_id: &str, journey: &ThroughputJourney) {
    let machine_specs = MachineSpecs::capture().await;
    let measurement = measure_throughput(model_id, journey).await;
    let record = ThroughputRecord {
        timestamp: format_utc_timestamp(current_unix_epoch_millis()),
        model_id: measurement.model_id.clone(),
        journey: journey.journey_kind,
        mtp_draft_depth: journey.mtp_draft_depth,
        prefill_tokens_per_second: (measurement.prefill_tokens_per_second.round()) as u32,
        decode_tokens_per_second: (measurement.decode_tokens_per_second.round()) as u32,
        git_commit: recorded_git_commit(),
        machine: machine_specs,
        total_input_tokens: measurement.total_input_tokens,
        total_output_tokens: measurement.total_output_tokens,
        cached_tokens: measurement.cached_tokens,
        prefill_time_seconds: measurement.prefill_time_seconds,
        decode_time_seconds: measurement.decode_time_seconds,
    };
    eprintln!(
        "[performance-throughput] {}",
        serde_json::to_string_pretty(&throughput_record_json(&record))
            .unwrap_or_else(|_| "{ \"error\": \"record serialization failed\"".to_owned())
    );
    if diagnostics_enabled() {
        eprintln!(
            "[performance-throughput] diagnostics run: durable history append skipped because attribution distorts throughput"
        );
        return;
    }
    let history_path = history_log_path();
    match append_throughput_history(&record, &history_path) {
        Ok(()) => eprintln!(
            "[performance-throughput] appended durable record to {}",
            history_path.display()
        ),
        Err(error) => eprintln!(
            "[performance-throughput] durable history append failed (measurement still reported): {error}"
        ),
    }
}

async fn measure_throughput(model_id: &str, journey: &ThroughputJourney) -> ThroughputMeasurement {
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id);
    let (
        _isolated_home,
        worker_executable_path,
        logging_directory,
        model_policy_catalog,
        worker_startup_configuration,
    ) = perf_worker_environment(model_id, &model_directory, journey.mtp_draft_depth);
    let performance_log_path = logging_directory.join("performance.jsonl");
    fs::create_dir_all(&logging_directory)
        .expect("the throughput performance log directory should be created");

    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        MODEL_LOAD_TIMEOUT,
        GenerationPerformanceLog::open(&logging_directory)
            .expect("the throughput performance log should open"),
        model_policy_catalog,
        worker_startup_configuration,
    )
    .await
    .expect("the supervisor should launch the measured worker");
    wait_until_idle(&worker_handle).await;

    eprintln!("[performance-throughput] warmup model={model_id}");
    drive_completion_with_exit_diagnostics(
        &worker_handle,
        RequestId::new(1),
        model_id,
        &journey.warmup_input_prompt,
        journey.warmup_images.clone(),
        journey.warmup_output_tokens,
        journey.temperature_thousandths,
        &logging_directory,
    )
    .await;
    eprintln!("[performance-throughput] measured model={model_id}");
    drive_completion_with_exit_diagnostics(
        &worker_handle,
        RequestId::new(2),
        model_id,
        &journey.measured_input_prompt,
        journey.measured_images.clone(),
        journey.measured_output_tokens,
        journey.temperature_thousandths,
        &logging_directory,
    )
    .await;
    let samples = read_throughput_samples(&performance_log_path);
    worker_handle
        .shutdown()
        .await
        .expect("the measured worker should terminate and be reaped");
    summarize(model_id, &samples)
}
