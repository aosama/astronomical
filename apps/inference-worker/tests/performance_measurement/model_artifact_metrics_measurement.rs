use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use astronomical_inference_worker::worker_startup::sample_iogpu_wired_limit_bytes;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationPerformanceLog,
    ResolvedRuntimeConfigResolver, WorkerHandle, WorkerHealthStatus,
};
use serde_json::json;
use tokio::time::{Instant, interval, sleep, timeout};

use super::model_process_metrics::{find_worker_process_id, measure_worker_physical_footprint};

const BENCHMARK_TIMEOUT: Duration = Duration::from_secs(115);
const DOCUMENT_WORD_COUNT: usize = 5_000;
const FIXED_BENCHMARK_PREFILL_CHUNCK_TOKENS: u32 = 8_192;
const MAXIMUM_SUMMARY_TOKENS: u16 = 2_000;
const MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
const READY_ATTEMPT_LIMIT: u8 = 70;
const SOURCE_DOCUMENT_FIXTURE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const CACHE_SCALING_FIXTURE: &str =
    include_str!("../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

struct StreamingMeasurement {
    request_started_at: Instant,
    first_output_at: Instant,
    response_completed_at: Instant,
    completion_token_count: u64,
    cached_token_count: u64,
    generated_text: String,
    maximum_prefill_mlx_active_memory_bytes: Option<u64>,
    maximum_prefill_mlx_allocator_cache_memory_bytes: Option<u64>,
    maximum_prefill_mlx_peak_memory_bytes: Option<u64>,
    prompt_token_count: u64,
}

struct MetricsCase {
    instruction: &'static str,
    maximum_output_tokens: u16,
    source_document: &'static str,
    source_word_count: usize,
    should_require_three_paragraphs: bool,
}

#[test]
fn should_keep_static_metrics_fixture_at_exactly_five_thousand_words() {
    assert_eq!(
        static_source_document(SOURCE_DOCUMENT_FIXTURE)
            .split_whitespace()
            .count(),
        DOCUMENT_WORD_COUNT
    );
}

#[test]
fn should_keep_cache_scaling_fixture_at_exactly_fifty_thousand_words() {
    assert_eq!(CACHE_SCALING_FIXTURE.split_whitespace().count(), 50_000);
}

#[test]
fn should_disable_prefill_chunck_optimizer_for_summary_metrics_worker() {
    let (isolated_worker_home, _worker_executable_path) = isolated_prompt_cache_worker_launcher(
        Path::new("/bin/true"),
        Path::new("/tmp/model-artifact-measurement"),
    );
    let configuration_document_path = isolated_worker_home
        .path()
        .join(".astronomical/config.json");
    let configuration_document: serde_json::Value = serde_json::from_slice(
        &fs::read(&configuration_document_path)
            .expect("the metrics worker config should be readable"),
    )
    .expect("the metrics worker config should be JSON");

    assert_eq!(
        configuration_document["prefill_chunck_size_optimizer_enabled"], false,
        "summary metrics must measure fixed-size prefill chunks, not adaptive optimizer output"
    );
    assert_eq!(
        configuration_document["fixed_prefill_chunck_tokens"],
        FIXED_BENCHMARK_PREFILL_CHUNCK_TOKENS,
        "summary metrics should use the fixed benchmark prefill chunk size"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model and benchmarks a 5,000-word summarization request"]
async fn should_measure_model_artifact_summarization_throughput_and_peak_memory() {
    timeout(
        BENCHMARK_TIMEOUT,
        run_model_artifact_measurement(five_thousand_word_case()),
    )
    .await
    .expect("the model-artifact measurement must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model and measures cold-cache 50,000-word prefill"]
async fn should_measure_model_artifact_cold_prefill_at_fifty_thousand_words() {
    timeout(
        BENCHMARK_TIMEOUT,
        run_model_artifact_measurement(fifty_thousand_word_case()),
    )
    .await
    .expect("the model-artifact cold-cache 50K measurement must finish within 115 seconds");
}

async fn run_model_artifact_measurement(metrics_case: MetricsCase) {
    let configured_model_directory = configured_model_directory_from_user_config();
    let production_worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the production inference-worker executable path");
    let (isolated_worker_home, worker_executable_path) = isolated_prompt_cache_worker_launcher(
        Path::new(&production_worker_executable_path),
        &configured_model_directory,
    );
    let maximum_gpu_wired_memory_bytes = sample_iogpu_wired_limit_bytes()
        .await
        .expect("the machine GPU wired-memory limit should be available");

    eprintln!("[performance-measurement] launching the model-artifact Ornith worker");
    let performance_log_directory = isolated_worker_home.path().join("logs");
    fs::create_dir_all(&performance_log_directory)
        .expect("the metrics performance log directory should be created");
    let worker_runtime_config = ResolvedRuntimeConfigResolver::new(
        isolated_worker_home.path().to_path_buf(),
        PathBuf::from(&production_worker_executable_path),
    )
    .load()
    .expect("the isolated metrics worker configuration should resolve");
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        Duration::from_secs(60),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the metrics performance log should open"),
        crate::common::single_model_directories(MODEL_ID, &configured_model_directory),
        u32::from(metrics_case.maximum_output_tokens),
        worker_runtime_config.worker_startup_configuration(),
    )
    .await
    .expect("the supervisor should launch the real worker");
    wait_until_idle(&worker_handle).await;
    let worker_process_id = find_worker_process_id().await;
    let source_document = static_source_document(metrics_case.source_document);
    eprintln!(
        "[performance-measurement] sending {} source words from the static fixture",
        metrics_case.source_word_count,
    );
    let streaming_measurement =
        measure_worker_summarization(&worker_handle, &source_document, &metrics_case).await;
    let physical_footprint_result = measure_worker_physical_footprint(worker_process_id).await;

    worker_handle
        .shutdown()
        .await
        .expect("the measured worker should terminate and be reaped");

    let physical_footprint =
        physical_footprint_result.expect("Apple footprint should report worker memory");
    assert_eq!(
        source_document.split_whitespace().count(),
        metrics_case.source_word_count
    );
    if metrics_case.maximum_output_tokens > 1 {
        assert!(!streaming_measurement.generated_text.trim().is_empty());
    }
    if metrics_case.should_require_three_paragraphs {
        assert!(
            streaming_measurement
                .generated_text
                .split("\n\n")
                .filter(|paragraph| !paragraph.trim().is_empty())
                .count()
                >= 3,
            "the deterministic request should produce at least three paragraphs"
        );
    }
    assert!(streaming_measurement.prompt_token_count > 0);
    assert!(streaming_measurement.completion_token_count > 0);
    assert_eq!(
        streaming_measurement.cached_token_count, 0,
        "the performance test must measure model prefill rather than SSD cache restoration"
    );

    let prompt_processing_duration = streaming_measurement
        .first_output_at
        .duration_since(streaming_measurement.request_started_at);
    let generation_duration = streaming_measurement
        .response_completed_at
        .duration_since(streaming_measurement.first_output_at);
    let total_request_duration = streaming_measurement
        .response_completed_at
        .duration_since(streaming_measurement.request_started_at);
    let prompt_tokens_per_second =
        streaming_measurement.prompt_token_count as f64 / prompt_processing_duration.as_secs_f64();
    let generation_tokens_per_second =
        (streaming_measurement.completion_token_count > 1).then(|| {
            (streaming_measurement.completion_token_count - 1) as f64
                / generation_duration.as_secs_f64()
        });
    let output_paragraph_count = streaming_measurement
        .generated_text
        .split("\n\n")
        .filter(|paragraph| !paragraph.trim().is_empty())
        .count();

    let metrics_report = json!({
        "completion_tokens": streaming_measurement.completion_token_count,
        "cached_tokens": streaming_measurement.cached_token_count,
        "generation_seconds": generation_duration.as_secs_f64(),
        "generation_tokens_per_second": generation_tokens_per_second,
        "gpu_wired_limit_bytes": maximum_gpu_wired_memory_bytes,
        "max_output_tokens": metrics_case.maximum_output_tokens,
        "maximum_prefill_mlx_active_memory_bytes": streaming_measurement.maximum_prefill_mlx_active_memory_bytes,
        "maximum_prefill_mlx_allocator_cache_memory_bytes": streaming_measurement.maximum_prefill_mlx_allocator_cache_memory_bytes,
        "maximum_prefill_mlx_peak_memory_bytes": streaming_measurement.maximum_prefill_mlx_peak_memory_bytes,
        "model": MODEL_ID,
        "output_paragraphs": output_paragraph_count,
        "peak_worker_physical_footprint_bytes": physical_footprint.peak_bytes,
        "prompt_processing_seconds": prompt_processing_duration.as_secs_f64(),
        "prompt_processing_tokens_per_second": prompt_tokens_per_second,
        "prompt_tokens": streaming_measurement.prompt_token_count,
        "seed": 1,
        "source_words": metrics_case.source_word_count,
        "temperature": 1.0,
        "top_p": 0.95,
        "total_request_seconds": total_request_duration.as_secs_f64(),
    });
    eprintln!("[performance-measurement] {metrics_report}");
}

fn configured_model_directory_from_user_config() -> PathBuf {
    crate::common::configured_model_artifact_directory_by_id(MODEL_ID)
}

fn isolated_prompt_cache_worker_launcher(
    production_worker_executable_path: &Path,
    model_directory: &Path,
) -> (tempfile::TempDir, PathBuf) {
    let isolated_worker_home = tempfile::tempdir()
        .expect("the isolated prompt-cache metrics worker home should be created");
    let configuration_directory = isolated_worker_home.path().join(".astronomical");
    fs::create_dir(&configuration_directory)
        .expect("the isolated prompt-cache metrics configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "prefill_chunck_size_optimizer_enabled": false,
        "fixed_prefill_chunck_tokens": FIXED_BENCHMARK_PREFILL_CHUNCK_TOKENS,
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the isolated prompt-cache metrics configuration should serialize"),
    )
    .expect("the isolated prompt-cache metrics configuration should be written");

    (
        isolated_worker_home,
        production_worker_executable_path.to_path_buf(),
    )
}

fn static_source_document(source_document: &str) -> String {
    source_document
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

async fn wait_until_idle(worker_handle: &WorkerHandle) {
    for readiness_attempt in 1..=READY_ATTEMPT_LIMIT {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready
            && worker_health_snapshot.ready_model_id.is_none()
        {
            eprintln!(
                "[performance-measurement] idle worker ready after {readiness_attempt} attempts"
            );
            return;
        }
        let remaining_seconds = u16::from(READY_ATTEMPT_LIMIT - readiness_attempt);
        eprintln!(
            "[performance-measurement] startup attempt {readiness_attempt}/{READY_ATTEMPT_LIMIT}, ETA <= {remaining_seconds}s"
        );
        sleep(Duration::from_secs(1)).await;
    }
    panic!("the real worker did not become idle before the metrics deadline");
}

async fn measure_worker_summarization(
    worker_handle: &WorkerHandle,
    source_document: &str,
    metrics_case: &MetricsCase,
) -> StreamingMeasurement {
    let user_prompt = format!("{}\n\n{source_document}", metrics_case.instruction);
    let request_started_at = Instant::now();
    let mut stream_receiver = worker_handle
        .start_chat_generation(ChatGenerationCommand {
            request_id: RequestId::new(1),
            model: MODEL_ID.to_owned(),
            messages: vec![ChatMessage::User {
                content: user_prompt,
                images: Vec::new(),
            }],
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: ChatGenerationSettings {
                max_output_tokens: metrics_case.maximum_output_tokens,
                temperature_thousandths: Some(1_000),
                top_p_thousandths: Some(950),
                seed: Some(1),
                thinking_budget: None,
            },
        })
        .await
        .expect("the worker should start the metrics generation request");
    let mut first_output_at: Option<Instant> = None;
    let mut generated_text = String::new();
    let mut prompt_token_count: Option<u64> = None;
    let mut completion_token_count: Option<u64> = None;
    let mut cached_token_count: Option<u64> = None;
    let mut maximum_prefill_mlx_active_memory_bytes: Option<u64> = None;
    let mut maximum_prefill_mlx_allocator_cache_memory_bytes: Option<u64> = None;
    let mut maximum_prefill_mlx_peak_memory_bytes: Option<u64> = None;
    let mut progress_interval = interval(Duration::from_secs(10));
    progress_interval.tick().await;
    eprintln!(
        "[performance-measurement] worker request accepted, waiting for first generated output"
    );
    loop {
        let stream_event = tokio::select! {
            stream_event = stream_receiver.recv() => stream_event,
            _ = progress_interval.tick(), if first_output_at.is_none() => {
                let elapsed = request_started_at.elapsed();
                let remaining = BENCHMARK_TIMEOUT.saturating_sub(elapsed);
                eprintln!(
                    "[performance-measurement] prompt processing elapsed={}s, ETA <= {}s",
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
            ChatGenerationStreamEvent::ReasoningFragment(text)
            | ChatGenerationStreamEvent::TextFragment(text) => {
                if first_output_at.is_none() {
                    first_output_at = Some(Instant::now());
                    eprintln!("[performance-measurement] first generated output received");
                }
                generated_text.push_str(&text);
            }
            ChatGenerationStreamEvent::ToolCall {
                function_name,
                arguments_json,
                ..
            } => {
                if first_output_at.is_none() {
                    first_output_at = Some(Instant::now());
                    eprintln!("[performance-measurement] first generated tool call received");
                }
                generated_text.push_str(&function_name);
                generated_text.push_str(&arguments_json);
            }
            ChatGenerationStreamEvent::PrefillProgress {
                mlx_active_memory_bytes,
                mlx_allocator_cache_memory_bytes,
                mlx_peak_memory_bytes,
                ..
            } => {
                maximum_prefill_mlx_active_memory_bytes = maximum_optional_u64(
                    maximum_prefill_mlx_active_memory_bytes,
                    mlx_active_memory_bytes,
                );
                maximum_prefill_mlx_allocator_cache_memory_bytes = maximum_optional_u64(
                    maximum_prefill_mlx_allocator_cache_memory_bytes,
                    mlx_allocator_cache_memory_bytes,
                );
                maximum_prefill_mlx_peak_memory_bytes = maximum_optional_u64(
                    maximum_prefill_mlx_peak_memory_bytes,
                    mlx_peak_memory_bytes,
                );
            }
            ChatGenerationStreamEvent::Completed {
                prompt_token_count: completed_prompt_token_count,
                generated_token_count,
                cached_token_count: completed_cached_token_count,
                ..
            } => {
                prompt_token_count = Some(u64::from(completed_prompt_token_count));
                completion_token_count = Some(u64::from(generated_token_count));
                cached_token_count = Some(u64::from(completed_cached_token_count));
                break;
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                panic!("the worker failed the metrics generation request: {reason:?}");
            }
            ChatGenerationStreamEvent::Error(error_code) => {
                panic!("the worker stream reported an error: {error_code:?}");
            }
        }
    }
    let response_completed_at = Instant::now();
    StreamingMeasurement {
        request_started_at,
        first_output_at: first_output_at.unwrap_or(response_completed_at),
        response_completed_at,
        completion_token_count: completion_token_count
            .expect("the terminal worker event should report completion tokens"),
        cached_token_count: cached_token_count
            .expect("the terminal worker event should report cached prompt tokens"),
        generated_text,
        maximum_prefill_mlx_active_memory_bytes,
        maximum_prefill_mlx_allocator_cache_memory_bytes,
        maximum_prefill_mlx_peak_memory_bytes,
        prompt_token_count: prompt_token_count
            .expect("the terminal worker event should report prompt tokens"),
    }
}

fn maximum_optional_u64(current_maximum: Option<u64>, candidate: Option<u64>) -> Option<u64> {
    match (current_maximum, candidate) {
        (Some(current_maximum), Some(candidate)) => Some(current_maximum.max(candidate)),
        (Some(current_maximum), None) => Some(current_maximum),
        (None, Some(candidate)) => Some(candidate),
        (None, None) => None,
    }
}

fn five_thousand_word_case() -> MetricsCase {
    MetricsCase {
        instruction: "Summarize the following document in exactly three concise paragraphs. Do not use headings or bullet points.",
        maximum_output_tokens: MAXIMUM_SUMMARY_TOKENS,
        source_document: SOURCE_DOCUMENT_FIXTURE,
        source_word_count: DOCUMENT_WORD_COUNT,
        should_require_three_paragraphs: true,
    }
}

fn fifty_thousand_word_case() -> MetricsCase {
    MetricsCase {
        instruction: "Read the following public-domain book excerpt and reply with OK.",
        maximum_output_tokens: 1,
        source_document: CACHE_SCALING_FIXTURE,
        source_word_count: 50_000,
        should_require_three_paragraphs: false,
    }
}
