use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
    WorkerStartupConfiguration,
};
use astronomical_supervisor::{
    ActiveRequestProgress, ChatGenerationExecutor, ChatGenerationStreamEvent,
    GenerationPerformanceLog, ResolvedRuntimeConfigResolver, RuntimeModelPolicy, WorkerHandle,
    WorkerHealthStatus,
};
use serde_json::json;
use tokio::time::{Instant, MissedTickBehavior, interval, sleep};

use super::model_persistent_prompt_cache_benchmark_case::PersistentPromptCacheWarmupCase;
use super::model_process_metrics::{
    WorkerPhysicalFootprint, find_worker_process_id, measure_worker_physical_footprint,
};

pub(super) const MODEL_ID: &str = crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID;
pub(super) const BENCHMARK_TIMEOUT: Duration = Duration::from_secs(115);
const READY_ATTEMPT_LIMIT: u8 = 70;

pub(super) struct StreamingMeasurement {
    pub(super) request_started_at: Instant,
    pub(super) first_output_at: Instant,
    pub(super) response_completed_at: Instant,
    pub(super) completion_token_count: u64,
    pub(super) cached_token_count: u64,
    pub(super) generated_text: String,
    pub(super) prefill_progress_samples: Vec<PrefillProgressSample>,
    pub(super) prompt_token_count: u64,
}

impl StreamingMeasurement {
    pub(super) fn prompt_duration(&self) -> Duration {
        self.first_output_at.duration_since(self.request_started_at)
    }

    pub(super) fn generation_duration(&self) -> Duration {
        self.response_completed_at
            .duration_since(self.first_output_at)
    }

    pub(super) fn total_duration(&self) -> Duration {
        self.response_completed_at
            .duration_since(self.request_started_at)
    }
}

#[derive(Clone, Debug)]
pub(super) struct PrefillProgressSample {
    pub(super) elapsed_millis: u64,
    pub(super) processed_tokens: u32,
    pub(super) total_tokens: u32,
}

impl PrefillProgressSample {
    pub(super) fn tokens_per_second(&self) -> f64 {
        if self.elapsed_millis == 0 {
            return 0.0;
        }
        self.processed_tokens as f64 / (self.elapsed_millis as f64 / 1_000.0)
    }
}

struct CompletedTokenCounts {
    completion_token_count: u64,
    cached_token_count: u64,
    prompt_token_count: u64,
}

#[derive(Clone, Copy)]
pub(super) struct PersistentPromptCacheDiskUsage {
    pub(super) block_count: usize,
    pub(super) byte_count: u64,
}

pub(super) struct WorkerPassMeasurement {
    pub(super) physical_footprint: WorkerPhysicalFootprint,
    pub(super) idle_worker_startup_duration: Duration,
    pub(super) streaming_measurement: StreamingMeasurement,
}

pub(super) async fn run_worker_pass(
    pass_label: &str,
    benchmark_started_at: Instant,
    worker_executable_path: PathBuf,
    worker_startup_configuration: WorkerStartupConfiguration,
    model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
    source_document: &str,
    persistent_prompt_cache_warmup_case: &PersistentPromptCacheWarmupCase,
    persistent_prompt_cache_directory: &Path,
) -> WorkerPassMeasurement {
    eprintln!("[persistent-prompt-cache-warmup] launching the {pass_label} worker");
    let worker_started_at = Instant::now();
    let performance_log_directory = worker_startup_configuration.logging_directory.clone();
    fs::create_dir_all(&performance_log_directory)
        .expect("the persistent prompt-cache performance log directory should be created");
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        Duration::from_secs(60),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the persistent prompt-cache performance log should open"),
        model_policy_catalog,
        worker_startup_configuration,
    )
    .await
    .expect("the supervisor should launch the measured worker");
    wait_until_idle(&worker_handle).await;
    let idle_worker_startup_duration = worker_started_at.elapsed();
    let worker_process_id = find_worker_process_id().await;

    let streaming_measurement = measure_worker_summarization(
        pass_label,
        benchmark_started_at,
        BENCHMARK_TIMEOUT,
        &worker_handle,
        source_document,
        persistent_prompt_cache_warmup_case,
        persistent_prompt_cache_directory,
    )
    .await;
    let physical_footprint = measure_worker_physical_footprint(worker_process_id)
        .await
        .expect("Apple footprint should measure worker memory");
    worker_handle
        .shutdown()
        .await
        .expect("the measured worker should terminate and be reaped");

    WorkerPassMeasurement {
        physical_footprint,
        idle_worker_startup_duration,
        streaming_measurement,
    }
}

pub(super) fn persistent_prompt_cache_enabled_worker_configuration(
    model_directory: &Path,
    persistent_prompt_cache_maximum_size_gb: u64,
) -> (
    tempfile::TempDir,
    PathBuf,
    PathBuf,
    Arc<HashMap<String, RuntimeModelPolicy>>,
    WorkerStartupConfiguration,
) {
    let production_worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the production inference-worker executable path");
    let isolated_worker_home =
        tempfile::tempdir().expect("the persistent prompt-cache worker home should be created");
    let configuration_directory = isolated_worker_home.path().join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the persistent prompt-cache configuration directory should be created");
    let persistent_prompt_cache_directory = configuration_directory.join("cache");
    fs::create_dir_all(&persistent_prompt_cache_directory)
        .expect("the persistent prompt-cache directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "prompt_cache_max_size_gb": persistent_prompt_cache_maximum_size_gb,
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the persistent prompt-cache configuration should serialize"),
    )
    .expect("the persistent prompt-cache configuration should be written");

    let worker_runtime_config = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_worker_home.path().to_path_buf(),
        PathBuf::from(&production_worker_executable_path),
    )
    .load()
    .expect("the persistent prompt-cache worker configuration should resolve");
    let model_policy_catalog = worker_runtime_config.model_policy_catalog.clone();
    (
        isolated_worker_home,
        persistent_prompt_cache_directory,
        PathBuf::from(production_worker_executable_path),
        model_policy_catalog,
        worker_runtime_config.worker_startup_configuration(),
    )
}

pub(super) fn static_source_document(source_document: &str) -> String {
    source_document
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn persistent_prompt_cache_disk_usage(
    persistent_prompt_cache_directory: &Path,
) -> PersistentPromptCacheDiskUsage {
    let (block_count, byte_count) =
        persistent_prompt_cache_safetensors_disk_usage(persistent_prompt_cache_directory);
    PersistentPromptCacheDiskUsage {
        block_count,
        byte_count,
    }
}

fn persistent_prompt_cache_safetensors_disk_usage(
    persistent_prompt_cache_directory: &Path,
) -> (usize, u64) {
    let mut kv_block_file_count = 0_usize;
    let mut safetensors_byte_count = 0_u64;
    let mut pending_directories = vec![persistent_prompt_cache_directory.to_path_buf()];
    while let Some(directory_path) = pending_directories.pop() {
        for directory_entry in fs::read_dir(&directory_path)
            .expect("the persistent prompt-cache directory should be readable")
            .flatten()
        {
            let cache_entry_path = directory_entry.path();
            if cache_entry_path.is_dir() {
                pending_directories.push(cache_entry_path);
                continue;
            }
            if cache_entry_path.is_file()
                && cache_entry_path
                    .extension()
                    .is_some_and(|extension| extension == "safetensors")
            {
                if cache_entry_path
                    .file_name()
                    .is_some_and(|file_name| file_name == "sequence.safetensors")
                {
                    kv_block_file_count += 1;
                }
                safetensors_byte_count += directory_entry
                    .metadata()
                    .expect("persistent prompt-cache safetensors metadata should be readable")
                    .len();
            }
        }
    }
    (kv_block_file_count, safetensors_byte_count)
}

async fn wait_until_idle(worker_handle: &WorkerHandle) {
    for readiness_attempt in 1..=READY_ATTEMPT_LIMIT {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready
            && worker_health_snapshot.ready_model_id.is_none()
        {
            eprintln!(
                "[persistent-prompt-cache-warmup] idle worker ready after {readiness_attempt} attempts"
            );
            return;
        }
        let remaining_seconds = u16::from(READY_ATTEMPT_LIMIT - readiness_attempt);
        eprintln!(
            "[persistent-prompt-cache-warmup] startup attempt {readiness_attempt}/{READY_ATTEMPT_LIMIT}, ETA <= {remaining_seconds}s"
        );
        sleep(Duration::from_secs(1)).await;
    }
    panic!(
        "the real worker did not become idle before the persistent prompt-cache warmup deadline"
    );
}

async fn measure_worker_summarization(
    pass_label: &str,
    benchmark_started_at: Instant,
    benchmark_timeout: Duration,
    worker_handle: &WorkerHandle,
    source_document: &str,
    persistent_prompt_cache_warmup_case: &PersistentPromptCacheWarmupCase,
    persistent_prompt_cache_directory: &Path,
) -> StreamingMeasurement {
    let request_started_at = Instant::now();
    let mut stream_receiver = worker_handle
        .start_chat_generation(ChatGenerationCommand {
            request_id: RequestId::new(1),
            model: MODEL_ID.to_owned(),
            messages: vec![ChatMessage::User {
                content: format!(
                    "{}\n\n{source_document}",
                    persistent_prompt_cache_warmup_case.instruction
                ),
                images: Vec::new(),
            }],
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: ChatGenerationSettings {
                max_output_tokens: persistent_prompt_cache_warmup_case.maximum_output_tokens,
                temperature_thousandths: None,
                top_p_thousandths: None,
                seed: None,
                thinking_budget: Some(256),
            },
            qwen_thinking_channel_seed: None,
        })
        .await
        .expect("the worker should start the persistent prompt-cache warmup generation request");
    let mut first_output_at: Option<Instant> = None;
    let mut generated_text = String::new();
    let mut completed_token_counts: Option<CompletedTokenCounts> = None;
    let mut status_poll_interval = interval(Duration::from_millis(250));
    status_poll_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    status_poll_interval.tick().await;
    let mut next_progress_log_at = Duration::from_secs(10);
    let mut latest_sampled_prefill_processed_tokens = 0_u32;
    let mut prefill_progress_samples = Vec::new();
    eprintln!(
        "[persistent-prompt-cache-warmup:{pass_label}] request accepted, waiting for first output"
    );
    loop {
        let stream_event = tokio::select! {
            stream_event = stream_receiver.recv() => stream_event,
            _ = status_poll_interval.tick() => {
                sample_prefill_progress(
                    pass_label,
                    worker_handle,
                    &mut latest_sampled_prefill_processed_tokens,
                    &mut prefill_progress_samples,
                );
                let elapsed = benchmark_started_at.elapsed();
                if elapsed >= next_progress_log_at {
                    let remaining = benchmark_timeout.saturating_sub(elapsed);
                    let saved_block_count = persistent_prompt_cache_disk_usage(
                        persistent_prompt_cache_directory,
                    )
                    .block_count;
                    eprintln!(
                        "[persistent-prompt-cache-warmup:{pass_label}] request active; SSD blocks={saved_block_count}, benchmark elapsed {:.0}s, deadline <= {:.0}s",
                        elapsed.as_secs_f64(),
                        remaining.as_secs_f64(),
                    );
                    next_progress_log_at += Duration::from_secs(10);
                }
                continue;
            }
        };
        let Some(stream_event) = stream_event else {
            break;
        };
        match stream_event {
            ChatGenerationStreamEvent::ReasoningFragment(text)
            | ChatGenerationStreamEvent::TextFragment(text) => {
                record_first_output(pass_label, &mut first_output_at, "generated output");
                generated_text.push_str(&text);
            }
            ChatGenerationStreamEvent::ToolCall {
                function_name,
                arguments_json,
                ..
            } => {
                record_first_output(pass_label, &mut first_output_at, "generated tool call");
                generated_text.push_str(&function_name);
                generated_text.push_str(&arguments_json);
            }
            ChatGenerationStreamEvent::PrefillProgress { .. } => {}
            ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                generated_token_count,
                cached_token_count,
                ..
            } => {
                completed_token_counts = Some(CompletedTokenCounts {
                    completion_token_count: u64::from(generated_token_count),
                    cached_token_count: u64::from(cached_token_count),
                    prompt_token_count: u64::from(prompt_token_count),
                });
                break;
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                panic!(
                    "the worker failed the persistent prompt-cache warmup generation request: {reason:?}"
                );
            }
            ChatGenerationStreamEvent::Error(error_code) => {
                panic!("the worker stream reported an error: {error_code:?}");
            }
        }
    }
    let response_completed_at = Instant::now();
    let completed_token_counts = completed_token_counts
        .expect("the terminal worker event should report prompt, completion, and cached tokens");
    StreamingMeasurement {
        request_started_at,
        first_output_at: first_output_at.unwrap_or(response_completed_at),
        response_completed_at,
        completion_token_count: completed_token_counts.completion_token_count,
        cached_token_count: completed_token_counts.cached_token_count,
        generated_text,
        prefill_progress_samples,
        prompt_token_count: completed_token_counts.prompt_token_count,
    }
}

fn sample_prefill_progress(
    pass_label: &str,
    worker_handle: &WorkerHandle,
    latest_sampled_prefill_processed_tokens: &mut u32,
    prefill_progress_samples: &mut Vec<PrefillProgressSample>,
) {
    let worker_health_snapshot = worker_handle.worker_health_snapshot();
    let Some(active_request_progress) = worker_health_snapshot.active_request_progress else {
        return;
    };
    let ActiveRequestProgress::Prefill {
        processed_tokens,
        total_tokens,
        elapsed_millis,
        ..
    } = active_request_progress
    else {
        return;
    };
    if processed_tokens <= *latest_sampled_prefill_processed_tokens {
        return;
    }
    let prefill_progress_sample = PrefillProgressSample {
        elapsed_millis,
        processed_tokens,
        total_tokens,
    };
    eprintln!(
        "[persistent-prompt-cache-warmup:{pass_label}] prefill progress {}/{} tokens in {}ms ({:.1} tok/s)",
        prefill_progress_sample.processed_tokens,
        prefill_progress_sample.total_tokens,
        prefill_progress_sample.elapsed_millis,
        prefill_progress_sample.tokens_per_second(),
    );
    *latest_sampled_prefill_processed_tokens = prefill_progress_sample.processed_tokens;
    prefill_progress_samples.push(prefill_progress_sample);
}

fn record_first_output(
    pass_label: &str,
    first_output_at: &mut Option<Instant>,
    first_output_description: &str,
) {
    if first_output_at.is_some() {
        return;
    }
    *first_output_at = Some(Instant::now());
    eprintln!(
        "[persistent-prompt-cache-warmup:{pass_label}] first {first_output_description} received"
    );
}
