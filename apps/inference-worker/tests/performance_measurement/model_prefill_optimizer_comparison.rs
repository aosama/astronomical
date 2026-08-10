use std::{fs, path::PathBuf, time::Duration};

use astronomical_inference_worker::worker_startup::sample_iogpu_wired_limit_bytes;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_supervisor::{ChatGenerationExecutor, ChatGenerationStreamEvent};
use tokio::time::{Instant, timeout};

use super::model_prefill_benchmark_report::{
    PrefillMeasurementAccumulator, TypedOutputEventDigest, build_prefill_benchmark_report,
};
use super::model_prefill_optimizer_candidate_observation::assert_cumulative_latency_optimizer_evidence;
use super::model_prefill_qualification_worker::{
    PREFILL_QUALIFICATION_MAXIMUM_OUTPUT_TOKENS, PREFILL_QUALIFICATION_MODEL_ID,
    build_prefill_qualification_prompt, configured_prefill_qualification_model_directory,
    expert_memory_mode_label, launch_prepared_prefill_qualification_worker,
    observed_final_expert_memory_mode, prefill_memory_limit_validation_error,
    prepare_prefill_qualification_worker, required_prefill_qualification_u32,
    wait_until_prefill_qualification_worker_is_idle, warm_prefill_qualification_worker,
};
use super::model_process_metrics::{
    WorkerPhysicalFootprint, find_worker_process_id, measure_worker_physical_footprint,
};
const BENCHMARK_TIMEOUT: Duration = Duration::from_secs(115);
const FIXED_PREFILL_CHUNCK_TOKENS: u32 = 2_048;
const TARGET_PROMPT_TOKENS: usize = 90_000;
const WARMUP_PROMPT_TOKENS: usize = 1_024;
const WARM_REQUEST_MAXIMUM_OUTPUT_TOKENS: u16 = 1;

#[derive(Clone, Copy, Debug)]
enum PrefillBenchmarkMode {
    Fixed2_048,
    FixedCandidate(u32),
    Optimizer,
}

struct PrefillBenchmarkOutcome {
    typed_output_event_digest: String,
    prefill_memory_validation_error: Option<String>,
}

impl PrefillBenchmarkOutcome {
    fn require_prefill_memory_within_configured_limits(&self) {
        if let Some(prefill_memory_validation_error) = &self.prefill_memory_validation_error {
            panic!("{prefill_memory_validation_error}");
        }
    }
}

impl PrefillBenchmarkMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Fixed2_048 => "fixed_2048",
            Self::FixedCandidate(_) => "fixed_candidate",
            Self::Optimizer => "optimizer",
        }
    }

    const fn fixed_prefill_chunck_tokens(self) -> Option<u32> {
        match self {
            Self::Fixed2_048 => Some(FIXED_PREFILL_CHUNCK_TOKENS),
            Self::FixedCandidate(prefill_chunck_tokens) => Some(prefill_chunck_tokens),
            Self::Optimizer => None,
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads Ornith and measures exactly 90,000 prompt tokens with fixed 2,048-token chunks"]
async fn should_measure_ninety_thousand_prompt_tokens_with_fixed_2048_prefill_chuncks() {
    timeout(
        BENCHMARK_TIMEOUT,
        run_prefill_benchmark_with_peak_gate(
            PrefillBenchmarkMode::Fixed2_048,
            TARGET_PROMPT_TOKENS,
            None,
        ),
    )
    .await
    .expect("the fixed 90K prefill benchmark must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads Ornith and measures exactly 90,000 prompt tokens with the optimizer"]
async fn should_measure_ninety_thousand_prompt_tokens_with_the_optimizer() {
    timeout(
        BENCHMARK_TIMEOUT,
        run_prefill_benchmark_with_peak_gate(
            PrefillBenchmarkMode::Optimizer,
            TARGET_PROMPT_TOKENS,
            None,
        ),
    )
    .await
    .expect("the optimized 90K prefill benchmark must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "runs one selected fixed-chunk prefill qualification cell"]
async fn should_measure_one_selected_prefill_candidate_qualification_cell() {
    let prefill_chunck_tokens = required_prefill_qualification_u32(
        "ASTRONOMICAL_PREFILL_QUALIFICATION_CHUNCK_TOKENS",
        &[2_048, 4_096, 8_192],
    );
    let prompt_token_count = required_prefill_qualification_u32(
        "ASTRONOMICAL_PREFILL_QUALIFICATION_PROMPT_TOKENS",
        &[1_024, 4_097, 8_193],
    ) as usize;
    let maximum_mlx_memory_gb = required_prefill_qualification_u32(
        "ASTRONOMICAL_PREFILL_QUALIFICATION_MAXIMUM_MLX_MEMORY_GB",
        &[10, 30],
    ) as u64;
    timeout(
        BENCHMARK_TIMEOUT,
        run_prefill_benchmark_with_peak_gate(
            PrefillBenchmarkMode::FixedCandidate(prefill_chunck_tokens),
            prompt_token_count,
            Some(maximum_mlx_memory_gb),
        ),
    )
    .await
    .expect("the selected prefill qualification cell must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "compares one fixed candidate with fixed 2048 for worker-output parity"]
async fn should_preserve_typed_output_event_digest_against_fixed_2048() {
    let candidate_prefill_chunck_tokens = required_prefill_qualification_u32(
        "ASTRONOMICAL_PREFILL_PARITY_CHUNCK_TOKENS",
        &[4_096, 8_192],
    );
    let prompt_token_count = required_prefill_qualification_u32(
        "ASTRONOMICAL_PREFILL_PARITY_PROMPT_TOKENS",
        &[1_024, 4_097, 8_193],
    ) as usize;
    let maximum_mlx_memory_gb = required_prefill_qualification_u32(
        "ASTRONOMICAL_PREFILL_PARITY_MAXIMUM_MLX_MEMORY_GB",
        &[10, 30],
    ) as u64;
    let (baseline_prefill_benchmark_outcome, candidate_prefill_benchmark_outcome) =
        timeout(BENCHMARK_TIMEOUT, async {
            let baseline_prefill_benchmark_outcome = run_prefill_benchmark(
                PrefillBenchmarkMode::Fixed2_048,
                prompt_token_count,
                Some(maximum_mlx_memory_gb),
            )
            .await;
            let candidate_prefill_benchmark_outcome = run_prefill_benchmark(
                PrefillBenchmarkMode::FixedCandidate(candidate_prefill_chunck_tokens),
                prompt_token_count,
                Some(maximum_mlx_memory_gb),
            )
            .await;
            (
                baseline_prefill_benchmark_outcome,
                candidate_prefill_benchmark_outcome,
            )
        })
        .await
        .expect("the paired prefill parity qualification must finish within 115 seconds");

    assert_eq!(
        candidate_prefill_benchmark_outcome.typed_output_event_digest,
        baseline_prefill_benchmark_outcome.typed_output_event_digest,
        "fixed candidate changed the worker output-event sequence"
    );
    baseline_prefill_benchmark_outcome.require_prefill_memory_within_configured_limits();
    candidate_prefill_benchmark_outcome.require_prefill_memory_within_configured_limits();
}

async fn run_prefill_benchmark_with_peak_gate(
    benchmark_mode: PrefillBenchmarkMode,
    target_prompt_tokens: usize,
    maximum_mlx_memory_gb: Option<u64>,
) {
    let prefill_benchmark_outcome =
        run_prefill_benchmark(benchmark_mode, target_prompt_tokens, maximum_mlx_memory_gb).await;
    prefill_benchmark_outcome.require_prefill_memory_within_configured_limits();
}

async fn run_prefill_benchmark(
    benchmark_mode: PrefillBenchmarkMode,
    target_prompt_tokens: usize,
    maximum_mlx_memory_gb: Option<u64>,
) -> PrefillBenchmarkOutcome {
    let benchmark_label = benchmark_mode.label();
    let configured_model_directory = configured_prefill_qualification_model_directory();
    let exact_prompt_content =
        build_prefill_qualification_prompt(&configured_model_directory, target_prompt_tokens);
    let warmup_prompt_content =
        build_prefill_qualification_prompt(&configured_model_directory, WARMUP_PROMPT_TOKENS);
    let prepared_prefill_qualification_worker = prepare_prefill_qualification_worker(
        &configured_model_directory,
        benchmark_mode.fixed_prefill_chunck_tokens(),
        maximum_mlx_memory_gb,
    );
    let maximum_gpu_wired_memory_bytes = sample_iogpu_wired_limit_bytes()
        .await
        .expect("the GPU wired-memory limit should be available");

    eprintln!("[prefill-optimizer:{benchmark_label}] launching worker; ETA <= 70s");
    let worker_started_at = Instant::now();
    let worker_handle = launch_prepared_prefill_qualification_worker(
        &prepared_prefill_qualification_worker,
        &configured_model_directory,
    )
    .await;
    wait_until_prefill_qualification_worker_is_idle(&worker_handle, benchmark_label).await;
    let worker_startup_seconds = worker_started_at.elapsed().as_secs_f64();
    let worker_process_id = find_worker_process_id().await;
    let idle_worker_footprint = measure_worker_physical_footprint(worker_process_id)
        .await
        .expect("Apple footprint should measure ready worker memory");
    warm_prefill_qualification_worker(
        &worker_handle,
        benchmark_command(
            format!("Warmup-only cache namespace.\n\n{warmup_prompt_content}"),
            WARM_REQUEST_MAXIMUM_OUTPUT_TOKENS,
        ),
        WARM_REQUEST_MAXIMUM_OUTPUT_TOKENS,
    )
    .await;

    let request_started_at = Instant::now();
    let mut stream_receiver = worker_handle
        .start_chat_generation(benchmark_command(
            exact_prompt_content,
            PREFILL_QUALIFICATION_MAXIMUM_OUTPUT_TOKENS,
        ))
        .await
        .expect("the benchmark request should start");
    let mut first_output_at: Option<Instant> = None;
    let mut first_output_footprint: Option<WorkerPhysicalFootprint> = None;
    let mut prefill_measurements = PrefillMeasurementAccumulator::new();
    let mut completed_measurement: Option<(u32, u16, u32, String)> = None;
    let mut typed_output_event_digest = TypedOutputEventDigest::new();
    let mut next_progress_token_count = 5_000_u32;

    while let Some(stream_event) = stream_receiver.recv().await {
        match stream_event {
            ChatGenerationStreamEvent::PrefillProgress {
                processed_tokens,
                total_tokens,
                elapsed_millis,
                forward_prefill_chunck_elapsed_millis,
                completed_prefill_chunck_tokens,
                mlx_active_memory_bytes,
                mlx_allocator_cache_memory_bytes,
                mlx_peak_memory_bytes,
            } => {
                assert_eq!(total_tokens, target_prompt_tokens as u32);
                prefill_measurements.record(
                    processed_tokens,
                    elapsed_millis,
                    forward_prefill_chunck_elapsed_millis,
                    completed_prefill_chunck_tokens,
                    mlx_active_memory_bytes,
                    mlx_allocator_cache_memory_bytes,
                    mlx_peak_memory_bytes,
                );
                if processed_tokens >= next_progress_token_count {
                    let latest_prefill_chunck_measurement = prefill_measurements
                        .chuncks()
                        .last()
                        .expect("completed prefill progress should have one chunk measurement");
                    eprintln!(
                        "[prefill-optimizer:{benchmark_label}] processed={processed_tokens}/{total_tokens} native_elapsed_ms={elapsed_millis} latest_forward_ms={} latest_allocator_clear_and_telemetry_ms={} mlx_active_bytes={} mlx_allocator_cache_bytes={} mlx_peak_bytes={}",
                        latest_prefill_chunck_measurement.forward_prefill_chunck_elapsed_millis,
                        latest_prefill_chunck_measurement
                            .elapsed_millis
                            .saturating_sub(
                                latest_prefill_chunck_measurement
                                    .forward_prefill_chunck_elapsed_millis,
                            ),
                        latest_prefill_chunck_measurement.mlx_active_memory_bytes,
                        latest_prefill_chunck_measurement.mlx_allocator_cache_memory_bytes,
                        latest_prefill_chunck_measurement.mlx_peak_memory_bytes,
                    );
                    next_progress_token_count = next_progress_token_count.saturating_add(5_000);
                }
            }
            ChatGenerationStreamEvent::ReasoningFragment(reasoning_fragment) => {
                typed_output_event_digest.record_reasoning_fragment(&reasoning_fragment);
                record_first_output_footprint(
                    &mut first_output_at,
                    &mut first_output_footprint,
                    worker_process_id,
                    benchmark_label,
                )
                .await;
            }
            ChatGenerationStreamEvent::TextFragment(text_fragment) => {
                typed_output_event_digest.record_text_fragment(&text_fragment);
                record_first_output_footprint(
                    &mut first_output_at,
                    &mut first_output_footprint,
                    worker_process_id,
                    benchmark_label,
                )
                .await;
            }
            ChatGenerationStreamEvent::ToolCall {
                tool_call_index,
                function_name,
                arguments_json,
            } => {
                typed_output_event_digest.record_tool_call(
                    tool_call_index,
                    &function_name,
                    &arguments_json,
                );
                record_first_output_footprint(
                    &mut first_output_at,
                    &mut first_output_footprint,
                    worker_process_id,
                    benchmark_label,
                )
                .await;
            }
            ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                generated_token_count,
                cached_token_count,
                reason,
                ..
            } => {
                completed_measurement = Some((
                    prompt_token_count,
                    generated_token_count,
                    cached_token_count,
                    format!("{reason:?}"),
                ));
                break;
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                panic!("the benchmark worker failed the request: {reason:?}");
            }
            ChatGenerationStreamEvent::Error(error_code) => {
                panic!("the benchmark worker stream failed: {error_code:?}");
            }
        }
    }

    let response_completed_at = Instant::now();
    let completed_footprint = measure_worker_physical_footprint(worker_process_id)
        .await
        .expect("Apple footprint should measure completed worker memory");

    let (prompt_token_count, generated_token_count, cached_token_count, completion_reason) =
        completed_measurement.expect("the benchmark should receive a completion event");
    assert_eq!(prompt_token_count, target_prompt_tokens as u32);
    assert_eq!(
        cached_token_count, 0,
        "the benchmark must perform cold prefill"
    );
    assert_eq!(
        generated_token_count, PREFILL_QUALIFICATION_MAXIMUM_OUTPUT_TOKENS,
        "the benchmark must include the representative 512 generated tokens"
    );
    if matches!(benchmark_mode, PrefillBenchmarkMode::Optimizer) {
        assert_cumulative_latency_optimizer_evidence(&prefill_measurements);
    }
    let prefill_memory_validation_error =
        prefill_memory_limit_validation_error(&prefill_measurements, maximum_mlx_memory_gb);
    let final_expert_memory_mode =
        observed_final_expert_memory_mode(&worker_handle, benchmark_label).await;
    let first_output_at = first_output_at.unwrap_or(response_completed_at);
    let first_output_footprint = first_output_footprint.unwrap_or(idle_worker_footprint);
    let typed_output_event_digest = typed_output_event_digest.finish();
    let benchmark_report = build_prefill_benchmark_report(
        benchmark_mode.label(),
        benchmark_mode.fixed_prefill_chunck_tokens(),
        target_prompt_tokens,
        PREFILL_QUALIFICATION_MAXIMUM_OUTPUT_TOKENS,
        worker_startup_seconds,
        request_started_at,
        first_output_at,
        response_completed_at,
        generated_token_count,
        completion_reason,
        typed_output_event_digest.clone(),
        expert_memory_mode_label(final_expert_memory_mode),
        prepared_prefill_qualification_worker.optimizer_state_loaded_before_run,
        maximum_gpu_wired_memory_bytes,
        idle_worker_footprint,
        first_output_footprint,
        completed_footprint,
        &prefill_measurements,
    );
    eprintln!("[prefill-optimizer-report] {benchmark_report}");
    if let Some(report_path) = benchmark_report_path() {
        fs::write(
            report_path,
            serde_json::to_vec_pretty(&benchmark_report)
                .expect("the benchmark report should serialize"),
        )
        .expect("the benchmark report should be written");
    }
    worker_handle
        .shutdown()
        .await
        .expect("the measured worker should terminate and be reaped");
    PrefillBenchmarkOutcome {
        typed_output_event_digest,
        prefill_memory_validation_error,
    }
}

pub(super) fn benchmark_command(
    exact_prompt_content: String,
    maximum_output_tokens: u16,
) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(1),
        model: PREFILL_QUALIFICATION_MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: exact_prompt_content,
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: maximum_output_tokens,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: Some(256),
        },
    }
}

async fn record_first_output_footprint(
    first_output_at: &mut Option<Instant>,
    first_output_footprint: &mut Option<WorkerPhysicalFootprint>,
    worker_process_id: u32,
    benchmark_label: &str,
) {
    if first_output_at.is_some() {
        return;
    }
    *first_output_at = Some(Instant::now());
    *first_output_footprint = Some(
        measure_worker_physical_footprint(worker_process_id)
            .await
            .expect("Apple footprint should measure first-output memory"),
    );
    eprintln!("[prefill-optimizer:{benchmark_label}] first output received");
}

fn benchmark_report_path() -> Option<PathBuf> {
    std::env::var_os("ASTRONOMICAL_BENCHMARK_REPORT_PATH").map(PathBuf::from)
}

#[test]
fn should_target_exactly_ninety_thousand_rendered_prompt_tokens() {
    assert_eq!(TARGET_PROMPT_TOKENS, 90_000);
    assert!(WARMUP_PROMPT_TOKENS < FIXED_PREFILL_CHUNCK_TOKENS as usize);
}
