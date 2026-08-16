//! Focused generation-speed measurement for Qwen3.8-27B-MTPLX-4bit at three MTP depths.
//!
//! Runs a target-only baseline (MTP disabled), then depth 1 and depth 3 with MTP enabled,
//! printing a compact results table. Each cell includes a warmup pass and a measured 1 024-token
//! generation using the Romeo and Juliet fixture.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, MtpRuntimeState,
    RequestId,
};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, PerformanceAttributionLog,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5InferenceRequest,
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
};
use astronomical_runtime_integration::MlxRuntime;

const MODEL_ID: &str = "Qwen3.8-27B-MTPLX-4bit";
const REPRESENTATIVE_INPUT_TOKEN_COUNT: usize = 1_024;
const REPRESENTATIVE_OUTPUT_TOKEN_COUNT: u16 = 1_024;
const WARMUP_OUTPUT_TOKEN_COUNT: u16 = 32;
const SOURCE_FILE_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_MTP_SPEED_SOURCE_FILE";
const ATTRIBUTION_MODE_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_MTP_SPEED_ATTRIBUTION";
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

// ── Public test entry points ──────────────────────────────────────────────────

#[tokio::test]
#[ignore = "measures generation speed on the GPU; run with --ignored --features direct-mlx"]
async fn should_measure_qwen3_8_generation_speed_target_only() {
    run_speed_cell("target_only", false, None).await;
}

#[tokio::test]
#[ignore = "measures generation speed on the GPU; run with --ignored --features direct-mlx"]
async fn should_measure_qwen3_8_generation_speed_depth_one() {
    run_speed_cell("depth_one", true, Some(1)).await;
}

#[tokio::test]
#[ignore = "measures generation speed on the GPU; run with --ignored --features direct-mlx"]
async fn should_measure_qwen3_8_generation_speed_depth_three() {
    run_speed_cell("depth_three", true, Some(3)).await;
}

// ── Core measurement logic ───────────────────────────────────────────────────

async fn run_speed_cell(cell_name: &str, mtp_enabled: bool, draft_depth: Option<u8>) {
    tokio::time::timeout(Duration::from_secs(120), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
        let prompt = representative_prompt(&model_directory);

        eprintln!(
            "[mtp-speed] cell={cell_name} depth={draft_depth:?} status=loading"
        );

        let mut loaded = load_engine(&model_directory, mtp_enabled, draft_depth).await;
        let attribution_enabled = std::env::var(ATTRIBUTION_MODE_ENVIRONMENT_VARIABLE)
            .map_or(true, |mode| mode != "disabled");
        let warmup_attribution = if attribution_enabled {
            PerformanceAttribution::enabled()
        } else {
            PerformanceAttribution::disabled()
        };

        eprintln!(
            "[mtp-speed] cell={cell_name} depth={draft_depth:?} status=warmup"
        );

        // Warmup: generate a short sequence to stabilize caches and paging.
        loaded
            .engine
            .reset_mlx_peak_memory_for_tests()
            .await
            .expect("warmup should start with a fresh MLX peak");

        let warmup_id = RequestId::new(80_000);
        let _warmup = run_generation(
            &mut loaded.engine,
            warmup_id,
            &prompt,
            WARMUP_OUTPUT_TOKEN_COUNT,
            warmup_attribution,
        )
        .await;

        eprintln!(
            "[mtp-speed] cell={cell_name} depth={draft_depth:?} status=measuring"
        );

        // Measured generation: 1 024 output tokens.
        loaded
            .engine
            .reset_mlx_peak_memory_for_tests()
            .await
            .expect("the measured request should start with a fresh MLX peak");

        let measured_id = RequestId::new(80_001);
        let measured_attribution = if attribution_enabled {
            PerformanceAttribution::enabled()
        } else {
            PerformanceAttribution::disabled()
        };
        let measured = run_generation(
            &mut loaded.engine,
            measured_id,
            &prompt,
            REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
            measured_attribution,
        )
        .await;

        // Attribution log counters for MTP diagnostics.
        let (operational_fallback_count, proposed_draft_count, effective_depth_total) =
            if attribution_enabled {
                let report =
                    generation_report(&loaded.performance_attribution_log_path, measured_id);
                print_operation_report(&report, cell_name);
                (
                    counter_amount(&report, "mtp_operational_fallback_count"),
                    counter_amount(&report, "mtp_proposed_draft_count"),
                    counter_amount(&report, "mtp_effective_depth_total"),
                )
            } else {
                (0, 0, 0)
            };

        // Token fingerprint for reproducibility checks.
        let mut hasher = DefaultHasher::new();
        measured.generated_token_ids.hash(&mut hasher);
        let fingerprint = hasher.finish();

        let output_token_count = measured.generated_token_ids.len();
        let tok_s = (output_token_count.saturating_sub(1) as f64)
            / measured.generation_elapsed_seconds.max(f64::EPSILON);

        let active_mib = measured.maximum_active_mlx_memory_bytes as f64 / (1024.0 * 1024.0);
        let peak_mib = measured.maximum_peak_mlx_memory_bytes as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[mtp-speed] cell={} depth={:?} status=done output_tokens={} tok_s={:.2} total_s={:.3} gen_s={:.3} active_MiB={:.1} peak_MiB={:.1} fallbacks={} proposed={} accepted_depth={} fingerprint={:#010x}",
            cell_name,
            draft_depth,
            output_token_count,
            tok_s,
            measured.total_request_elapsed_seconds,
            measured.generation_elapsed_seconds,
            active_mib,
            peak_mib,
            operational_fallback_count,
            proposed_draft_count,
            effective_depth_total,
            fingerprint,
        );

        drop(loaded);
        MlxRuntime::initialize(
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await,
        )
        .expect("runtime should re-enter after the measurement")
        .clear_allocator_cache()
        .expect("measurement should release reclaimable allocator storage");
    })
    .await
    .unwrap_or_else(|_| panic!("the {cell_name} measurement should finish within 120 seconds"));
}

// ── Engine construction and loading ───────────────────────────────────────────

struct LoadedEngine {
    engine: Qwen3_5Engine,
    performance_attribution_log_path: std::path::PathBuf,
    _temporary_log_directory: tempfile::TempDir,
}

async fn load_engine(
    model_directory: &std::path::Path,
    mtp_enabled: bool,
    draft_depth: Option<u8>,
) -> LoadedEngine {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .unwrap_or_else(|e| panic!("artifact validation should succeed: {e}"));

    let think_end_token_id = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the Qwen3.8 tokenizer should load")
        .think_end_token_id();

    let memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;

    let temporary_log_directory =
        tempfile::tempdir().expect("the speed measurement should create a temporary log directory");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    let performance_attribution_log =
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("speed measurement should open its attribution log");

    let mut engine =
        Qwen3_5Engine::new_with_runtime_chunking_speculative_prefill_mtp_depth_and_performance_attribution(
            validated_artifact,
            memory_limits.active_memory_limit_bytes(),
            memory_limits.allocator_cache_memory_limit_bytes(),
            None,
            Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(1_024)
                .expect("the fixed 1 024-token chunk should be valid"),
            think_end_token_id,
            model_directory.to_path_buf(),
            crate::common::standard_worker_chunking_configuration(),
            true,
            mtp_enabled,
            draft_depth,
            crate::common::disabled_worker_speculative_prefill_configuration(),
            PerformanceAttribution::disabled(),
            performance_attribution_log,
        )
        .expect("the Qwen3.8 engine configuration should be valid");

    let load_result = engine.load().await.expect("the Qwen3.8 engine should load");
    assert_eq!(
        load_result.mtp_runtime_state(),
        if mtp_enabled {
            MtpRuntimeState::Active
        } else {
            MtpRuntimeState::Disabled
        },
        "MTP runtime state should match the requested configuration"
    );
    if mtp_enabled {
        assert_eq!(
            load_result
                .mtp_depth_status()
                .effective_execution_draft_depth,
            draft_depth.or(Some(1)),
            "the effective MTP depth should match the requested depth"
        );
    }

    LoadedEngine {
        engine,
        performance_attribution_log_path,
        _temporary_log_directory: temporary_log_directory,
    }
}

// ── Prompt construction ──────────────────────────────────────────────────────

struct RepresentativePrompt {
    prompt_token_ids: Vec<u32>,
    image_pad_token_id: u32,
}

fn representative_prompt(model_directory: &std::path::Path) -> RepresentativePrompt {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the Qwen3.8 artifact should validate before tokenization");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the Qwen3.8 tokenizer should load");
    let representative_source = representative_source();

    let command = ChatGenerationCommand {
        request_id: RequestId::new(80_100),
        model: MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: format!(
                "Explain this source as a factual technical briefing of at least seven hundred words.\n\nSource material:\n{representative_source}"
            ),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: Some(256),
        },
    };

    let prepared = tokenizer
        .prepare_chat(&command, false)
        .expect("the representative Romeo and Juliet prompt should prepare");
    let complete_prompt = prepared.input_token_ids();
    let assistant_suffix_start = complete_prompt
        .iter()
        .rposition(|token_id| *token_id == tokenizer.im_end_token_id())
        .expect("the prepared prompt should contain the closing user marker");
    let assistant_suffix = &complete_prompt[assistant_suffix_start..];
    let prefix_length = REPRESENTATIVE_INPUT_TOKEN_COUNT
        .checked_sub(assistant_suffix.len())
        .expect("the assistant suffix should fit the representative prompt");
    assert!(prefix_length < assistant_suffix_start);
    let mut prompt_token_ids = complete_prompt[..prefix_length].to_vec();
    prompt_token_ids.extend_from_slice(assistant_suffix);
    assert_eq!(prompt_token_ids.len(), REPRESENTATIVE_INPUT_TOKEN_COUNT);

    RepresentativePrompt {
        prompt_token_ids,
        image_pad_token_id: tokenizer.image_pad_token_id(),
    }
}

fn representative_source() -> String {
    match std::env::var_os(SOURCE_FILE_ENVIRONMENT_VARIABLE) {
        Some(source_file) => std::fs::read_to_string(source_file)
            .expect("the configured MTP speed source file should be readable"),
        None => ROMEO_AND_JULIET_SOURCE.to_owned(),
    }
}

// ── Generation loop ──────────────────────────────────────────────────────────

struct GenerationMeasurement {
    generated_token_ids: Vec<u32>,
    generation_elapsed_seconds: f64,
    total_request_elapsed_seconds: f64,
    maximum_active_mlx_memory_bytes: u64,
    maximum_peak_mlx_memory_bytes: u64,
}

async fn run_generation(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    prompt: &RepresentativePrompt,
    output_token_count: u16,
    performance_attribution: PerformanceAttribution,
) -> GenerationMeasurement {
    // End-to-end timing starts before request admission.
    let request_started_at = Instant::now();
    engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                prompt.prompt_token_ids.clone(),
                output_token_count,
            )
            .with_image_pad_token_id(prompt.image_pad_token_id)
            .with_performance_attribution(performance_attribution),
        )
        .await
        .expect("the representative request should start");

    let mut generated_token_ids = Vec::with_capacity(output_token_count as usize);
    let mut first_generated_token_at = None;
    let mut maximum_active_mlx_memory_bytes = 0_u64;
    let mut maximum_peak_mlx_memory_bytes = 0_u64;

    loop {
        match engine
            .decode_next_token(request_id)
            .await
            .expect("the representative request should advance")
        {
            GeneratedToken::TokenId {
                token_id,
                mlx_memory_telemetry,
                generation_finalization,
                ..
            } => {
                let generated_at = Instant::now();
                let first_generated_at = *first_generated_token_at.get_or_insert(generated_at);
                generated_token_ids.push(token_id);

                if let Some(telemetry) = mlx_memory_telemetry {
                    maximum_active_mlx_memory_bytes =
                        maximum_active_mlx_memory_bytes.max(telemetry.active_memory_bytes);
                    maximum_peak_mlx_memory_bytes =
                        maximum_peak_mlx_memory_bytes.max(telemetry.peak_memory_bytes);
                }

                if generated_token_ids.len().is_multiple_of(32) {
                    eprintln!(
                        "[mtp-speed] status=progress generated_tokens={}/{}",
                        generated_token_ids.len(),
                        output_token_count
                    );
                }

                if let Some(finalization) = generation_finalization {
                    if let Some(telemetry) = finalization.mlx_memory_telemetry() {
                        maximum_active_mlx_memory_bytes =
                            maximum_active_mlx_memory_bytes.max(telemetry.active_memory_bytes);
                        maximum_peak_mlx_memory_bytes =
                            maximum_peak_mlx_memory_bytes.max(telemetry.peak_memory_bytes);
                    }
                    return GenerationMeasurement {
                        generated_token_ids,
                        generation_elapsed_seconds: generated_at
                            .saturating_duration_since(first_generated_at)
                            .as_secs_f64(),
                        total_request_elapsed_seconds: generated_at
                            .saturating_duration_since(request_started_at)
                            .as_secs_f64(),
                        maximum_active_mlx_memory_bytes,
                        maximum_peak_mlx_memory_bytes,
                    };
                }
            }
            GeneratedToken::PrefillProgress {
                mlx_memory_telemetry,
                ..
            } => {
                if let Some(telemetry) = mlx_memory_telemetry {
                    maximum_active_mlx_memory_bytes =
                        maximum_active_mlx_memory_bytes.max(telemetry.active_memory_bytes);
                    maximum_peak_mlx_memory_bytes =
                        maximum_peak_mlx_memory_bytes.max(telemetry.peak_memory_bytes);
                }
            }
            GeneratedToken::PromptProcessingPhaseStarted { .. }
            | GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => {
                panic!("the request should finalize on an emitted token, not end-of-sequence")
            }
        }
    }
}

// ── Attribution log helpers ───────────────────────────────────────────────────

fn generation_report(path: &std::path::Path, request_id: RequestId) -> serde_json::Value {
    let content = std::fs::read_to_string(path).expect("the attribution log should be readable");
    content
        .lines()
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .expect("the report should be valid JSON")
        })
        .find(|report| report["request_id"] == request_id.value())
        .expect("the requested attribution report should exist")
}

fn counter_amount(report: &serde_json::Value, identifier: &str) -> u64 {
    report["counters"]
        .as_array()
        .and_then(|counters| {
            counters
                .iter()
                .find(|counter| counter["counter"] == identifier)
        })
        .and_then(|counter| counter["amount"].as_u64())
        .unwrap_or(0)
}

fn print_operation_report(report: &serde_json::Value, cell_name: &str) {
    let Some(operation_reports) = report["operations"].as_array() else {
        eprintln!("[mtp-speed-op] cell={cell_name} operations=unavailable");
        return;
    };
    let mut operation_rows: Vec<(String, u64, u64)> = operation_reports
        .iter()
        .filter_map(|operation_report| {
            Some((
                operation_report["operation"].as_str()?.to_owned(),
                operation_report["total_elapsed_nanoseconds"].as_u64()?,
                operation_report["occurrence_count"].as_u64()?,
            ))
        })
        .collect();
    operation_rows.sort_by(|left, right| right.1.cmp(&left.1));
    for (operation_name, elapsed_nanoseconds, occurrence_count) in operation_rows {
        eprintln!(
            "[mtp-speed-op] cell={cell_name} operation={operation_name} elapsed_ms={:.3} occurrences={occurrence_count}",
            elapsed_nanoseconds as f64 / 1_000_000.0,
        );
    }
}
