//! Temperature-1 sampled MTP throughput measurement against target-only decode.
//!
//! The pair differs only by the configured `acceleration.mtp.enabled` decision:
//! both settings run the same binary, the same artifact, the same temperature
//! 1.0 top-k/top-p sampler with locked seeds, and the same Romeo and Juliet
//! prompt, in both process orders. Generation tokens per second counts the wall
//! clock from the first generated token (prefill excluded), and the accept
//! counters make any speedup or absence of one legible per #335's plan.

use std::time::{Duration, Instant};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, PerformanceAttributionLog,
    Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
};

use super::engine_support::{
    installed_model_has_complete_mtp_inventory, performance_counter_amount,
};

const MEASUREMENT_OUTPUT_TOKEN_COUNT: u16 = 64;
const MEASUREMENT_SEEDS: [u64; 3] = [101, 202, 303];
const MEASUREMENT_TIMEOUT_SECONDS: u64 = 115;
const ATTRIBUTION_TIMEOUT_SECONDS: u64 = 115;

#[tokio::test]
#[ignore = "loads the configured MTP artifacts four times and records bounded throughput runs"]
async fn should_measure_temperature_one_mtp_generation_throughput_against_target_only() {
    for artifact_directory in [
        crate::serving_acceptance::support::configured_depth_one_mtp_model_directory(),
        crate::serving_acceptance::support::dense_mtp_model_directory(),
    ] {
        let artifact_name = artifact_directory
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_owned());
        if !installed_model_has_complete_mtp_inventory(&artifact_directory) {
            eprintln!(
                "[mtp-throughput] status=skipped artifact={artifact_name} reason=installed_model_has_no_complete_mtp_inventory"
            );
            continue;
        }
        tokio::time::timeout(
            Duration::from_secs(MEASUREMENT_TIMEOUT_SECONDS),
            measure_one_artifact(&artifact_directory, &artifact_name),
        )
        .await
        .expect("the MTP throughput measurement should finish within its bound");
    }
}

async fn measure_one_artifact(artifact_directory: &std::path::Path, artifact_name: &str) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let romeo_and_juliet_prompt_token_ids = romeo_and_juliet_prompt_token_ids(artifact_directory);
    eprintln!(
        "[mtp-throughput] status=start artifact={artifact_name} prompt_tokens={} output_tokens={MEASUREMENT_OUTPUT_TOKEN_COUNT}",
        romeo_and_juliet_prompt_token_ids.len()
    );
    // Both process orders: candidate first, then baseline first, so allocator
    // and graph warm-up cannot favor one setting (issue #335 measurement plan).
    let mtp_first = [true, false];
    let baseline_first = [false, true];
    for (order_name, order) in [
        ("candidate_first", &mtp_first),
        ("baseline_first", &baseline_first),
    ] {
        for mtp_enabled in order {
            let (throughputs, accepted_draft_count, proposed_draft_count) =
                run_three_seeded_measurements(artifact_directory, *mtp_enabled).await;
            let median_throughput = median(&mut throughputs.clone());
            eprintln!(
                "[mtp-throughput] status=measured artifact={artifact_name} order={order_name} mtp_enabled={mtp_enabled} throughput_per_second={median_throughput:.2} runs={throughputs:?} accepted_drafts={accepted_draft_count} proposed_drafts={proposed_draft_count}"
            );
        }
    }
    eprintln!("[mtp-throughput] status=complete artifact={artifact_name}");
}

async fn run_three_seeded_measurements(
    artifact_directory: &std::path::Path,
    mtp_enabled: bool,
) -> (Vec<f64>, u64, u64) {
    let mut measurement_throughputs = Vec::with_capacity(MEASUREMENT_SEEDS.len());
    let mut engine = load_measurement_engine(artifact_directory, mtp_enabled).await;
    engine
        .load()
        .await
        .expect("the measurement engine should materialize the configured artifact");
    for seed in MEASUREMENT_SEEDS {
        let request_id = RequestId::new(
            u64::from_be_bytes([
                (seed >> 56) as u8,
                (seed >> 48) as u8,
                (seed >> 40) as u8,
                (seed >> 32) as u8,
                (seed >> 24) as u8,
                (seed >> 16) as u8,
                (seed >> 8) as u8,
                seed as u8,
            ]) | 1,
        );
        let request = Qwen3_5InferenceRequest::new_sampling(
            request_id,
            romeo_and_juliet_prompt_token_ids(artifact_directory),
            MEASUREMENT_OUTPUT_TOKEN_COUNT,
            1_000,
            1_000,
            Some(seed),
        )
        .with_image_pad_token_id(
            crate::serving_acceptance::mtp::engine_support::configured_mtp_artifact_test_inputs(
                artifact_directory,
            )
            .image_pad_token_id,
        )
        .with_performance_attribution(PerformanceAttribution::enabled());
        engine
            .start_generation(request)
            .await
            .expect("the measurement engine should accept the sampled request");
        let decode_started_at = Instant::now();
        let mut generated_token_count = 0_u32;
        loop {
            match tokio::time::timeout(
                Duration::from_secs(ATTRIBUTION_TIMEOUT_SECONDS),
                engine.decode_next_token(request_id),
            )
            .await
            .expect("each decode boundary should finish within its bound")
            .expect("each decode boundary should advance the request")
            {
                GeneratedToken::TokenId { .. } => {
                    generated_token_count += 1;
                }
                GeneratedToken::PrefillProgress { .. }
                | GeneratedToken::PromptProcessingPhaseStarted { .. }
                | GeneratedToken::GenerationPreparationStarted { .. } => {}
                GeneratedToken::EndOfSequence => break,
            }
            if generated_token_count >= u32::from(MEASUREMENT_OUTPUT_TOKEN_COUNT) {
                break;
            }
        }
        let decode_elapsed = decode_started_at.elapsed();
        let throughput = f64::from(generated_token_count) / decode_elapsed.as_secs_f64();
        measurement_throughputs.push(throughput);
        let run_report = attribution_last_report();
        let run_accepted_draft_count =
            performance_counter_amount(&run_report, "mtp_accepted_draft_count");
        let run_proposed_draft_count =
            performance_counter_amount(&run_report, "mtp_proposed_draft_count");
        eprintln!(
            "[mtp-throughput] status=run mtp_enabled={mtp_enabled} seed={seed} tokens={generated_token_count} decode_elapsed_seconds={:.2} throughput_per_second={throughput:.2} accepted_drafts={run_accepted_draft_count} proposed_drafts={run_proposed_draft_count}",
            decode_elapsed.as_secs_f64()
        );
    }
    drop(engine);
    let final_report = attribution_last_report();
    let accepted_draft_count =
        performance_counter_amount(&final_report, "mtp_accepted_draft_count");
    let proposed_draft_count =
        performance_counter_amount(&final_report, "mtp_proposed_draft_count");
    (
        measurement_throughputs,
        accepted_draft_count,
        proposed_draft_count,
    )
}

fn attribution_last_report() -> serde_json::Value {
    let attribution_reports = std::fs::read_to_string(measurement_attribution_log_path())
        .expect("the measurement attribution log should be readable");
    attribution_reports
        .lines()
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .expect("each attribution record should be valid JSON")
        })
        .next_back()
        .expect("the measurement should write attribution records")
}

fn measurement_attribution_log_path() -> std::path::PathBuf {
    std::env::temp_dir().join("astronomical-mtp-throughput-attribution.jsonl")
}

/// Returns the first chunk of the tracked Romeo and Juliet fixture, used as the
/// measured prompt so long-form prose exercises the decode path (issue rules).
fn romeo_and_juliet_prompt_token_ids(artifact_directory: &std::path::Path) -> Vec<u32> {
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt",
    );
    let romeo_and_juliet_text = std::fs::read_to_string(&fixture_path)
        .expect("the Romeo and Juliet fixture should be readable");
    let measured_excerpt: String = romeo_and_juliet_text.chars().take(2_400).collect();
    let validated_artifact = astronomical_model_serving::Qwen3_5ArtifactValidator::new()
        .validate(artifact_directory, 20_480)
        .expect("the measured artifact should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the measured tokenizer should prepare the fixture prompt");
    use astronomical_ipc_protocol::{
        ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice,
    };
    let prepared_prompt = tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id: RequestId::new(42_001),
                model: validated_artifact.model_id().to_owned(),
                messages: vec![ChatMessage::User {
                    content: format!(
                        "Continue the following passage in matching style:\n\n{measured_excerpt}"
                    ),
                    images: Vec::new(),
                }],
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                settings: ChatGenerationSettings {
                    max_output_tokens: MEASUREMENT_OUTPUT_TOKEN_COUNT,
                    temperature_thousandths: Some(1_000),
                    top_p_thousandths: Some(1_000),
                    seed: Some(101),
                    thinking_budget: None,
                },
                qwen_thinking_channel_seed: None,
            },
            false,
        )
        .expect("the fixture prompt should prepare");
    prepared_prompt.input_token_ids().to_vec()
}

/// Loads the measurement engine with a production chunk size so the fixture
/// prefill stays in one prompt-processing chunk and decode dominates.
async fn load_measurement_engine(
    artifact_directory: &std::path::Path,
    mtp_enabled: bool,
) -> Qwen3_5Engine {
    let validated_artifact = astronomical_model_serving::Qwen3_5ArtifactValidator::new()
        .validate(artifact_directory, 20_480)
        .expect("the measured artifact should validate before engine loading");
    let think_end_token_id = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the measured tokenizer should expose validated control tokens")
        .think_end_token_id();
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let attribution_log_path = measurement_attribution_log_path();
    let _ = std::fs::remove_file(&attribution_log_path);
    let performance_attribution_log = PerformanceAttributionLog::open(&attribution_log_path, true)
        .expect("the measurement should open its attribution log");
    let measurement_engine =
        Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
            validated_artifact,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            None,
            Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(2_048)
                .expect("the measurement prefill chunk should be valid"),
            think_end_token_id,
            artifact_directory.to_path_buf(),
            crate::common::standard_worker_chunking_configuration(),
            true,
            mtp_enabled,
            crate::common::disabled_worker_speculative_prefill_configuration(),
            PerformanceAttribution::enabled(),
            performance_attribution_log,
        )
        .expect("the measured engine settings should be valid");
    measurement_engine
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).expect("throughputs compare"));
    values[values.len() / 2]
}
