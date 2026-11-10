//! Real-model proof of the opt-in Neural Engine prefill (issue #161).
//!
//! One loaded dense model runs the same Shakespeare prompt through raw model
//! forwards, alternating between the GPU alone and one Neural Engine MLP program
//! per layer, so the arms differ only in the programs. The order swaps each round
//! because warm-up and heat otherwise favour whichever arm runs second. Earlier
//! journeys used synthetic weights and timed fixed-shape projections alone; this
//! includes attention, recurrent state, the per-layer synchronization a hand-off
//! forces, and real weights.
//!
//! Run with no model loaded in either instance, because a resident model competes
//! for the same memory and GPU and would contaminate the comparison.

use std::time::{Duration, Instant};

use astronomical_config::{AstronomicalConfig, ModelCapabilities};
use astronomical_ipc_protocol::NeuralEnginePrefillStatus;
use astronomical_memory_policy::serving_ceiling::machine_ceiling_from_physical_ram_bytes;
use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5ResidentModel, Qwen3_5Tokenizer,
};
use astronomical_neural_engine_research::process_memory::{
    MemoryGrowth, MemorySnapshot, mebibytes, physical_memory_bytes,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::serial;
use tokio::time::timeout;

const JOURNEY_NAME: &str = "neural-engine-real-model";
pub(super) const DENSE_MODEL_ID: &str = "Ornith-1.5-9B-MLX-4bit-vision";
pub(super) const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

pub(super) const CHUNK_TOKEN_COUNT: usize = 2_048;
// Five chunks is the repository's 10,000-token measurement, within tolerance.
const PREFILL_CHUNK_COUNT: usize = 5;
const WARM_UP_TOKEN_COUNT: usize = 500;
const WARM_UP_DECODE_TOKEN_COUNT: usize = 50;
const DECODE_TOKEN_COUNT: usize = 500;
const MEASUREMENT_ROUND_COUNT: usize = 3;
pub(super) const FULL_ATTENTION_KV_GROWTH_TOKENS: i32 = 256;
pub(super) const MAXIMUM_OUTPUT_TOKENS: u32 = 20_480;
const MINIMUM_FINAL_LOGIT_COSINE: f64 = 0.999;
// Model load, one program per layer, and both arms; a documented real-model
// exception to the usual budget, like the other artifact load journeys.
pub(super) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(600);

struct ArmEvidence {
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    final_logits: Vec<f32>,
    greedy_token_ids: Vec<u32>,
}

#[tokio::test]
#[ignore = "loads a real dense artifact and about 20 GiB of Neural Engine programs"]
#[serial]
async fn should_compare_real_dense_prefill_with_and_without_the_neural_engine() {
    timeout(JOURNEY_TIMEOUT, compare_prefill_arms())
        .await
        .expect("the real-model Neural Engine journey must finish inside its bound");
}

fn fixture_prompt_token_ids(tokenizer: &Qwen3_5Tokenizer) -> Vec<u32> {
    let rendered_prompt =
        format!("<|im_start|>user\n{ROMEO_AND_JULIET_SOURCE}<|im_end|>\n<|im_start|>assistant\n");
    tokenizer
        .encode_prompt(&rendered_prompt)
        .expect("the Shakespeare fixture encodes")
}

pub(super) fn last_token_logits(runtime: &MlxRuntime, logits: &MlxArray) -> Vec<f32> {
    let single_precision = runtime
        .astype(logits, MlxDtype::Float32)
        .expect("the logits convert");
    let all_logits = runtime
        .array_to_vec_f32(&single_precision)
        .expect("the logits read back");
    let vocabulary_size = *logits.shape().last().expect("logits have a shape") as usize;
    all_logits[all_logits.len() - vocabulary_size..].to_vec()
}

fn greedy_token(logits: &[f32]) -> u32 {
    let mut best_token_index = 0;
    for (token_index, logit) in logits.iter().enumerate() {
        if *logit > logits[best_token_index] {
            best_token_index = token_index;
        }
    }
    best_token_index as u32
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    let mut dot_product = 0.0_f64;
    let mut left_squared = 0.0_f64;
    let mut right_squared = 0.0_f64;
    for (left_value, right_value) in left.iter().zip(right) {
        dot_product += f64::from(*left_value) * f64::from(*right_value);
        left_squared += f64::from(*left_value) * f64::from(*left_value);
        right_squared += f64::from(*right_value) * f64::from(*right_value);
    }
    dot_product / (left_squared.sqrt() * right_squared.sqrt())
}

/// First-use kernels must not inflate either arm's first measured chunk or token.
fn warm_up(model: &Qwen3_5ResidentModel, prompt_token_ids: &[u32], label: &str) {
    let mut decoder_state = model
        .empty_request_decoder_state_matching_loaded_weights(FULL_ATTENTION_KV_GROWTH_TOKENS)
        .expect("the warm-up state allocates");
    model
        .prefill_chunk(
            &prompt_token_ids[..WARM_UP_TOKEN_COUNT],
            0,
            &mut decoder_state,
        )
        .expect("the warm-up prefill runs");
    let chunk_end = WARM_UP_TOKEN_COUNT + CHUNK_TOKEN_COUNT;
    let logits = model
        .forward_chunk(
            &prompt_token_ids[WARM_UP_TOKEN_COUNT..chunk_end],
            WARM_UP_TOKEN_COUNT as u32,
            &mut decoder_state,
        )
        .expect("the warm-up full chunk runs");
    logits.evaluate().expect("the warm-up logits evaluate");
    greedy_continuation(
        model,
        &last_token_logits(model.runtime(), &logits),
        &mut decoder_state,
        chunk_end,
        WARM_UP_DECODE_TOKEN_COUNT,
    );
    eprintln!("[{JOURNEY_NAME}] status=progress arm={label} phase=warm_up_complete");
}

fn run_arm(model: &Qwen3_5ResidentModel, prompt_token_ids: &[u32], label: &str) -> ArmEvidence {
    warm_up(model, prompt_token_ids, label);
    let mut decoder_state = model
        .empty_request_decoder_state_matching_loaded_weights(FULL_ATTENTION_KV_GROWTH_TOKENS)
        .expect("the measured state allocates");
    let prefill_started_at = Instant::now();
    for chunk_index in 0..PREFILL_CHUNK_COUNT - 1 {
        let chunk_start = chunk_index * CHUNK_TOKEN_COUNT;
        model
            .prefill_chunk(
                &prompt_token_ids[chunk_start..chunk_start + CHUNK_TOKEN_COUNT],
                chunk_start as u32,
                &mut decoder_state,
            )
            .expect("the measured prefill chunk runs");
    }
    let final_chunk_start = (PREFILL_CHUNK_COUNT - 1) * CHUNK_TOKEN_COUNT;
    let final_chunk_logits = model
        .forward_chunk(
            &prompt_token_ids[final_chunk_start..final_chunk_start + CHUNK_TOKEN_COUNT],
            final_chunk_start as u32,
            &mut decoder_state,
        )
        .expect("the final chunk runs");
    final_chunk_logits
        .evaluate()
        .expect("the final logits evaluate");
    let prefill_elapsed = prefill_started_at.elapsed();
    let final_logits = last_token_logits(model.runtime(), &final_chunk_logits);
    let decode_started_at = Instant::now();
    let greedy_token_ids = greedy_continuation(
        model,
        &final_logits,
        &mut decoder_state,
        PREFILL_CHUNK_COUNT * CHUNK_TOKEN_COUNT,
        DECODE_TOKEN_COUNT,
    );
    // The first token came from the prefill's own logits, so it took no decode step.
    let decode_step_count = (DECODE_TOKEN_COUNT - 1) as f64;
    ArmEvidence {
        prefill_tokens_per_second: (PREFILL_CHUNK_COUNT * CHUNK_TOKEN_COUNT) as f64
            / prefill_elapsed.as_secs_f64(),
        decode_tokens_per_second: decode_step_count / decode_started_at.elapsed().as_secs_f64(),
        final_logits,
        greedy_token_ids,
    }
}

pub(super) fn greedy_continuation(
    model: &Qwen3_5ResidentModel,
    final_logits: &[f32],
    decoder_state: &mut astronomical_model_serving::RequestDecoderStateStack,
    first_position: usize,
    token_count: usize,
) -> Vec<u32> {
    let mut greedy_token_ids = vec![greedy_token(final_logits)];
    while greedy_token_ids.len() < token_count {
        let position = first_position + greedy_token_ids.len() - 1;
        let previous_token_id = greedy_token_ids[greedy_token_ids.len() - 1];
        let logits = model
            .forward_chunk(
                std::slice::from_ref(&previous_token_id),
                position as u32,
                decoder_state,
            )
            .expect("the decode step runs");
        greedy_token_ids.push(greedy_token(&last_token_logits(model.runtime(), &logits)));
    }
    greedy_token_ids
}

pub(super) fn load_dense_model(
    development_config: &AstronomicalConfig,
) -> (Qwen3_5ResidentModel, Vec<u32>) {
    load_named_dense_model(development_config, DENSE_MODEL_ID)
}

pub(super) fn load_named_dense_model(
    development_config: &AstronomicalConfig,
    dense_model_id: &str,
) -> (Qwen3_5ResidentModel, Vec<u32>) {
    let discovered_model =
        crate::common::configured_discovered_model_by_id(development_config, dense_model_id);
    let ModelCapabilities::Chat(_) = &discovered_model.capabilities else {
        panic!("{dense_model_id} must be a chat model");
    };
    let model_directory = discovered_model.model_directory.clone();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, MAXIMUM_OUTPUT_TOKENS)
        .expect("the artifact passes production validation");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the tokenizer loads");
    let prompt_token_ids = fixture_prompt_token_ids(&tokenizer);
    assert!(prompt_token_ids.len() >= PREFILL_CHUNK_COUNT * CHUNK_TOKEN_COUNT);
    // This Mac's own maximum, not the user's setting: the comparison should not
    // depend on where a developer last left their ceiling.
    let memory_ceiling_bytes =
        machine_ceiling_from_physical_ram_bytes(physical_memory_bytes()) as usize;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(memory_ceiling_bytes, memory_ceiling_bytes)
            .expect("the configured limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let mut model = Qwen3_5ResidentModel::load(
        runtime,
        validated_artifact,
        &model_directory,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the dense model binds");
    model
        .materialize_complete_weights_matching_production()
        .expect("the dense weights materialize");
    (model, prompt_token_ids)
}

fn has_only_finite_values(values: &[f32]) -> bool {
    for value in values {
        if !value.is_finite() {
            return false;
        }
    }
    true
}

pub(super) fn count_matching_tokens(left: &[u32], right: &[u32]) -> usize {
    let mut matching_token_count = 0;
    for (left_token, right_token) in left.iter().zip(right) {
        if left_token != right_token {
            break;
        }
        matching_token_count += 1;
    }
    matching_token_count
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn run_paused_arm(
    model: &Qwen3_5ResidentModel,
    prompt_token_ids: &[u32],
    is_paused: bool,
    round_index: usize,
    measurement_started_at: Instant,
) -> ArmEvidence {
    let label = if is_paused { "gpu" } else { "both" };
    model.set_neural_engine_mlp_prefill_paused(is_paused);
    model.set_neural_engine_output_gate_paused(is_paused);
    let evidence = run_arm(model, prompt_token_ids, label);
    eprintln!(
        "[{JOURNEY_NAME}] ROUND round={}/{MEASUREMENT_ROUND_COUNT} arm={label} prefill_tps={:.1} decode_tps={:.2} elapsed_s={:.0}",
        round_index + 1,
        evidence.prefill_tokens_per_second,
        evidence.decode_tokens_per_second,
        measurement_started_at.elapsed().as_secs_f64(),
    );
    evidence
}

/// Runs every round and returns the GPU arm's and both engines' evidence, with
/// the order swapped each round so neither arm always benefits from going second.
fn run_alternating_rounds(
    model: &Qwen3_5ResidentModel,
    prompt_token_ids: &[u32],
) -> (Vec<ArmEvidence>, Vec<ArmEvidence>) {
    let measurement_started_at = Instant::now();
    let mut gpu_evidence = Vec::with_capacity(MEASUREMENT_ROUND_COUNT);
    let mut both_engines_evidence = Vec::with_capacity(MEASUREMENT_ROUND_COUNT);
    for round_index in 0..MEASUREMENT_ROUND_COUNT {
        let gpu_goes_first = round_index % 2 == 0;
        for is_paused in [gpu_goes_first, !gpu_goes_first] {
            let evidence = run_paused_arm(
                model,
                prompt_token_ids,
                is_paused,
                round_index,
                measurement_started_at,
            );
            if is_paused {
                gpu_evidence.push(evidence);
            } else {
                both_engines_evidence.push(evidence);
            }
        }
    }
    (gpu_evidence, both_engines_evidence)
}

/// Each round's B over A, then the median. Speed drifts across a run as the Mac
/// warms, so comparing separate medians can invent a difference no round showed.
fn median_paired_ratio(baseline_samples: &[f64], candidate_samples: &[f64]) -> f64 {
    let mut ratios = Vec::with_capacity(baseline_samples.len());
    for (baseline_sample, candidate_sample) in baseline_samples.iter().zip(candidate_samples) {
        ratios.push(candidate_sample / baseline_sample);
    }
    median(ratios)
}

fn decode_samples(evidence: &[ArmEvidence]) -> Vec<f64> {
    let mut samples = Vec::with_capacity(evidence.len());
    for arm_evidence in evidence {
        samples.push(arm_evidence.decode_tokens_per_second);
    }
    samples
}

fn prefill_samples(evidence: &[ArmEvidence]) -> Vec<f64> {
    let mut samples = Vec::with_capacity(evidence.len());
    for arm_evidence in evidence {
        samples.push(arm_evidence.prefill_tokens_per_second);
    }
    samples
}

fn report_comparison(gpu_evidence: &[ArmEvidence], both_engines_evidence: &[ArmEvidence]) {
    let gpu_prefill = prefill_samples(gpu_evidence);
    let both_prefill = prefill_samples(both_engines_evidence);
    let gpu_decode = decode_samples(gpu_evidence);
    let both_decode = decode_samples(both_engines_evidence);
    let final_logit_cosine = cosine_similarity(
        &gpu_evidence[0].final_logits,
        &both_engines_evidence[0].final_logits,
    );
    let gpu_repeat_cosine =
        cosine_similarity(&gpu_evidence[0].final_logits, &gpu_evidence[1].final_logits);
    let matching_token_count = count_matching_tokens(
        &gpu_evidence[0].greedy_token_ids,
        &both_engines_evidence[0].greedy_token_ids,
    );
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED prefill_gpu_tps={:.1} prefill_both_tps={:.1} prefill_paired_speedup={:.3}x \
         decode_gpu_tps={:.2} decode_both_tps={:.2} decode_paired_ratio={:.3}x final_logit_cosine={final_logit_cosine:.8} \
         gpu_repeat_cosine={gpu_repeat_cosine:.8} greedy_matching_prefix={matching_token_count}/{DECODE_TOKEN_COUNT}",
        median(gpu_prefill.clone()),
        median(both_prefill.clone()),
        median_paired_ratio(&gpu_prefill, &both_prefill),
        median(gpu_decode.clone()),
        median(both_decode.clone()),
        median_paired_ratio(&gpu_decode, &both_decode),
    );
    assert!(has_only_finite_values(
        &both_engines_evidence[0].final_logits
    ));
    assert!(final_logit_cosine >= MINIMUM_FINAL_LOGIT_COSINE);
}

/// Builds the programs through serving's own steps and waits, keeping compiled ones in
/// `program_cache_directory` when given.
pub(super) fn prepare_programs(
    model: &mut Qwen3_5ResidentModel,
    program_cache_directory: Option<&std::path::Path>,
) -> NeuralEnginePrefillStatus {
    let mut chunking = crate::common::standard_worker_chunking_configuration();
    chunking.experimental_neural_engine_prefill_enabled = true;
    model.prepare_neural_engine_prefill_matching_production(&chunking, program_cache_directory)
}

async fn compare_prefill_arms() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    let development_config =
        AstronomicalConfig::load_from_development_location().expect("the config loads");
    let (mut model, prompt_token_ids) = load_dense_model(&development_config);
    eprintln!("[{JOURNEY_NAME}] status=progress phase=model_loaded model={DENSE_MODEL_ID}");

    let before_load = MemorySnapshot::capture();
    let load_started_at = Instant::now();
    let status = prepare_programs(&mut model, None);
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED program_load_s={:.1} status={status:?}",
        load_started_at.elapsed().as_secs_f64()
    );
    assert!(
        matches!(status, NeuralEnginePrefillStatus::Active { .. }),
        "the programs must load for this comparison: {status:?}"
    );

    let (gpu_evidence, both_engines_evidence) = run_alternating_rounds(&model, &prompt_token_ids);
    let growth = MemoryGrowth::between(before_load, MemorySnapshot::capture());
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED programs_footprint_mib={:.0} programs_wired_mib={:.0}",
        mebibytes(growth.process_physical_footprint_bytes),
        mebibytes(growth.system_wired_bytes),
    );
    report_comparison(&gpu_evidence, &both_engines_evidence);
}

/// Prefill alone, for a change that cannot affect decode. A 500-token decode in
/// every arm would more than double a run that asks nothing about decode.
pub(super) fn prefill_only_tokens_per_second(
    model: &Qwen3_5ResidentModel,
    prompt_token_ids: &[u32],
) -> f64 {
    let mut decoder_state = model
        .empty_request_decoder_state_matching_loaded_weights(FULL_ATTENTION_KV_GROWTH_TOKENS)
        .expect("the measured state allocates");
    let prefill_started_at = Instant::now();
    for chunk_index in 0..PREFILL_CHUNK_COUNT {
        let chunk_start = chunk_index * CHUNK_TOKEN_COUNT;
        let chunk_logits = model
            .forward_chunk(
                &prompt_token_ids[chunk_start..chunk_start + CHUNK_TOKEN_COUNT],
                chunk_start as u32,
                &mut decoder_state,
            )
            .expect("the measured chunk runs");
        chunk_logits.evaluate().expect("the chunk evaluates");
    }
    (PREFILL_CHUNK_COUNT * CHUNK_TOKEN_COUNT) as f64 / prefill_started_at.elapsed().as_secs_f64()
}

/// The paired speedup of the loaded programs over the GPU alone, prefill only. The
/// first round is a warm-up for both arms and is not counted.
fn report_paired_prefill_speedup(
    model: &Qwen3_5ResidentModel,
    prompt_token_ids: &[u32],
    setup_label: &str,
) {
    let mut gpu_samples = Vec::with_capacity(MEASUREMENT_ROUND_COUNT);
    let mut both_samples = Vec::with_capacity(MEASUREMENT_ROUND_COUNT);
    for round_index in 0..=MEASUREMENT_ROUND_COUNT {
        let gpu_goes_first = round_index % 2 == 0;
        for is_paused in [gpu_goes_first, !gpu_goes_first] {
            model.set_neural_engine_mlp_prefill_paused(is_paused);
            model.set_neural_engine_output_gate_paused(is_paused);
            let tokens_per_second = prefill_only_tokens_per_second(model, prompt_token_ids);
            if round_index == 0 {
                continue;
            }
            if is_paused {
                gpu_samples.push(tokens_per_second);
            } else {
                both_samples.push(tokens_per_second);
            }
        }
    }
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED setup={setup_label} gpu_tps={:?} both_tps={:?} prefill_paired_speedup={:.3}x",
        gpu_samples,
        both_samples,
        median_paired_ratio(&gpu_samples, &both_samples),
    );
}

/// The production loader with a program cache, prefill only: what a user gets
/// from the second load of a model onwards.
async fn compare_prefill_with_kept_programs() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    let development_config =
        AstronomicalConfig::load_from_development_location().expect("the config loads");
    let (mut model, prompt_token_ids) = load_dense_model(&development_config);
    let program_cache_directory = std::env::temp_dir().join(PROGRAM_CACHE_DIRECTORY_NAME);
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED cache_existed_at_start={}",
        program_cache_directory.is_dir()
    );
    let load_started_at = Instant::now();
    let status = prepare_programs(&mut model, Some(&program_cache_directory));
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED program_load_s={:.1} cache_gib={:.1} status={status:?}",
        load_started_at.elapsed().as_secs_f64(),
        directory_size_gibibytes(&program_cache_directory),
    );
    assert!(matches!(status, NeuralEnginePrefillStatus::Active { .. }));
    report_paired_prefill_speedup(&model, &prompt_token_ids, "mlp-and-output-gates");
}

#[tokio::test]
#[ignore = "loads a real dense artifact and its Neural Engine programs through a program cache"]
#[serial]
async fn should_serve_real_dense_prefill_from_kept_programs() {
    timeout(JOURNEY_TIMEOUT, compare_prefill_with_kept_programs())
        .await
        .expect("the kept-program journey must finish inside its bound");
}

pub(super) fn directory_size_gibibytes(directory: &std::path::Path) -> f64 {
    let report = std::process::Command::new("du")
        .arg("-sk")
        .arg(directory)
        .output()
        .expect("du runs");
    let report_text = String::from_utf8_lossy(&report.stdout);
    let kibibytes: f64 = report_text
        .split_whitespace()
        .next()
        .expect("du reports a size")
        .parse()
        .expect("the size is a number");
    kibibytes / 1_048_576.0
}

// A fixed name, because macOS keeps a compiled program's Neural Engine work by its
// path, and a second run must find what the first one kept.
const PROGRAM_CACHE_DIRECTORY_NAME: &str = "astronomical-neural-engine-program-cache-journey";
