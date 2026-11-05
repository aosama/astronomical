//! Real-model proof of the opt-in Neural Engine prefill (issue #161).
//!
//! One loaded dense model runs the same Shakespeare prompt twice through raw
//! model forwards, first on the GPU alone and then with one Neural Engine MLP
//! program per layer, so the two arms differ only in the programs. Earlier
//! journeys used synthetic weights and timed fixed-shape projections alone; this
//! is the first measurement that includes attention, recurrent state, the
//! per-layer synchronization a hand-off forces, and real weights.
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
use astronomical_neural_engine::process_memory::physical_memory_bytes;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;
use tokio::time::timeout;

const JOURNEY_NAME: &str = "neural-engine-real-model";
const DENSE_MODEL_ID: &str = "Ornith-1.5-9B-vision-OptiQ-static-4.5bpw";
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

const CHUNK_TOKEN_COUNT: usize = 2_048;
// Five chunks is the repository's 10,000-token measurement, within tolerance.
const PREFILL_CHUNK_COUNT: usize = 5;
const WARM_UP_TOKEN_COUNT: usize = 500;
const GREEDY_TOKEN_COUNT: usize = 64;
const FULL_ATTENTION_KV_GROWTH_TOKENS: i32 = 256;
const MAXIMUM_OUTPUT_TOKENS: u32 = 20_480;
const MINIMUM_FINAL_LOGIT_COSINE: f64 = 0.999;
// Model load, one program per layer, and both arms; a documented real-model
// exception to the usual budget, like the other artifact load journeys.
const JOURNEY_TIMEOUT: Duration = Duration::from_secs(420);

struct ArmEvidence {
    prefill_tokens_per_second: f64,
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

fn last_token_logits(runtime: &MlxRuntime, logits: &MlxArray) -> Vec<f32> {
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

/// First-use kernels must not inflate either arm's first measured chunk.
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
    model
        .prefill_chunk(
            &prompt_token_ids[WARM_UP_TOKEN_COUNT..WARM_UP_TOKEN_COUNT + CHUNK_TOKEN_COUNT],
            WARM_UP_TOKEN_COUNT as u32,
            &mut decoder_state,
        )
        .expect("the warm-up full chunk runs");
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
        eprintln!(
            "[{JOURNEY_NAME}] status=progress arm={label} chunk={}/{PREFILL_CHUNK_COUNT} elapsed_ms={}",
            chunk_index + 1,
            prefill_started_at.elapsed().as_millis()
        );
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
    let greedy_token_ids = greedy_continuation(model, &final_logits, &mut decoder_state);
    ArmEvidence {
        prefill_tokens_per_second: (PREFILL_CHUNK_COUNT * CHUNK_TOKEN_COUNT) as f64
            / prefill_elapsed.as_secs_f64(),
        final_logits,
        greedy_token_ids,
    }
}

fn greedy_continuation(
    model: &Qwen3_5ResidentModel,
    final_logits: &[f32],
    decoder_state: &mut astronomical_model_serving::RequestDecoderStateStack,
) -> Vec<u32> {
    let mut greedy_token_ids = vec![greedy_token(final_logits)];
    while greedy_token_ids.len() < GREEDY_TOKEN_COUNT {
        let position = PREFILL_CHUNK_COUNT * CHUNK_TOKEN_COUNT + greedy_token_ids.len() - 1;
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

fn load_dense_model(
    development_config: &AstronomicalConfig,
) -> (Qwen3_5ResidentModel, Vec<u32>, u32) {
    let discovered_model =
        crate::common::configured_discovered_model_by_id(development_config, DENSE_MODEL_ID);
    let ModelCapabilities::Chat(chat_capabilities) = &discovered_model.capabilities else {
        panic!("{DENSE_MODEL_ID} must be a chat model");
    };
    let maximum_context_tokens = chat_capabilities.context_window;
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
    (model, prompt_token_ids, maximum_context_tokens)
}

fn has_only_finite_values(values: &[f32]) -> bool {
    for value in values {
        if !value.is_finite() {
            return false;
        }
    }
    true
}

fn count_matching_tokens(left: &[u32], right: &[u32]) -> usize {
    let mut matching_token_count = 0;
    for (left_token, right_token) in left.iter().zip(right) {
        if left_token != right_token {
            break;
        }
        matching_token_count += 1;
    }
    matching_token_count
}

async fn compare_prefill_arms() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    let development_config =
        AstronomicalConfig::load_from_development_location().expect("the config loads");
    let (mut model, prompt_token_ids, maximum_context_tokens) =
        load_dense_model(&development_config);
    eprintln!("[{JOURNEY_NAME}] status=progress phase=model_loaded model={DENSE_MODEL_ID}");

    let gpu_evidence = run_arm(&model, &prompt_token_ids, "gpu");
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED arm=gpu prefill_tps={:.1}",
        gpu_evidence.prefill_tokens_per_second
    );

    let mut chunking = crate::common::standard_worker_chunking_configuration();
    chunking.experimental_neural_engine_mlp_prefill_enabled = true;
    let load_started_at = Instant::now();
    let status = model
        .prepare_neural_engine_mlp_prefill_matching_production(&chunking, maximum_context_tokens);
    eprintln!(
        "[{JOURNEY_NAME}] status=progress phase=programs_loaded elapsed_s={:.1} status={status:?}",
        load_started_at.elapsed().as_secs_f64()
    );
    assert!(
        matches!(status, NeuralEnginePrefillStatus::Active { .. }),
        "the programs must load for this comparison: {status:?}"
    );

    let both_engines_evidence = run_arm(&model, &prompt_token_ids, "both");
    let final_logit_cosine = cosine_similarity(
        &gpu_evidence.final_logits,
        &both_engines_evidence.final_logits,
    );
    let matching_token_count = count_matching_tokens(
        &gpu_evidence.greedy_token_ids,
        &both_engines_evidence.greedy_token_ids,
    );
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED arm=both prefill_tps={:.1} speedup={:.3}x final_logit_cosine={final_logit_cosine:.8} \
         greedy_matching_prefix={matching_token_count}/{GREEDY_TOKEN_COUNT} same_first_token={}",
        both_engines_evidence.prefill_tokens_per_second,
        both_engines_evidence.prefill_tokens_per_second / gpu_evidence.prefill_tokens_per_second,
        gpu_evidence.greedy_token_ids[0] == both_engines_evidence.greedy_token_ids[0],
    );
    assert!(has_only_finite_values(&both_engines_evidence.final_logits));
    assert!(final_logit_cosine >= MINIMUM_FINAL_LOGIT_COSINE);
}
