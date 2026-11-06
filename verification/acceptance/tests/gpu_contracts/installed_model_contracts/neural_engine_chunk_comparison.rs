//! How steady a comparison of neighbouring real chunks is, on several real models.
//!
//! Timing a model at load with made-up tokens read 1.06, 0.68, 1.07 and 0.95 against a
//! true 1.1×, so a decision about a model's programs needs evidence from real prompts.
//! Each pass here is one prompt whose chunks take turns on the GPU alone and on both
//! engines, the only way to compare the engines without a second run of the prompt.
//! Run with no model loaded in either instance.

use std::thread::sleep;
use std::time::{Duration, Instant};

use astronomical_config::AstronomicalConfig;
use astronomical_ipc_protocol::NeuralEnginePrefillStatus;
use astronomical_model_serving::Qwen3_5ResidentModel;
use serial_test::serial;

use super::neural_engine_mlp_prefill::{
    CHUNK_TOKEN_COUNT, FULL_ATTENTION_KV_GROWTH_TOKENS, load_named_dense_model,
};

const JOURNEY_NAME: &str = "neural-engine-chunk-comparison";
const WARM_PASS_COUNT: usize = 6;
const COLD_PASS_COUNT: usize = 2;
const LARGEST_CHUNK_COUNT_PER_PASS: usize = 12;
// Longer than the driver keeps programs wired after their last use.
const COLD_PASS_IDLE: Duration = Duration::from_secs(25);

fn set_engines(model: &Qwen3_5ResidentModel, is_gpu_alone: bool) {
    model.set_neural_engine_mlp_prefill_paused(is_gpu_alone);
    model.set_neural_engine_output_gate_paused(is_gpu_alone);
}

/// One prompt's chunks in turn, returning each chunk's milliseconds in order.
fn one_pass_chunk_millis(
    model: &Qwen3_5ResidentModel,
    prompt_token_ids: &[u32],
    chunk_count: usize,
    does_gpu_go_first: bool,
) -> Vec<f64> {
    let mut decoder_state = model
        .empty_request_decoder_state_matching_loaded_weights(FULL_ATTENTION_KV_GROWTH_TOKENS)
        .expect("the measured state allocates");
    let mut chunk_millis = Vec::with_capacity(chunk_count);
    for chunk_index in 0..chunk_count {
        set_engines(model, (chunk_index % 2 == 0) == does_gpu_go_first);
        let chunk_start = chunk_index * CHUNK_TOKEN_COUNT;
        let chunk_started_at = Instant::now();
        let chunk_logits = model
            .forward_chunk(
                &prompt_token_ids[chunk_start..chunk_start + CHUNK_TOKEN_COUNT],
                chunk_start as u32,
                &mut decoder_state,
            )
            .expect("the measured chunk runs");
        chunk_logits.evaluate().expect("the chunk evaluates");
        chunk_millis.push(chunk_started_at.elapsed().as_secs_f64() * 1_000.0);
    }
    chunk_millis
}

fn compare_chunks(dense_model_id: &str) {
    let development_config =
        AstronomicalConfig::load_from_development_location().expect("the config loads");
    let (mut model, prompt_token_ids) = load_named_dense_model(&development_config, dense_model_id);
    let mut chunking = crate::common::standard_worker_chunking_configuration();
    chunking.experimental_neural_engine_prefill_enabled = true;
    let status = model.prepare_neural_engine_prefill_with_token_count(&chunking, 960);
    assert!(
        matches!(status, NeuralEnginePrefillStatus::Active { .. }),
        "the programs must load for this comparison: {status:?}"
    );
    let chunk_count =
        (prompt_token_ids.len() / CHUNK_TOKEN_COUNT).min(LARGEST_CHUNK_COUNT_PER_PASS);
    let journey_started_at = Instant::now();
    let total_pass_count = WARM_PASS_COUNT + COLD_PASS_COUNT;
    for pass_index in 0..total_pass_count {
        let is_cold = pass_index >= WARM_PASS_COUNT;
        if is_cold {
            sleep(COLD_PASS_IDLE);
        }
        let does_gpu_go_first = pass_index % 2 == 0;
        let chunk_millis =
            one_pass_chunk_millis(&model, &prompt_token_ids, chunk_count, does_gpu_go_first);
        let finished_count = pass_index + 1;
        let remaining_seconds = journey_started_at.elapsed().as_secs_f64() / finished_count as f64
            * (total_pass_count - finished_count) as f64;
        eprintln!(
            "[{JOURNEY_NAME}] MEASURED model={dense_model_id} pass={pass_index} cold={is_cold} gpu_first={does_gpu_go_first} chunk_ms={chunk_millis:.0?} eta_s={remaining_seconds:.0}",
        );
    }
}

#[tokio::test]
#[ignore = "loads a real 0.8B model and its programs"]
#[serial]
async fn should_compare_real_chunks_on_the_smallest_model() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    compare_chunks("Qwen3.5-0.8B-MLX-4bit");
}

#[tokio::test]
#[ignore = "loads a real 2B model and its programs"]
#[serial]
async fn should_compare_real_chunks_on_the_two_billion_parameter_model() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    compare_chunks("Qwen3.5-2B-4bit");
}

#[tokio::test]
#[ignore = "loads a real 9B model and its programs"]
#[serial]
async fn should_compare_real_chunks_on_the_nine_billion_parameter_model() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    compare_chunks("Ornith-1.5-9B-MLX-4bit-vision");
}
