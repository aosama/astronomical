//! How fast a dense model runs when its decoder layers stream from the SSD
//! (issue #1985), measured from the bottom of the call stack up.
//!
//! 1. Raw layer read rate: one decoder layer through the bounded-range
//!    reader, bytes per second.
//! 2. One streamed layer: read and run over a probe-sized prompt span, and
//!    over one decode token.
//! 3. Full runs: 10,000 Shakespeare tokens in, 500 out, at 0, 8, 16 and 24
//!    kept layers, each after a 500-in, 50-out warmup.
//!
//! The streamed_dense model loads under the machine's own limit with the
//! kept count pinned, prefills layer by layer, and decodes greedily. A cold
//! file cache keeps the weight files out of the macOS file cache throughout,
//! so reads cost what they cost on a Mac whose RAM is no larger than the
//! model; no prompt cache is involved. Run it on an optimized build.

use std::time::{Duration, Instant};

use astronomical_model_serving::{
    DenseLayerProbeShape, Qwen3_5ArtifactValidator, Qwen3_5Config, Qwen3_5StreamingModel,
};
use astronomical_runtime_integration::MlxRuntime;
use serial_test::serial;

#[path = "../../support/romeo_and_juliet.rs"]
mod romeo_and_juliet;

#[path = "../../chat_serving/installed_journeys/expert_paging/cold_file_cache.rs"]
mod cold_file_cache;

const KEPT_LAYER_COUNTS: [usize; 4] = [0, 8, 16, 24];
const PROMPT_TOKEN_COUNT: usize = 10_000;
const OUTPUT_TOKEN_COUNT: usize = 500;
const WARMUP_PROMPT_TOKEN_COUNT: usize = 500;
const WARMUP_OUTPUT_TOKEN_COUNT: usize = 50;
/// The engine's SSD-streaming prefill span.
const PREFILL_SPAN_TOKEN_COUNT: usize = 4096;
/// Enough Shakespeare words for 10,000 tokens.
const PROMPT_WORD_COUNT: usize = 9000;
/// Layers whose cold reads make the raw read rate.
const RAW_READ_LAYER_COUNT: usize = 8;
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
/// Long enough for the cold file cache to sweep the layer just read.
const COLD_FILE_CACHE_SETTLE: Duration = Duration::from_millis(500);

#[tokio::test]
#[ignore = "streams the streamed_dense model from a cold SSD for many minutes"]
#[serial]
async fn should_measure_dense_streaming_speed_from_the_layer_read_up() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_streamed_dense_model_directory();
    let prompt_token_ids = shakespeare_prompt_token_ids(&model_directory);
    let machine_mlx_memory_ceiling_bytes =
        crate::common::sample_machine_mlx_memory_ceiling_bytes().await;
    let (streaming_model, qwen3_5_config) =
        load_streaming_model(&model_directory, machine_mlx_memory_ceiling_bytes);
    let cold_file_cache = cold_file_cache::ColdFileCache::start(&model_directory);

    measure_raw_layer_read_rate(&streaming_model, &qwen3_5_config);
    measure_one_streamed_layer(&streaming_model, &qwen3_5_config);
    for (kept_count_position, kept_layer_count) in KEPT_LAYER_COUNTS.into_iter().enumerate() {
        pin_kept_layers(&streaming_model, kept_layer_count);
        let warmup_rates = measure_full_run(
            &streaming_model,
            &qwen3_5_config,
            &prompt_token_ids[..WARMUP_PROMPT_TOKEN_COUNT],
            WARMUP_OUTPUT_TOKEN_COUNT,
        );
        eprintln!(
            "[dense-speed] warmup kept={kept_layer_count} prefill_tok_s={:.1} decode_tok_s={:.2}",
            warmup_rates.prefill_tokens_per_second, warmup_rates.decode_tokens_per_second
        );
        let full_run_rates = measure_full_run(
            &streaming_model,
            &qwen3_5_config,
            &prompt_token_ids,
            OUTPUT_TOKEN_COUNT,
        );
        eprintln!(
            "[dense-speed] full_run kept={kept_layer_count} prompt_tokens={} output_tokens={OUTPUT_TOKEN_COUNT} prefill_seconds={:.2} prefill_tok_s={:.1} decode_tok_s={:.2} progress={}/{}",
            prompt_token_ids.len(),
            full_run_rates.prefill_seconds,
            full_run_rates.prefill_tokens_per_second,
            full_run_rates.decode_tokens_per_second,
            kept_count_position + 1,
            KEPT_LAYER_COUNTS.len()
        );
    }
    let sweep_count = cold_file_cache.finish();
    eprintln!("[dense-speed] status=success cold_file_cache_sweeps={sweep_count}");
}

fn load_streaming_model(
    model_directory: &std::path::Path,
    machine_mlx_memory_ceiling_bytes: usize,
) -> (Qwen3_5StreamingModel, Qwen3_5Config) {
    let mlx_memory_limits = astronomical_runtime_integration::MlxMemoryLimits::new(
        machine_mlx_memory_ceiling_bytes,
        machine_mlx_memory_ceiling_bytes,
    )
    .expect("the machine MLX memory limits should be valid");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, OUTPUT_TOKEN_COUNT as u32)
        .expect("the streamed dense artifact should validate");
    let qwen3_5_config = validated_artifact.config().clone();
    let streaming_model = Qwen3_5StreamingModel::load(
        MlxRuntime::initialize(mlx_memory_limits).expect("the runtime should initialize"),
        validated_artifact,
        model_directory,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the dense model should load on the streaming engine");
    streaming_model
        .materialize_target_weights()
        .expect("the model weights should materialize");
    (streaming_model, qwen3_5_config)
}

fn pin_kept_layers(streaming_model: &Qwen3_5StreamingModel, kept_layer_count: usize) {
    streaming_model
        .plan_dense_layer_residency(
            DenseLayerProbeShape {
                prefill_chunk_token_count: PREFILL_SPAN_TOKEN_COUNT,
                full_attention_kv_state_growth_tokens: 256,
            },
            Some(kept_layer_count),
        )
        .expect("the dense layers should plan");
}

/// Step 1: cold reads of the last layers, which no run keeps.
fn measure_raw_layer_read_rate(
    streaming_model: &Qwen3_5StreamingModel,
    qwen3_5_config: &Qwen3_5Config,
) {
    pin_kept_layers(streaming_model, 0);
    let layer_count = qwen3_5_config.layer_count() as usize;
    let mut read_bytes = 0_u64;
    let read_started_at = Instant::now();
    for layer_index in layer_count - RAW_READ_LAYER_COUNT..layer_count {
        read_bytes += streaming_model
            .read_dense_layer_for_measurement(layer_index)
            .expect("the layer should read");
    }
    let read_seconds = read_started_at.elapsed().as_secs_f64();
    eprintln!(
        "[dense-speed] raw_layer_read layers={RAW_READ_LAYER_COUNT} bytes={read_bytes} seconds={read_seconds:.3} gb_per_second={:.2} ms_per_layer={:.1}",
        read_bytes as f64 / read_seconds / 1e9,
        read_seconds * 1000.0 / RAW_READ_LAYER_COUNT as f64
    );
}

/// Step 2: one linear-attention and one full-attention layer, read and run
/// over a prompt span and over one decode token.
fn measure_one_streamed_layer(
    streaming_model: &Qwen3_5StreamingModel,
    qwen3_5_config: &Qwen3_5Config,
) {
    let first_full_attention_layer_index = (0..qwen3_5_config.layer_count() as usize)
        .find(|layer_index| qwen3_5_config.decoder_layer_is_full_attention(*layer_index))
        .expect("the model should have a full-attention layer");
    for layer_index in [
        first_full_attention_layer_index
            .checked_sub(1)
            .expect("a linear-attention layer should precede the first full-attention layer"),
        first_full_attention_layer_index,
    ] {
        for token_count in [PREFILL_SPAN_TOKEN_COUNT, 1] {
            // The first run compiles this shape's kernels; the second, with
            // the layer's bytes evicted again by the sweeper, is the cost.
            streaming_model
                .run_streamed_dense_layer_for_measurement(layer_index, token_count)
                .expect("the streamed layer should run");
            std::thread::sleep(COLD_FILE_CACHE_SETTLE);
            let layer_started_at = Instant::now();
            streaming_model
                .run_streamed_dense_layer_for_measurement(layer_index, token_count)
                .expect("the streamed layer should run");
            eprintln!(
                "[dense-speed] one_streamed_layer layer={layer_index} full_attention={} tokens={token_count} ms={:.1}",
                qwen3_5_config.decoder_layer_is_full_attention(layer_index),
                layer_started_at.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}

struct FullRunRates {
    prefill_seconds: f64,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
}

/// Step 3: one prompt prefilled layer by layer, then greedy decode.
fn measure_full_run(
    streaming_model: &Qwen3_5StreamingModel,
    qwen3_5_config: &Qwen3_5Config,
    prompt_token_ids: &[u32],
    output_token_count: usize,
) -> FullRunRates {
    let mut request_decoder_state = crate::common::standard_request_decoder_state(qwen3_5_config);
    let prefill_started_at = Instant::now();
    let mut final_logits = streaming_model
        .prefill_prompt_layer_major(
            prompt_token_ids,
            0,
            PREFILL_SPAN_TOKEN_COUNT,
            &mut request_decoder_state,
        )
        .expect("the layer-major prefill should run")
        .to_vec_f32()
        .expect("the prefill logits should be readable");
    let prefill_seconds = prefill_started_at.elapsed().as_secs_f64();
    let decode_started_at = Instant::now();
    let mut last_progress_at = Instant::now();
    for decode_step in 0..output_token_count {
        let next_token_id = highest_logit_token_id(&final_logits);
        let position = u32::try_from(prompt_token_ids.len() + decode_step)
            .expect("the decode position should fit u32");
        final_logits = streaming_model
            .forward_chunk(&[next_token_id], position, &mut request_decoder_state)
            .expect("the decode step should run")
            .to_vec_f32()
            .expect("the decode logits should be readable");
        if last_progress_at.elapsed() >= PROGRESS_INTERVAL {
            report_decode_progress(decode_step + 1, output_token_count, decode_started_at);
            last_progress_at = Instant::now();
        }
    }
    FullRunRates {
        prefill_seconds,
        prefill_tokens_per_second: prompt_token_ids.len() as f64 / prefill_seconds,
        decode_tokens_per_second: output_token_count as f64
            / decode_started_at.elapsed().as_secs_f64(),
    }
}

fn report_decode_progress(
    decoded_token_count: usize,
    output_token_count: usize,
    decode_started_at: Instant,
) {
    let elapsed_seconds = decode_started_at.elapsed().as_secs_f64();
    let seconds_per_token = elapsed_seconds / decoded_token_count as f64;
    eprintln!(
        "[dense-speed] status=progress decoded={decoded_token_count}/{output_token_count} percentage={:.0} eta_seconds={:.0}",
        decoded_token_count as f64 * 100.0 / output_token_count as f64,
        seconds_per_token * (output_token_count - decoded_token_count) as f64
    );
}

fn highest_logit_token_id(final_logits: &[f32]) -> u32 {
    let mut highest_token_id = 0;
    for (token_id, logit) in final_logits.iter().enumerate() {
        if *logit > final_logits[highest_token_id] {
            highest_token_id = token_id;
        }
    }
    u32::try_from(highest_token_id).expect("a vocabulary index should fit u32")
}

/// Exactly 10,000 Shakespeare tokens.
fn shakespeare_prompt_token_ids(model_directory: &std::path::Path) -> Vec<u32> {
    let tokenizer = tokenizers::Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the dense model's tokenizer should load");
    let mut prompt_token_ids = tokenizer
        .encode(romeo_and_juliet::summary_prompt(PROMPT_WORD_COUNT), false)
        .expect("the Shakespeare prompt should encode")
        .get_ids()
        .to_vec();
    assert!(
        prompt_token_ids.len() >= PROMPT_TOKEN_COUNT,
        "the Shakespeare prompt must reach {PROMPT_TOKEN_COUNT} tokens"
    );
    prompt_token_ids.truncate(PROMPT_TOKEN_COUNT);
    prompt_token_ids
}
