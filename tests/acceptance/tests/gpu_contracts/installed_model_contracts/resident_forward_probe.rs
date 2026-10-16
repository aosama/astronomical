//! Lowest-layer forward probe for the resident sparse-MoE artifact.
//!
//! The daemon served the large sparse-MoE artifact at 0.25 decode tok/s and
//! ~24 prefill tok/s on the reference machine while attributing essentially
//! the whole cost to graph evaluation (graph build near zero). This journey
//! localizes that cost by driving `Qwen3_5ResidentModel` directly:
//! artifact validation, weight materialization, and raw prefill/decode
//! forwards — no daemon, no worker IPC, no memory admission, no chunk
//! planner, no persistent prompt cache, no streaming.
//!
//! Verdict rule: if the crawl reproduces at this layer, the degradation
//! lives at or below the raw model forward (weight residency and paging);
//! if this layer is healthy, the degradation lives in a layer above and
//! the probe moves upward through the stack.
//!
//! Run with the development daemon stopped: its resident model or wired
//! claim competes for the same physical memory and contaminates the
//! measurement. The measured prefill chunk is fixed at the repository's
//! standard 2,048 tokens; the warmup exercises first-use JIT kernels; the
//! decode phase times single-token forwards through the vocabulary head.
//! The first healthy capture on the reference machine becomes the floor
//! this probe asserts; until that capture exists the journey records
//! rates and asserts only structural invariants.
//!
//! Loading a 35-billion-parameter artifact from disk is the repository's
//! documented exception to the 60-second test boundary (same as the
//! Ornith load journey).

use std::time::{Duration, Instant};

use astronomical_config::AstronomicalConfig;
use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5ModelChunkingConfiguration, Qwen3_5ResidentModel,
    Qwen3_5Tokenizer,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;
use tokio::time::timeout;

const PROBE_NAME: &str = "resident-forward-probe";

const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_10000_tokens_romeo_and_juliet.txt");

const WARMUP_TOKEN_COUNT: usize = 16;

const MEASURED_PREFILL_TOKEN_COUNT: usize = 2_048;

const DECODE_STEP_COUNT: usize = 12;

const MAXIMUM_OUTPUT_TOKENS: u32 = 20_480;

/// Matches the incident worker: both MLX process limits at the effective
/// 36 GB (sticker scale) ceiling the reference machine served with.
const INCIDENT_EFFECTIVE_MLX_CEILING_BYTES: usize = 36 * 1_073_741_824;

/// The production standard worker chunking tuple the daemon resolves for
/// the resident engine; the raw forwards must run with the same shapes.
const FULL_ATTENTION_KV_GROWTH_TOKENS: u32 = 256;
const PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 0;
const SSD_PAGING_PREFILL_LAYER_INTERVAL: u32 = 1;
const SSD_PAGING_GENERATION_LAYER_INTERVAL: u32 = 3;

const PROBE_TIMEOUT_SECONDS: u64 = 240;

#[tokio::test]
#[ignore = "loads the large resident sparse-MoE artifact and drives raw model forwards to localize the serving crawl"]
#[serial]
async fn should_drive_raw_resident_model_prefill_and_decode_forwards_measurably() {
    timeout(
        Duration::from_secs(PROBE_TIMEOUT_SECONDS),
        drive_raw_resident_model_forwards(),
    )
    .await
    .expect("the raw resident forward probe must finish inside its documented real-model bound");
}

async fn drive_raw_resident_model_forwards() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    let development_config = AstronomicalConfig::load_from_development_location()
        .expect("the Development configuration should load for the raw forward probe");
    let model_id = crate::common::large_sparse_moe_model_id();
    let model_directory =
        crate::common::configured_discovered_model_by_id(&development_config, model_id)
            .model_directory;
    eprintln!("[{PROBE_NAME}] status=start phase=artifact_validation model={model_id}");

    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, MAXIMUM_OUTPUT_TOKENS)
        .unwrap_or_else(|validation_error| {
            panic!(
                "[{PROBE_NAME}] the artifact should pass production validation: {validation_error}"
            )
        });
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the tokenizer should load from the validated artifact");
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(&tokenizer);
    assert!(
        prompt_token_ids.len() >= WARMUP_TOKEN_COUNT + MEASURED_PREFILL_TOKEN_COUNT,
        "the Romeo and Juliet fixture must tokenize to at least one warmup plus one measured chunk"
    );

    let model_chunking = Qwen3_5ModelChunkingConfiguration::new(
        FULL_ATTENTION_KV_GROWTH_TOKENS,
        PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
        SSD_PAGING_PREFILL_LAYER_INTERVAL,
        SSD_PAGING_GENERATION_LAYER_INTERVAL,
    )
    .expect("the production standard chunking tuple should be valid");
    let mlx_memory_limits = MlxMemoryLimits::new(
        INCIDENT_EFFECTIVE_MLX_CEILING_BYTES,
        INCIDENT_EFFECTIVE_MLX_CEILING_BYTES,
    )
    .expect("the incident MLX memory limits should be valid");
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the MLX runtime should initialize under the incident limits");

    eprintln!("[{PROBE_NAME}] status=progress phase=model_load");
    let load_started_at = Instant::now();
    let mut model = Qwen3_5ResidentModel::load(
        runtime,
        validated_artifact,
        &model_directory,
        model_chunking,
    )
    .expect("the resident model should bind from the validated artifact");
    model
        .materialize_complete_weights_matching_production()
        .expect("the resident model should materialize its complete weights");
    eprintln!(
        "[{PROBE_NAME}] status=success phase=model_load elapsed_ms={}",
        load_started_at.elapsed().as_millis()
    );

    let kv_growth_tokens =
        i32::try_from(FULL_ATTENTION_KV_GROWTH_TOKENS).expect("the growth tokens fit i32");

    // Warmup: first-use JIT kernel compilation must not inflate the
    // measured chunk.
    let mut warmup_state = model
        .empty_request_decoder_state_matching_loaded_weights(kv_growth_tokens)
        .expect("the warmup decoder state should allocate");
    let warmup_started_at = Instant::now();
    model
        .prefill_chunk(
            &prompt_token_ids[..WARMUP_TOKEN_COUNT],
            0,
            &mut warmup_state,
        )
        .expect("the warmup prefill forward should complete");
    eprintln!(
        "[{PROBE_NAME}] status=success phase=warmup_prefill tokens={WARMUP_TOKEN_COUNT} elapsed_ms={}",
        warmup_started_at.elapsed().as_millis()
    );

    // Measured prefill: one fixed 2,048-token Romeo and Juliet chunk, the
    // repository's standard prompt-processing chunk size.
    let mut generation_state = model
        .empty_request_decoder_state_matching_loaded_weights(kv_growth_tokens)
        .expect("the measured decoder state should allocate");
    let prefill_started_at = Instant::now();
    model
        .prefill_chunk(
            &prompt_token_ids[..MEASURED_PREFILL_TOKEN_COUNT],
            0,
            &mut generation_state,
        )
        .expect("the measured prefill forward should complete");
    let prefill_elapsed = prefill_started_at.elapsed();
    eprintln!(
        "[{PROBE_NAME}] MEASURED phase=prefill tokens={MEASURED_PREFILL_TOKEN_COUNT} elapsed_ms={} prefill_tok_per_second={:.1}",
        prefill_elapsed.as_millis(),
        MEASURED_PREFILL_TOKEN_COUNT as f64 / prefill_elapsed.as_secs_f64()
    );

    // Decode: single-token forwards through the vocabulary head. A fixed
    // next-token id keeps every step's graph identical; decode cost is
    // weight-dominated, so the choice does not affect timing.
    let fixed_decode_token_id = prompt_token_ids[MEASURED_PREFILL_TOKEN_COUNT];
    let vocabulary_size = tokenizer.model_vocabulary_size();
    let mut decode_step_millis = Vec::with_capacity(DECODE_STEP_COUNT);
    for decode_step_index in 0..DECODE_STEP_COUNT {
        let decode_position = u32::try_from(MEASURED_PREFILL_TOKEN_COUNT + decode_step_index)
            .expect("the decode position stays inside u32");
        let decode_started_at = Instant::now();
        let logits = model
            .forward_chunk(
                std::slice::from_ref(&fixed_decode_token_id),
                decode_position,
                &mut generation_state,
            )
            .expect("the decode forward should complete");
        logits
            .evaluate()
            .expect("the decode logits should evaluate");
        decode_step_millis.push(decode_started_at.elapsed().as_millis());
        if decode_step_index == 0 {
            let logits_shape = logits.shape();
            assert_eq!(
                logits_shape.last().copied(),
                Some(i32::try_from(vocabulary_size).expect("the vocabulary size fits i32")),
                "the first decode logits must come from the real vocabulary head"
            );
        }
    }
    let total_decode_millis: f64 = decode_step_millis.iter().map(|millis| *millis as f64).sum();
    let total_decode_seconds = total_decode_millis / 1_000.0;
    eprintln!(
        "[{PROBE_NAME}] MEASURED phase=decode steps={DECODE_STEP_COUNT} total_elapsed_ms={total_decode_millis:.0} step_millis={:?} decode_tok_per_second={:.2}",
        decode_step_millis,
        DECODE_STEP_COUNT as f64 / total_decode_seconds
    );
    eprintln!(
        "[{PROBE_NAME}] status=success verdict_inputs=prefill_elapsed_ms:{},decode_total_ms:{total_decode_millis:.0}",
        prefill_elapsed.as_millis()
    );
}

/// Renders and encodes a bounded Romeo and Juliet prompt through the
/// artifact's own tokenizer, matching the chat shape the daemon serves.
fn romeo_and_juliet_prompt_token_ids(tokenizer: &Qwen3_5Tokenizer) -> Vec<u32> {
    let rendered_prompt = format!(
        "<|im_start|>user\nUse this Romeo and Juliet source:\n{ROMEO_AND_JULIET_SOURCE}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    tokenizer
        .encode_prompt(&rendered_prompt)
        .expect("the rendered Romeo and Juliet prompt should encode")
}
