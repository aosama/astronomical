//! A streamed dense prompt forward must stay within what the plan reserved
//! for it (issue #1985).
//!
//! The dense layer plan keeps layers beside a reserve for one streamed
//! forward: two layer slots and the working memory measured on one layer at
//! the engine's prefill chunk. Anything a real forward holds beyond that,
//! other than the decoder state it writes, which admission charges as
//! context, overruns the limit the plan was made for. Two such terms were
//! found live: kept layers joining one graph, and each linear-attention
//! layer's lazy convolution state pinning its whole input until the end of
//! the forward. Either one more than doubled the peak at 4,096 tokens.

use std::time::Duration;

use astronomical_model_serving::{
    DenseLayerProbeShape, Qwen3_5ArtifactValidator, Qwen3_5StreamingModel,
};
use astronomical_runtime_integration::MlxRuntime;
use serial_test::serial;
use tokio::time::timeout;

#[path = "../../support/romeo_and_juliet.rs"]
mod romeo_and_juliet;

/// The engine's SSD-streaming prefill chunk, which the probe measures at.
const PROBE_CHUNK_TOKEN_COUNT: usize = 4096;
/// Enough Shakespeare words for more than one probe-sized chunk.
const PROMPT_WORD_COUNT: usize = 3400;
/// Every layer streamed, and some kept so kept layers precede streamed ones.
const KEPT_LAYER_COUNTS: [usize; 2] = [0, 7];
/// Measured forwards as `(preceding tokens, chunk tokens)`: a full
/// probe-sized chunk, a shorter whole prompt, and a second chunk that
/// attends over a full first chunk, since the probe runs at position zero.
const MEASURED_CHUNKS: [(usize, usize); 3] = [
    (0, PROBE_CHUNK_TOKEN_COUNT),
    (0, 1246),
    (PROBE_CHUNK_TOKEN_COUNT, 1024),
];
/// The final normalization and last-token logits, which no layer holds.
const OUTSIDE_LAYER_SLACK_BYTES: u64 = 8 * 1024 * 1024;
/// The streamed_dense model's hidden states are bfloat16.
const HIDDEN_STATE_ELEMENT_BYTES: u64 = 2;
const FORWARD_MEMORY_BUDGET: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "loads the streamed_dense model and measures real prompt forwards"]
#[serial]
async fn should_keep_a_streamed_prompt_forward_within_the_planned_reserve() {
    timeout(
        FORWARD_MEMORY_BUDGET,
        assert_prompt_forwards_stay_within_reserve(),
    )
    .await
    .expect("the dense prompt forward memory journey must finish within its budget");
}

async fn assert_prompt_forwards_stay_within_reserve() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_streamed_dense_model_directory();
    let prompt_token_ids = shakespeare_prompt_token_ids(&model_directory);
    assert!(
        prompt_token_ids.len() >= PROBE_CHUNK_TOKEN_COUNT,
        "the prompt must fill one probe-sized chunk"
    );
    // The kept count is pinned, so the journey runs under the machine's own
    // limit rather than a configured one.
    let machine_mlx_memory_ceiling_bytes =
        crate::common::sample_machine_mlx_memory_ceiling_bytes().await;
    let mlx_memory_limits = astronomical_runtime_integration::MlxMemoryLimits::new(
        machine_mlx_memory_ceiling_bytes,
        machine_mlx_memory_ceiling_bytes,
    )
    .expect("the machine MLX memory limits should be valid");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 64)
        .expect("the streamed dense artifact should validate");
    let qwen3_5_config = validated_artifact.config().clone();
    let streaming_model = Qwen3_5StreamingModel::load(
        MlxRuntime::initialize(mlx_memory_limits).expect("the runtime should initialize"),
        validated_artifact,
        &model_directory,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the dense model should load on the streaming engine");
    streaming_model
        .materialize_target_weights()
        .expect("the model weights should materialize");

    for kept_layer_count in KEPT_LAYER_COUNTS {
        let residency_plan = streaming_model
            .plan_dense_layer_residency(
                DenseLayerProbeShape {
                    prefill_chunk_token_count: PROBE_CHUNK_TOKEN_COUNT,
                    full_attention_kv_state_growth_tokens: 256,
                },
                Some(kept_layer_count),
            )
            .expect("the dense layers should plan")
            .expect("a dense model should have a plan");
        for (preceding_token_count, chunk_token_count) in MEASURED_CHUNKS {
            let forward_growth = measure_chunk_forward_growth(
                &streaming_model,
                &qwen3_5_config,
                &prompt_token_ids[..preceding_token_count + chunk_token_count],
                preceding_token_count,
            );
            eprintln!(
                "[dense-forward-memory] kept={kept_layer_count} preceding={preceding_token_count} tokens={chunk_token_count} growth_beyond_state={forward_growth} reserve={}",
                residency_plan.streamed_forward_reserve_bytes
            );
            assert!(
                forward_growth
                    <= residency_plan.streamed_forward_reserve_bytes + OUTSIDE_LAYER_SLACK_BYTES,
                "a {chunk_token_count}-token chunk after {preceding_token_count} tokens with {kept_layer_count} kept layers grew {forward_growth} bytes beyond its state; the plan reserved {}",
                residency_plan.streamed_forward_reserve_bytes
            );
        }
        assert_layer_major_prefill_stays_within_reserve(
            &streaming_model,
            &qwen3_5_config,
            &prompt_token_ids,
            residency_plan.streamed_forward_reserve_bytes,
            kept_layer_count,
        );
    }
}

/// The whole prompt in one layer-major prefill over probe-sized spans holds
/// one span's working memory plus every span's hidden states.
fn assert_layer_major_prefill_stays_within_reserve(
    streaming_model: &Qwen3_5StreamingModel,
    qwen3_5_config: &astronomical_model_serving::Qwen3_5Config,
    prompt_token_ids: &[u32],
    streamed_forward_reserve_bytes: u64,
    kept_layer_count: usize,
) {
    let runtime = streaming_model.runtime();
    let mut request_decoder_state = crate::common::standard_request_decoder_state(qwen3_5_config);
    runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("the allocator cache should clear");
    runtime.reset_peak_memory().expect("the peak should reset");
    let final_logits = streaming_model
        .prefill_prompt_layer_major(
            prompt_token_ids,
            0,
            PROBE_CHUNK_TOKEN_COUNT,
            &mut request_decoder_state,
        )
        .expect("the layer-major prefill should run")
        .to_vec_f32()
        .expect("the final logits should be readable");
    assert!(final_logits.iter().all(|logit| logit.is_finite()));
    let peak_memory_bytes = runtime
        .memory_snapshot()
        .expect("the MLX memory snapshot should be readable")
        .peak_memory_bytes() as u64;
    runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("the allocator cache should clear");
    let active_bytes_with_state = runtime
        .memory_snapshot()
        .expect("the MLX memory snapshot should be readable")
        .active_memory_bytes() as u64;
    let prefill_growth = peak_memory_bytes.saturating_sub(active_bytes_with_state);
    let held_token_count = (prompt_token_ids.len() + PROBE_CHUNK_TOKEN_COUNT) as u64;
    let hidden_state_bytes =
        held_token_count * qwen3_5_config.hidden_size() as u64 * HIDDEN_STATE_ELEMENT_BYTES;
    let allowed_growth =
        streamed_forward_reserve_bytes + hidden_state_bytes + OUTSIDE_LAYER_SLACK_BYTES;
    eprintln!(
        "[dense-forward-memory] kept={kept_layer_count} layer_major_tokens={} growth_beyond_state={prefill_growth} allowed={allowed_growth}",
        prompt_token_ids.len()
    );
    assert!(
        prefill_growth <= allowed_growth,
        "a {}-token layer-major prefill with {kept_layer_count} kept layers grew {prefill_growth} bytes beyond its state; it may hold {allowed_growth}",
        prompt_token_ids.len()
    );
}

/// Peak bytes the chunk after `preceding_token_count` tokens held beyond
/// the weights and the decoder state it leaves behind; the preceding tokens
/// run first, unmeasured, to fill that state.
fn measure_chunk_forward_growth(
    streaming_model: &Qwen3_5StreamingModel,
    qwen3_5_config: &astronomical_model_serving::Qwen3_5Config,
    prompt_token_ids: &[u32],
    preceding_token_count: usize,
) -> u64 {
    let runtime = streaming_model.runtime();
    let mut request_decoder_state = crate::common::standard_request_decoder_state(qwen3_5_config);
    if preceding_token_count > 0 {
        streaming_model
            .forward_chunk(
                &prompt_token_ids[..preceding_token_count],
                0,
                &mut request_decoder_state,
            )
            .expect("the preceding chunk should run");
    }
    runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("the allocator cache should clear");
    runtime.reset_peak_memory().expect("the peak should reset");
    let final_logits = streaming_model
        .forward_chunk(
            &prompt_token_ids[preceding_token_count..],
            u32::try_from(preceding_token_count).expect("the chunk start should fit u32"),
            &mut request_decoder_state,
        )
        .expect("the measured chunk should run")
        .to_vec_f32()
        .expect("the final logits should be readable");
    assert!(final_logits.iter().all(|logit| logit.is_finite()));
    let peak_memory_bytes = runtime
        .memory_snapshot()
        .expect("the MLX memory snapshot should be readable")
        .peak_memory_bytes() as u64;
    runtime
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("the allocator cache should clear");
    let active_bytes_with_state = runtime
        .memory_snapshot()
        .expect("the MLX memory snapshot should be readable")
        .active_memory_bytes() as u64;
    peak_memory_bytes.saturating_sub(active_bytes_with_state)
}

fn shakespeare_prompt_token_ids(model_directory: &std::path::Path) -> Vec<u32> {
    let tokenizer = tokenizers::Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the dense model's tokenizer should load");
    tokenizer
        .encode(romeo_and_juliet::summary_prompt(PROMPT_WORD_COUNT), false)
        .expect("the Shakespeare prompt should encode")
        .get_ids()
        .to_vec()
}
