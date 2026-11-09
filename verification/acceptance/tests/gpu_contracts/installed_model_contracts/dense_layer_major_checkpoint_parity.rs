//! A layer-major prefill must publish the prompt-cache state the chunk path
//! publishes, bit for bit (issue #1985).
//!
//! The engine prefills a dense model that streams layers one layer at a time
//! over the spans it would otherwise run as chunks, and publishes a block at
//! every 1,792-token boundary from the state that span leaves. This journey
//! prefills one Shakespeare prompt both ways from fresh state, over the
//! engine's own span plan, with every layer streamed and with eight kept:
//! the published boundaries must fall at the same positions and hold
//! identical tensors. The plan covers a boundary inside a span (captured by
//! the span's collector) and one at a span's end (captured from its state).

use std::time::Duration;

use astronomical_model_serving::{
    DenseLayerProbeShape, Qwen3_5ArtifactValidator, Qwen3_5StreamingModel,
};
use astronomical_runtime_integration::MlxRuntime;
use serial_test::serial;
use tokio::time::timeout;

#[path = "../../support/romeo_and_juliet.rs"]
mod romeo_and_juliet;

/// The engine's SSD-streaming chunk and the 9B model's cache block.
const CHUNK_TOKEN_COUNT: usize = 4096;
const BLOCK_TOKEN_COUNT: usize = 1792;
/// About 4,600 tokens: two spans, with boundaries at 1,792 and 3,584.
const PROMPT_WORD_COUNT: usize = 3000;
const KEPT_LAYER_COUNTS: [usize; 2] = [0, 8];
const CHECKPOINT_PARITY_BUDGET: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "loads the streamed_dense model and compares published prompt-cache state"]
#[serial]
async fn should_publish_the_chunk_path_prompt_cache_state_from_a_layer_major_prefill() {
    timeout(CHECKPOINT_PARITY_BUDGET, assert_checkpoint_parity())
        .await
        .expect("the checkpoint parity journey must finish within its budget");
}

async fn assert_checkpoint_parity() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_streamed_dense_model_directory();
    let tokenizer = tokenizers::Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the dense model's tokenizer should load");
    let prompt_token_ids = tokenizer
        .encode(romeo_and_juliet::summary_prompt(PROMPT_WORD_COUNT), false)
        .expect("the Shakespeare prompt should encode")
        .get_ids()
        .to_vec();
    assert!(prompt_token_ids.len() > 2 * BLOCK_TOKEN_COUNT);
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
        streaming_model
            .plan_dense_layer_residency(
                DenseLayerProbeShape {
                    prefill_chunk_token_count: CHUNK_TOKEN_COUNT,
                    full_attention_kv_state_growth_tokens: 256,
                },
                Some(kept_layer_count),
            )
            .expect("the dense layers should plan");
        let chunk_path_boundaries = streaming_model
            .prefill_chunk_by_chunk_with_prompt_cache_checkpoints(
                &prompt_token_ids,
                CHUNK_TOKEN_COUNT,
                BLOCK_TOKEN_COUNT,
                &mut crate::common::standard_request_decoder_state(&qwen3_5_config),
            )
            .expect("the chunk path should prefill");
        let layer_major_boundaries = streaming_model
            .prefill_layer_major_with_prompt_cache_checkpoints(
                &prompt_token_ids,
                CHUNK_TOKEN_COUNT,
                BLOCK_TOKEN_COUNT,
                &mut crate::common::standard_request_decoder_state(&qwen3_5_config),
            )
            .expect("the layer-major prefill should run");
        let boundary_positions: Vec<usize> = chunk_path_boundaries
            .iter()
            .map(|(boundary_token, _)| *boundary_token)
            .collect();
        assert_eq!(
            boundary_positions,
            vec![BLOCK_TOKEN_COUNT, 2 * BLOCK_TOKEN_COUNT],
            "the chunk path should publish both boundaries"
        );
        assert!(
            layer_major_boundaries == chunk_path_boundaries,
            "with {kept_layer_count} kept layers the layer-major prefill must publish the chunk path's boundaries bit for bit"
        );
        eprintln!(
            "[dense-checkpoint-parity] kept={kept_layer_count} prompt_tokens={} boundaries={boundary_positions:?} tensors_per_boundary={}",
            prompt_token_ids.len(),
            chunk_path_boundaries[0].1.len()
        );
    }
}
