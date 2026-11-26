//! A greedy answer to the Shakespeare prompt from a model loaded on the
//! streaming engine, without the engine: the journeys that change one thing
//! about a loaded model (kept layers in register order, the fused expert
//! combination) compare the answers before and after the change.

use model_serving::{
    LayerProbeShape, Qwen3_5ArtifactValidator, Qwen3_5Config, Qwen3_5StreamingModel,
};
use runtime_integration::MlxRuntime;

#[path = "../../support/romeo_and_juliet.rs"]
mod romeo_and_juliet;

/// Words of the Shakespeare fixture: about 1,000 tokens, four prefill chunks
/// of this journey's chunk size, so the comparison crosses chunk boundaries.
const PROMPT_WORD_COUNT: usize = 600;
const OUTPUT_TOKEN_COUNT: u32 = 64;
const DECODE_STEP_COUNT: usize = 8;
const PREFILL_CHUNK_TOKEN_COUNT: usize = 256;

/// The tokens a greedy decode chose and the logits of its last step.
pub(super) struct GreedyAnswer {
    pub(super) generated_token_ids: Vec<u32>,
    pub(super) final_logits: Vec<f32>,
}

/// Probes the dense layers and keeps exactly `kept_layer_count` of them.
pub(super) fn keep_dense_layers(streaming_model: &Qwen3_5StreamingModel, kept_layer_count: usize) {
    streaming_model
        .probe_dense_layers(
            LayerProbeShape {
                prefill_chunk_token_count: PREFILL_CHUNK_TOKEN_COUNT,
                full_attention_kv_state_growth_tokens: PREFILL_CHUNK_TOKEN_COUNT as i32,
            },
            Some(kept_layer_count),
        )
        .expect("the dense layers should be probed and kept");
}

/// The dense model on the streaming engine under `mlx_memory_limits`, with
/// its configuration.
fn load_streaming_model(
    model_directory: &std::path::Path,
    mlx_memory_limits: runtime_integration::MlxMemoryLimits,
) -> (Qwen3_5StreamingModel, Qwen3_5Config) {
    let (validated_artifact, qwen3_5_config) = validated_artifact(model_directory);
    let streaming_model = Qwen3_5StreamingModel::load(
        MlxRuntime::initialize(mlx_memory_limits).expect("the streaming runtime should initialize"),
        validated_artifact,
        model_directory,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the model should load on the streaming engine");
    (streaming_model, qwen3_5_config)
}

/// A model loaded on the streaming engine under the machine's own limit,
/// with its weights materialized, and its configuration.
pub(super) async fn load_materialized_streaming_model(
    model_directory: &std::path::Path,
) -> (Qwen3_5StreamingModel, Qwen3_5Config) {
    let (streaming_model, qwen3_5_config) =
        load_streaming_model(model_directory, machine_mlx_memory_limits().await);
    streaming_model
        .materialize_target_weights()
        .expect("the model weights should materialize");
    (streaming_model, qwen3_5_config)
}

/// The machine's own MLX limit: these journeys pin how many layers are kept,
/// so they run under it rather than a configured one that might not fit.
async fn machine_mlx_memory_limits() -> runtime_integration::MlxMemoryLimits {
    let machine_mlx_memory_ceiling_bytes =
        crate::common::sample_machine_mlx_memory_ceiling_bytes().await;
    runtime_integration::MlxMemoryLimits::new(
        machine_mlx_memory_ceiling_bytes,
        machine_mlx_memory_ceiling_bytes,
    )
    .expect("the machine MLX memory limits should be valid")
}

fn validated_artifact(
    model_directory: &std::path::Path,
) -> (model_serving::ValidatedQwen3_5Artifact, Qwen3_5Config) {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, OUTPUT_TOKEN_COUNT)
        .expect("the artifact should validate");
    let qwen3_5_config = validated_artifact.config().clone();
    (validated_artifact, qwen3_5_config)
}

pub(super) fn shakespeare_prompt_token_ids(model_directory: &std::path::Path) -> Vec<u32> {
    let tokenizer = tokenizers::Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the model's tokenizer should load");
    tokenizer
        .encode(romeo_and_juliet::summary_prompt(PROMPT_WORD_COUNT), false)
        .expect("the Shakespeare prompt should encode")
        .get_ids()
        .to_vec()
}

/// The streaming model's greedy answer to the prompt: prefilled in fixed
/// chunks, then decoded greedily, returning the chosen tokens and the last
/// step's logits.
pub(super) fn streaming_greedy_answer(
    streaming_model: &Qwen3_5StreamingModel,
    qwen3_5_config: &Qwen3_5Config,
    prompt_token_ids: &[u32],
) -> GreedyAnswer {
    let mut request_decoder_state = crate::common::standard_request_decoder_state(qwen3_5_config);
    let mut final_logits = Vec::new();
    for chunk_start in (0..prompt_token_ids.len()).step_by(PREFILL_CHUNK_TOKEN_COUNT) {
        let chunk_end = (chunk_start + PREFILL_CHUNK_TOKEN_COUNT).min(prompt_token_ids.len());
        final_logits = logits_of(streaming_model.forward_chunk(
            &prompt_token_ids[chunk_start..chunk_end],
            u32::try_from(chunk_start).expect("the chunk start should fit u32"),
            &mut request_decoder_state,
        ));
    }
    greedy_decode(
        prompt_token_ids.len(),
        final_logits,
        &mut request_decoder_state,
        streaming_model,
    )
}

/// Decodes `DECODE_STEP_COUNT` greedy tokens after a prompt of
/// `prompt_token_count` tokens whose last logits are `prefill_logits`.
fn greedy_decode(
    prompt_token_count: usize,
    prefill_logits: Vec<f32>,
    request_decoder_state: &mut model_serving::RequestDecoderStateStack,
    streaming_model: &Qwen3_5StreamingModel,
) -> GreedyAnswer {
    let mut final_logits = prefill_logits;
    let mut generated_token_ids = Vec::with_capacity(DECODE_STEP_COUNT);
    for decode_step in 0..DECODE_STEP_COUNT {
        let next_token_id = highest_logit_token_id(&final_logits);
        generated_token_ids.push(next_token_id);
        let position = prompt_token_count + decode_step;
        final_logits = logits_of(streaming_model.forward_chunk(
            &[next_token_id],
            u32::try_from(position).expect("the decode position should fit u32"),
            request_decoder_state,
        ));
    }
    GreedyAnswer {
        generated_token_ids,
        final_logits,
    }
}

fn logits_of(
    forward_result: Result<mlx_bridge::MlxArray, model_serving::TextExecutionError>,
) -> Vec<f32> {
    let final_logits = forward_result
        .expect("the forward should run")
        .to_vec_f32()
        .expect("the final logits should be readable");
    assert!(final_logits.iter().all(|logit| logit.is_finite()));
    final_logits
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
