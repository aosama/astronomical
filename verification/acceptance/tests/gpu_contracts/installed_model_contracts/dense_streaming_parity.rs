//! Serving a dense model from the SSD must not change its answer (issue #1985).
//!
//! The streaming engine serves a dense model whose text weights do not fit
//! the limit: it reads embedding rows from the SSD and streams the decoder
//! layers it does not keep. The resident engine keeps everything in memory.
//! For the same Shakespeare prompt and greedy decode, the streaming engine
//! with every layer streamed, and with half kept, must produce the resident
//! tokens and bit-identical final logits, both when it prefills chunk by
//! chunk and when it runs each layer over every chunk before the next.
//!
//! The resident layer count is pinned rather than planned from the limit, so
//! both models load in this one process, one after the other.

use std::time::Duration;

use astronomical_model_serving::{
    DenseLayerProbeShape, Qwen3_5ArtifactValidator, Qwen3_5Config, Qwen3_5ResidentModel,
    Qwen3_5StreamingModel,
};
use astronomical_runtime_integration::MlxRuntime;
use serial_test::serial;
use tokio::time::timeout;

#[path = "../../support/romeo_and_juliet.rs"]
mod romeo_and_juliet;

/// Words of the Shakespeare fixture: about 1,000 tokens, four prefill chunks
/// of this journey's chunk size, so the comparison crosses chunk boundaries.
const PROMPT_WORD_COUNT: usize = 600;
const OUTPUT_TOKEN_COUNT: u32 = 64;
const PARITY_BUDGET: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "loads the streamed_dense model twice and requires exact final-logit parity"]
#[serial]
async fn should_give_the_resident_logits_when_dense_weights_stream_from_disk() {
    timeout(PARITY_BUDGET, assert_dense_streaming_logit_parity())
        .await
        .expect("the dense streaming parity journey must finish within its budget");
}

async fn assert_dense_streaming_logit_parity() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_streamed_dense_model_directory();
    let prompt_token_ids = shakespeare_prompt_token_ids(&model_directory);
    // The kept layer count is pinned, so the journey runs under the machine's
    // own limit rather than a configured one that might not fit 16 layers.
    let machine_mlx_memory_ceiling_bytes =
        crate::common::sample_machine_mlx_memory_ceiling_bytes().await;
    let mlx_memory_limits = astronomical_runtime_integration::MlxMemoryLimits::new(
        machine_mlx_memory_ceiling_bytes,
        machine_mlx_memory_ceiling_bytes,
    )
    .expect("the machine MLX memory limits should be valid");

    eprintln!("[dense-parity] status=progress step=1/2 engine=streaming");
    let streaming_answer = {
        let (validated_artifact, qwen3_5_config) = validated_dense_artifact(&model_directory);
        let streaming_model = Qwen3_5StreamingModel::load(
            MlxRuntime::initialize(mlx_memory_limits)
                .expect("the streaming runtime should initialize"),
            validated_artifact,
            &model_directory,
            crate::common::standard_qwen3_5_model_chunking_configuration(),
        )
        .expect("the dense model should load on the streaming engine");
        let dense_forward = StreamingForward(&streaming_model);
        let active_bytes_after_load = dense_forward.materialized_active_bytes();
        let mut answers_by_streaming_variant = Vec::with_capacity(KEPT_LAYER_COUNTS.len() * 2);
        for kept_layer_count in KEPT_LAYER_COUNTS {
            streaming_model
                .plan_dense_layer_residency(
                    DenseLayerProbeShape {
                        prefill_chunk_token_count: PREFILL_CHUNK_TOKEN_COUNT,
                        full_attention_kv_state_growth_tokens: PREFILL_CHUNK_TOKEN_COUNT as i32,
                    },
                    Some(kept_layer_count),
                )
                .expect("the dense layers should plan");
            answers_by_streaming_variant.push((
                format!("{kept_layer_count} kept layers, chunk by chunk"),
                greedy_answer(&qwen3_5_config, &prompt_token_ids, &dense_forward),
            ));
            answers_by_streaming_variant.push((
                format!("{kept_layer_count} kept layers, layer by layer"),
                greedy_answer_after_layer_major_prefill(
                    &qwen3_5_config,
                    &prompt_token_ids,
                    &streaming_model,
                ),
            ));
        }
        (active_bytes_after_load, answers_by_streaming_variant)
    };
    eprintln!("[dense-parity] status=progress step=2/2 engine=resident");
    let resident_answer = {
        let (validated_artifact, qwen3_5_config) = validated_dense_artifact(&model_directory);
        let resident_model = Qwen3_5ResidentModel::load(
            MlxRuntime::initialize(mlx_memory_limits)
                .expect("the resident runtime should initialize"),
            validated_artifact,
            &model_directory,
            crate::common::standard_qwen3_5_model_chunking_configuration(),
        )
        .expect("the dense model should load on the resident engine");
        let dense_forward = ResidentForward(&resident_model);
        let active_bytes_after_load = dense_forward.materialized_active_bytes();
        let answer = greedy_answer(&qwen3_5_config, &prompt_token_ids, &dense_forward);
        (active_bytes_after_load, answer)
    };

    let embedding_table_bytes = embedding_table_payload_bytes(&model_directory);
    let saved_active_bytes = resident_answer.0.saturating_sub(streaming_answer.0);
    assert!(
        saved_active_bytes + EMBEDDING_ACCOUNTING_SLACK_BYTES >= embedding_table_bytes,
        "the streaming engine must leave at least the {embedding_table_bytes}-byte embedding table out of memory; it saved {saved_active_bytes}"
    );
    for (streaming_variant, streaming_greedy_answer) in &streaming_answer.1 {
        assert_eq!(
            streaming_greedy_answer.generated_token_ids, resident_answer.1.generated_token_ids,
            "greedy decode with {streaming_variant} must choose the resident tokens"
        );
        assert!(
            streaming_greedy_answer.final_logits == resident_answer.1.final_logits,
            "streaming with {streaming_variant} must give bit-identical final logits"
        );
    }
    eprintln!(
        "[dense-parity] status=success prompt_tokens={} decode_tokens={} bytes_left_out_of_memory={saved_active_bytes}",
        prompt_token_ids.len(),
        resident_answer.1.generated_token_ids.len()
    );
}

/// Bytes the allocator may round a 0.8 GB table's absence by.
const EMBEDDING_ACCOUNTING_SLACK_BYTES: u64 = 16 * 1024 * 1024;
const DECODE_STEP_COUNT: usize = 8;
/// Every layer streamed, and half of the 9B model's 32 layers kept.
const KEPT_LAYER_COUNTS: [usize; 2] = [0, 16];
const PREFILL_CHUNK_TOKEN_COUNT: usize = 256;

/// The tokens a greedy decode chose and the logits of its last step.
struct GreedyAnswer {
    generated_token_ids: Vec<u32>,
    final_logits: Vec<f32>,
}

/// One forward on either engine; the journey runs the same steps on both.
trait DenseForward {
    fn forward_chunk(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut astronomical_model_serving::RequestDecoderStateStack,
    ) -> Result<mlx_c_rust::MlxArray, astronomical_model_serving::Qwen3_5ExecutionError>;

    /// Materializes the model as its engine would and reads MLX active memory.
    fn materialized_active_bytes(&self) -> u64;
}

struct StreamingForward<'model>(&'model Qwen3_5StreamingModel);
struct ResidentForward<'model>(&'model Qwen3_5ResidentModel);

impl DenseForward for StreamingForward<'_> {
    fn forward_chunk(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut astronomical_model_serving::RequestDecoderStateStack,
    ) -> Result<mlx_c_rust::MlxArray, astronomical_model_serving::Qwen3_5ExecutionError> {
        self.0
            .forward_chunk(token_ids, starting_position_tokens, request_decoder_state)
    }

    fn materialized_active_bytes(&self) -> u64 {
        self.0
            .materialize_target_weights()
            .expect("the model weights should materialize");
        self.0
            .runtime()
            .memory_snapshot()
            .expect("the MLX memory snapshot should be readable")
            .active_memory_bytes() as u64
    }
}

impl DenseForward for ResidentForward<'_> {
    fn forward_chunk(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut astronomical_model_serving::RequestDecoderStateStack,
    ) -> Result<mlx_c_rust::MlxArray, astronomical_model_serving::Qwen3_5ExecutionError> {
        self.0
            .forward_chunk(token_ids, starting_position_tokens, request_decoder_state)
    }

    fn materialized_active_bytes(&self) -> u64 {
        self.0
            .materialize_target_weights()
            .expect("the model weights should materialize");
        self.0
            .runtime()
            .memory_snapshot()
            .expect("the MLX memory snapshot should be readable")
            .active_memory_bytes() as u64
    }
}

/// Bytes of every `embed_tokens` tensor, read from the shard headers.
fn embedding_table_payload_bytes(model_directory: &std::path::Path) -> u64 {
    let shard_index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model_directory.join("model.safetensors.index.json"))
            .expect("the shard index should read"),
    )
    .expect("the shard index should parse");
    let mut embedding_shard_names = std::collections::BTreeSet::new();
    for (tensor_name, shard_name) in shard_index["weight_map"]
        .as_object()
        .expect("the shard index should hold a weight map")
    {
        if tensor_name.contains("embed_tokens") {
            embedding_shard_names.insert(shard_name.as_str().unwrap_or_default().to_owned());
        }
    }
    let mut embedding_bytes = 0_u64;
    for shard_name in embedding_shard_names {
        let shard_bytes =
            std::fs::read(model_directory.join(&shard_name)).expect("the shard should read");
        let header_length =
            u64::from_le_bytes(shard_bytes[..8].try_into().expect("a header length")) as usize;
        let header: serde_json::Value = serde_json::from_slice(&shard_bytes[8..8 + header_length])
            .expect("the header should parse");
        for (tensor_name, tensor_entry) in
            header.as_object().expect("the header should be an object")
        {
            if tensor_name.contains("embed_tokens") {
                let data_offsets = &tensor_entry["data_offsets"];
                embedding_bytes +=
                    data_offsets[1].as_u64().unwrap_or(0) - data_offsets[0].as_u64().unwrap_or(0);
            }
        }
    }
    embedding_bytes
}
fn validated_dense_artifact(
    model_directory: &std::path::Path,
) -> (
    astronomical_model_serving::ValidatedQwen3_5Artifact,
    Qwen3_5Config,
) {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, OUTPUT_TOKEN_COUNT)
        .expect("the streamed dense artifact should validate");
    let qwen3_5_config = validated_artifact.config().clone();
    (validated_artifact, qwen3_5_config)
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

/// Prefills the whole prompt layer by layer over the same spans the chunked
/// prefill uses, then decodes greedily: each streamed layer is read once for
/// the prompt, and every kernel still sees the chunked prefill's tokens.
fn greedy_answer_after_layer_major_prefill(
    qwen3_5_config: &Qwen3_5Config,
    prompt_token_ids: &[u32],
    streaming_model: &Qwen3_5StreamingModel,
) -> GreedyAnswer {
    let mut request_decoder_state = crate::common::standard_request_decoder_state(qwen3_5_config);
    let prefill_logits = logits_of(streaming_model.prefill_prompt_layer_major(
        prompt_token_ids,
        0,
        PREFILL_CHUNK_TOKEN_COUNT,
        &mut request_decoder_state,
    ));
    greedy_decode(
        prompt_token_ids.len(),
        prefill_logits,
        &mut request_decoder_state,
        &StreamingForward(streaming_model),
    )
}

/// Prefills the prompt in fixed chunks, then decodes greedily, returning
/// the chosen tokens and the last step's logits.
fn greedy_answer(
    qwen3_5_config: &Qwen3_5Config,
    prompt_token_ids: &[u32],
    dense_forward: &dyn DenseForward,
) -> GreedyAnswer {
    let mut request_decoder_state = crate::common::standard_request_decoder_state(qwen3_5_config);
    let mut final_logits = Vec::new();
    for chunk_start in (0..prompt_token_ids.len()).step_by(PREFILL_CHUNK_TOKEN_COUNT) {
        let chunk_end = (chunk_start + PREFILL_CHUNK_TOKEN_COUNT).min(prompt_token_ids.len());
        final_logits = logits_of(dense_forward.forward_chunk(
            &prompt_token_ids[chunk_start..chunk_end],
            u32::try_from(chunk_start).expect("the chunk start should fit u32"),
            &mut request_decoder_state,
        ));
    }
    greedy_decode(
        prompt_token_ids.len(),
        final_logits,
        &mut request_decoder_state,
        dense_forward,
    )
}

/// Decodes `DECODE_STEP_COUNT` greedy tokens after a prompt of
/// `prompt_token_count` tokens whose last logits are `prefill_logits`.
fn greedy_decode(
    prompt_token_count: usize,
    prefill_logits: Vec<f32>,
    request_decoder_state: &mut astronomical_model_serving::RequestDecoderStateStack,
    dense_forward: &dyn DenseForward,
) -> GreedyAnswer {
    let mut final_logits = prefill_logits;
    let mut generated_token_ids = Vec::with_capacity(DECODE_STEP_COUNT);
    for decode_step in 0..DECODE_STEP_COUNT {
        let next_token_id = highest_logit_token_id(&final_logits);
        generated_token_ids.push(next_token_id);
        let position = prompt_token_count + decode_step;
        final_logits = logits_of(dense_forward.forward_chunk(
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
    forward_result: Result<mlx_c_rust::MlxArray, astronomical_model_serving::Qwen3_5ExecutionError>,
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
