//! Contracts for the batched trunk on a small synthetic ModernBERT.
//!
//! The model is four layers of width 64 with a local window of 8, so a
//! sequence of a dozen tokens already crosses both the global and the local
//! layers and the window mask. Its weights are drawn from the Romeo and Juliet
//! fixture, quantized on the graphics processor and written to a temporary
//! safetensors file, which is then loaded exactly as a real artifact is.
//!
//! The central contract is that batching is invisible: a row of a padded
//! batch gets the hidden states it gets when it is encoded alone.

use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::serial;

use super::{ModernBertTrunk, ModernBertTrunkError};
use crate::modernbert::configuration::ModernBertConfiguration;
use crate::modernbert::token_batch::PaddedTokenBatch;
use crate::modernbert::weight_layout::{ModernBertWeightLayout, TokenEmbeddingStorage};
use crate::synthetic_weights_test_support::{
    SYNTHETIC_QUANTIZATION_BITS, SYNTHETIC_QUANTIZATION_GROUP_SIZE, SyntheticWeights,
    initialize_test_runtime, largest_difference, load_synthetic_weights,
};

const HIDDEN_SIZE: i32 = 64;
const INTERMEDIATE_SIZE: i32 = 64;
const VOCABULARY_SIZE: i32 = 96;
const LAYER_COUNT: u32 = 4;
const PAD_TOKEN_ID: u32 = 0;
const TENSOR_PREFIX: &str = "encoder";
/// Padding changes which additions the fused attention performs, so a padded
/// row matches its unpadded self to rounding, not bit for bit.
const HIDDEN_STATE_TOLERANCE: f32 = 1e-4;

fn synthetic_configuration() -> ModernBertConfiguration {
    ModernBertConfiguration::from_config_value(&serde_json::json!({
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": LAYER_COUNT,
        "num_attention_heads": 2,
        "max_position_embeddings": 512,
        "local_attention": 8,
        "global_attn_every_n_layers": 3,
        "rope_parameters": {
            "full_attention": { "rope_theta": 160_000.0 },
            "sliding_attention": { "rope_theta": 10_000.0 },
        },
        "layer_norm_eps": 1e-5,
        "pad_token_id": PAD_TOKEN_ID,
        "cls_token_id": 1,
        "sep_token_id": 2,
        "quantization": {
            "group_size": SYNTHETIC_QUANTIZATION_GROUP_SIZE,
            "bits": SYNTHETIC_QUANTIZATION_BITS,
        },
    }))
    .expect("the synthetic geometry is supported")
}

/// Builds the synthetic model's tensors; `token_embedding_storage` picks how
/// the token-embedding table is stored, with identical values either way.
fn synthetic_model_tensors(
    runtime: &MlxRuntime,
    token_embedding_storage: TokenEmbeddingStorage,
) -> Vec<(String, MlxArray)> {
    let mut model_weights = SyntheticWeights::new();
    model_weights.push_quantized(
        runtime,
        &trunk_tensor("embeddings.tok_embeddings"),
        [VOCABULARY_SIZE, HIDDEN_SIZE],
    );
    if token_embedding_storage == TokenEmbeddingStorage::Dense {
        replace_quantized_embeddings_with_their_dense_values(runtime, &mut model_weights);
    }
    push_norm(runtime, &mut model_weights, "embeddings.norm.weight");
    for layer_index in 0..LAYER_COUNT {
        if layer_index > 0 {
            push_norm(
                runtime,
                &mut model_weights,
                &format!("layers.{layer_index}.attn_norm.weight"),
            );
        }
        model_weights.push_quantized(
            runtime,
            &trunk_tensor(&format!("layers.{layer_index}.attn.Wqkv")),
            [3 * HIDDEN_SIZE, HIDDEN_SIZE],
        );
        model_weights.push_quantized(
            runtime,
            &trunk_tensor(&format!("layers.{layer_index}.attn.Wo")),
            [HIDDEN_SIZE, HIDDEN_SIZE],
        );
        push_norm(
            runtime,
            &mut model_weights,
            &format!("layers.{layer_index}.mlp_norm.weight"),
        );
        model_weights.push_quantized(
            runtime,
            &trunk_tensor(&format!("layers.{layer_index}.mlp.Wi")),
            [2 * INTERMEDIATE_SIZE, HIDDEN_SIZE],
        );
        model_weights.push_quantized(
            runtime,
            &trunk_tensor(&format!("layers.{layer_index}.mlp.Wo")),
            [HIDDEN_SIZE, INTERMEDIATE_SIZE],
        );
    }
    push_norm(runtime, &mut model_weights, "final_norm.weight");
    model_weights.named_tensors
}

fn trunk_tensor(tensor_suffix: &str) -> String {
    format!("{TENSOR_PREFIX}.{tensor_suffix}")
}

/// LayerNorm weights sit near one in a trained model.
fn push_norm(runtime: &MlxRuntime, model_weights: &mut SyntheticWeights, tensor_suffix: &str) {
    model_weights.push_dense(runtime, &trunk_tensor(tensor_suffix), &[HIDDEN_SIZE], 1.0);
}

/// Swaps the three quantized embedding tensors for one dense table holding
/// exactly the values they dequantize to, so both storages mean the same model.
fn replace_quantized_embeddings_with_their_dense_values(
    runtime: &MlxRuntime,
    model_weights: &mut SyntheticWeights,
) {
    let biases = model_weights.named_tensors.pop().expect("biases").1;
    let scales = model_weights.named_tensors.pop().expect("scales").1;
    let packed_weight = model_weights.named_tensors.pop().expect("weight").1;
    let dense_table = runtime
        .dequantize_affine(
            &packed_weight,
            &scales,
            &biases,
            SYNTHETIC_QUANTIZATION_GROUP_SIZE,
            SYNTHETIC_QUANTIZATION_BITS,
        )
        .expect("the packed table should dequantize");
    model_weights.named_tensors.push((
        format!("{TENSOR_PREFIX}.embeddings.tok_embeddings.weight"),
        dense_table,
    ));
}

/// Hidden states of every row, each cut to its real length: `[row][token * width]`.
fn encode_rows(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    token_embedding_storage: TokenEmbeddingStorage,
    token_sequences: &[&[u32]],
) -> Vec<Vec<f32>> {
    let configuration = synthetic_configuration();
    let weight_layout = ModernBertWeightLayout::new(TENSOR_PREFIX, token_embedding_storage);
    let trunk = ModernBertTrunk {
        runtime,
        tensors,
        configuration: &configuration,
        weight_layout: &weight_layout,
    };
    let token_batch = PaddedTokenBatch::from_sequences(token_sequences, PAD_TOKEN_ID)
        .expect("the test sequences form a batch");
    let hidden_states = trunk
        .encode_hidden_states(&token_batch)
        .expect("the trunk should encode the batch");
    assert_eq!(
        hidden_states.shape(),
        vec![
            token_sequences.len() as i32,
            token_batch.padded_length() as i32,
            HIDDEN_SIZE
        ]
    );
    let flat_hidden_states = runtime
        .astype(&hidden_states, MlxDtype::Float32)
        .expect("the hidden states should cast")
        .to_vec_f32()
        .expect("the hidden states should evaluate");
    let row_width = token_batch.padded_length() * HIDDEN_SIZE as usize;
    let mut row_hidden_states = Vec::with_capacity(token_sequences.len());
    for (row_index, token_sequence) in token_sequences.iter().enumerate() {
        let row_start = row_index * row_width;
        let real_width = token_sequence.len() * HIDDEN_SIZE as usize;
        row_hidden_states.push(flat_hidden_states[row_start..row_start + real_width].to_vec());
    }
    row_hidden_states
}

/// Token identifiers 3, 4, 5, … of the requested length; none is the padding identifier.
fn token_sequence(token_count: usize, first_token_id: u32) -> Vec<u32> {
    let mut token_ids = Vec::with_capacity(token_count);
    for token_index in 0..token_count {
        token_ids.push(first_token_id + (token_index as u32 % 80));
    }
    token_ids
}

#[test]
#[serial]
fn should_give_each_row_of_a_padded_batch_the_hidden_states_it_gets_alone() {
    let runtime = initialize_test_runtime();
    let token_embedding_storage = TokenEmbeddingStorage::AffineQuantized;
    let tensors = load_synthetic_weights(
        &runtime,
        &synthetic_model_tensors(&runtime, token_embedding_storage),
    );
    // 3 tokens fit inside the window, 12 cross it, and 40 leave padded rows
    // whose whole window is padding.
    let short_sequence = token_sequence(3, 3);
    let medium_sequence = token_sequence(12, 7);
    let long_sequence = token_sequence(40, 11);
    let batched_rows = encode_rows(
        &runtime,
        &tensors,
        token_embedding_storage,
        &[&short_sequence, &medium_sequence, &long_sequence],
    );
    for (row_index, token_sequence) in [&short_sequence, &medium_sequence, &long_sequence]
        .into_iter()
        .enumerate()
    {
        let encoded_alone = encode_rows(
            &runtime,
            &tensors,
            token_embedding_storage,
            &[token_sequence],
        );
        assert!(
            batched_rows[row_index]
                .iter()
                .all(|hidden_value| hidden_value.is_finite()),
            "row {row_index} must stay finite beside padding"
        );
        let row_difference = largest_difference(&batched_rows[row_index], &encoded_alone[0]);
        assert!(
            row_difference <= HIDDEN_STATE_TOLERANCE,
            "row {row_index} differs from its unpadded self by {row_difference:e}"
        );
    }
}

#[test]
#[serial]
fn should_give_rows_of_an_unpadded_batch_the_hidden_states_they_get_alone() {
    let runtime = initialize_test_runtime();
    let token_embedding_storage = TokenEmbeddingStorage::AffineQuantized;
    let tensors = load_synthetic_weights(
        &runtime,
        &synthetic_model_tensors(&runtime, token_embedding_storage),
    );
    let first_sequence = token_sequence(12, 3);
    let second_sequence = token_sequence(12, 30);
    let batched_rows = encode_rows(
        &runtime,
        &tensors,
        token_embedding_storage,
        &[&first_sequence, &second_sequence],
    );
    let first_alone = encode_rows(
        &runtime,
        &tensors,
        token_embedding_storage,
        &[&first_sequence],
    );
    let second_alone = encode_rows(
        &runtime,
        &tensors,
        token_embedding_storage,
        &[&second_sequence],
    );
    assert!(largest_difference(&batched_rows[0], &first_alone[0]) <= HIDDEN_STATE_TOLERANCE);
    assert!(largest_difference(&batched_rows[1], &second_alone[0]) <= HIDDEN_STATE_TOLERANCE);
    assert!(
        largest_difference(&batched_rows[0], &batched_rows[1]) > HIDDEN_STATE_TOLERANCE,
        "different tokens must produce different hidden states"
    );
}

#[test]
#[serial]
fn should_hide_padding_when_every_row_is_inside_the_local_window() {
    let runtime = initialize_test_runtime();
    let token_embedding_storage = TokenEmbeddingStorage::AffineQuantized;
    let tensors = load_synthetic_weights(
        &runtime,
        &synthetic_model_tensors(&runtime, token_embedding_storage),
    );
    // Five tokens is the longest sequence the window of 8 never masks, so
    // this batch needs the padding mask and no window mask.
    let shorter_sequence = token_sequence(2, 3);
    let longer_sequence = token_sequence(5, 20);
    let batched_rows = encode_rows(
        &runtime,
        &tensors,
        token_embedding_storage,
        &[&shorter_sequence, &longer_sequence],
    );
    let shorter_alone = encode_rows(
        &runtime,
        &tensors,
        token_embedding_storage,
        &[&shorter_sequence],
    );
    let longer_alone = encode_rows(
        &runtime,
        &tensors,
        token_embedding_storage,
        &[&longer_sequence],
    );
    assert!(largest_difference(&batched_rows[0], &shorter_alone[0]) <= HIDDEN_STATE_TOLERANCE);
    assert!(largest_difference(&batched_rows[1], &longer_alone[0]) <= HIDDEN_STATE_TOLERANCE);
}

#[test]
#[serial]
fn should_encode_a_single_token_sequence() {
    let runtime = initialize_test_runtime();
    let token_embedding_storage = TokenEmbeddingStorage::AffineQuantized;
    let tensors = load_synthetic_weights(
        &runtime,
        &synthetic_model_tensors(&runtime, token_embedding_storage),
    );
    let encoded_rows = encode_rows(&runtime, &tensors, token_embedding_storage, &[&[5]]);
    assert_eq!(encoded_rows[0].len(), HIDDEN_SIZE as usize);
    assert!(
        encoded_rows[0]
            .iter()
            .all(|hidden_value| hidden_value.is_finite())
    );
}

#[test]
#[serial]
fn should_read_a_dense_embedding_table_like_the_quantized_one_it_equals() {
    let runtime = initialize_test_runtime();
    let quantized_tensors = load_synthetic_weights(
        &runtime,
        &synthetic_model_tensors(&runtime, TokenEmbeddingStorage::AffineQuantized),
    );
    let dense_tensors = load_synthetic_weights(
        &runtime,
        &synthetic_model_tensors(&runtime, TokenEmbeddingStorage::Dense),
    );
    let sequence = token_sequence(12, 3);
    let from_quantized = encode_rows(
        &runtime,
        &quantized_tensors,
        TokenEmbeddingStorage::AffineQuantized,
        &[&sequence],
    );
    let from_dense = encode_rows(
        &runtime,
        &dense_tensors,
        TokenEmbeddingStorage::Dense,
        &[&sequence],
    );
    let storage_difference = largest_difference(&from_quantized[0], &from_dense[0]);
    assert!(
        storage_difference <= HIDDEN_STATE_TOLERANCE,
        "the two embedding storages differ by {storage_difference:e}"
    );
}

#[test]
#[serial]
fn should_name_the_tensor_an_artifact_is_missing() {
    let runtime = initialize_test_runtime();
    let token_embedding_storage = TokenEmbeddingStorage::AffineQuantized;
    let mut named_tensors = synthetic_model_tensors(&runtime, token_embedding_storage);
    named_tensors.retain(|(tensor_name, _)| tensor_name != "encoder.final_norm.weight");
    let tensors = load_synthetic_weights(&runtime, &named_tensors);
    let configuration = synthetic_configuration();
    let weight_layout = ModernBertWeightLayout::new(TENSOR_PREFIX, token_embedding_storage);
    let trunk = ModernBertTrunk {
        runtime: &runtime,
        tensors: &tensors,
        configuration: &configuration,
        weight_layout: &weight_layout,
    };
    let token_batch =
        PaddedTokenBatch::from_sequences(&[&[3, 4]], PAD_TOKEN_ID).expect("a valid batch");
    let Err(ModernBertTrunkError::UnavailableTensor { tensor_name, .. }) =
        trunk.encode_hidden_states(&token_batch)
    else {
        panic!("a missing tensor must be reported by name");
    };
    assert_eq!(tensor_name, "encoder.final_norm.weight");
}
