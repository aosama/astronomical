//! Contracts for the decision head on small synthetic weights.
//!
//! The head is two layers of width 64 with weights drawn from the Romeo and
//! Juliet fixture. It is fed made-up hidden states directly, so these tests
//! say nothing about answer quality; agreement with the reference
//! implementation on real checkpoints is the acceptance journey's job. What
//! they pin is the head's own bookkeeping: padding is invisible, the question
//! type matters, and every row gets back exactly its own options.

use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};
use mlx_c_rust::MlxArray;
use serial_test::serial;

use super::{HeadPassRows, LayaDecisionHead};
use crate::decision_models::laya::configuration::LayaConfiguration;
use crate::modernbert::token_batch::PaddedTokenBatch;
use crate::modernbert::trunk::ModernBertTrunkError;
use crate::synthetic_weights_test_support::{
    SYNTHETIC_QUANTIZATION_BITS, SYNTHETIC_QUANTIZATION_GROUP_SIZE, SyntheticWeights,
    initialize_test_runtime, largest_difference, load_synthetic_weights, shakespeare_values,
};

const HIDDEN_SIZE: i32 = 64;
const PAD_TOKEN_ID: u32 = 0;
/// Padding changes which additions the fused attention performs, so a padded
/// row matches its unpadded self to rounding, not bit for bit.
const LOGIT_TOLERANCE: f32 = 1e-4;

fn synthetic_configuration(head_layer_count: u32) -> LayaConfiguration {
    LayaConfiguration::from_config_value(&serde_json::json!({
        "encoder": {
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": 2,
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
        },
        "head_layers": head_layer_count,
        "max_len": 64,
        "head_max_len": 24,
        "quantization": {
            "group_size": SYNTHETIC_QUANTIZATION_GROUP_SIZE,
            "bits": SYNTHETIC_QUANTIZATION_BITS,
        },
    }))
    .expect("the synthetic head settings are supported")
}

fn push_norm_pair(runtime: &MlxRuntime, head_weights: &mut SyntheticWeights, norm_prefix: &str) {
    head_weights.push_dense(
        runtime,
        &format!("{norm_prefix}.weight"),
        &[HIDDEN_SIZE],
        1.0,
    );
    head_weights.push_dense(runtime, &format!("{norm_prefix}.bias"), &[HIDDEN_SIZE], 0.0);
}

fn push_quantized_linear(
    runtime: &MlxRuntime,
    head_weights: &mut SyntheticWeights,
    linear_prefix: &str,
    shape: [i32; 2],
) {
    head_weights.push_quantized(runtime, linear_prefix, shape);
    head_weights.push_dense(runtime, &format!("{linear_prefix}.bias"), &[shape[0]], 0.0);
}

fn synthetic_head_tensors(runtime: &MlxRuntime, head_layer_count: u32) -> Vec<(String, MlxArray)> {
    let mut head_weights = SyntheticWeights::new();
    head_weights.push_dense(runtime, "type_emb.weight", &[3, HIDDEN_SIZE], 0.0);
    for head_layer_index in 0..head_layer_count {
        let layer_prefix = format!("head.layers.{head_layer_index}");
        push_norm_pair(runtime, &mut head_weights, &format!("{layer_prefix}.norm1"));
        push_norm_pair(runtime, &mut head_weights, &format!("{layer_prefix}.norm2"));
        push_quantized_linear(
            runtime,
            &mut head_weights,
            &format!("{layer_prefix}.self_attn.in_proj"),
            [3 * HIDDEN_SIZE, HIDDEN_SIZE],
        );
        push_quantized_linear(
            runtime,
            &mut head_weights,
            &format!("{layer_prefix}.self_attn.out_proj"),
            [HIDDEN_SIZE, HIDDEN_SIZE],
        );
        push_quantized_linear(
            runtime,
            &mut head_weights,
            &format!("{layer_prefix}.linear1"),
            [4 * HIDDEN_SIZE, HIDDEN_SIZE],
        );
        push_quantized_linear(
            runtime,
            &mut head_weights,
            &format!("{layer_prefix}.linear2"),
            [HIDDEN_SIZE, 4 * HIDDEN_SIZE],
        );
    }
    push_norm_pair(runtime, &mut head_weights, "scorer.layers.0");
    head_weights.push_dense(
        runtime,
        "scorer.layers.1.weight",
        &[HIDDEN_SIZE, HIDDEN_SIZE],
        0.0,
    );
    head_weights.push_dense(runtime, "scorer.layers.1.bias", &[HIDDEN_SIZE], 0.0);
    head_weights.push_dense(runtime, "scorer.layers.3.weight", &[1, HIDDEN_SIZE], 0.0);
    head_weights.push_dense(runtime, "scorer.layers.3.bias", &[1], 0.0);
    head_weights.named_tensors
}

/// One row of a head pass: how many real tokens it has, its question type and its markers.
struct TestRow {
    real_token_count: usize,
    question_type_index: u32,
    option_marker_positions: Vec<usize>,
}

/// Runs the head over made-up hidden states for `test_rows`, padded to the longest.
///
/// A row's hidden states are the same values whether it is run alone or in a
/// batch: they depend only on the token position and channel.
fn head_logits(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    configuration: &LayaConfiguration,
    test_rows: &[TestRow],
) -> Result<Vec<Vec<f32>>, ModernBertTrunkError> {
    let mut token_sequences = Vec::with_capacity(test_rows.len());
    for test_row in test_rows {
        token_sequences.push(vec![7_u32; test_row.real_token_count]);
    }
    let mut borrowed_sequences = Vec::with_capacity(test_rows.len());
    for token_sequence in &token_sequences {
        borrowed_sequences.push(token_sequence.as_slice());
    }
    let token_batch = PaddedTokenBatch::from_sequences(&borrowed_sequences, PAD_TOKEN_ID)
        .expect("the test rows form a batch");
    let padded_length = token_batch.padded_length() as i32;
    // Every row reads the same per-position values, tiled across the batch.
    let one_row_states = shakespeare_values(runtime, &[1, padded_length, HIDDEN_SIZE], 4_000, 0.0);
    let mut tiled_rows = Vec::with_capacity(test_rows.len());
    for _test_row in test_rows {
        tiled_rows.push(&one_row_states);
    }
    let hidden_states = runtime
        .concatenate_axis(&tiled_rows, 0)
        .expect("the hidden states should tile");
    let mut question_type_indices = Vec::with_capacity(test_rows.len());
    let mut option_marker_positions = Vec::with_capacity(test_rows.len());
    for test_row in test_rows {
        question_type_indices.push(test_row.question_type_index);
        option_marker_positions.push(test_row.option_marker_positions.as_slice());
    }
    let head = LayaDecisionHead {
        runtime,
        tensors,
        configuration,
    };
    head.option_logits(
        &hidden_states,
        &HeadPassRows {
            token_batch: &token_batch,
            question_type_indices: &question_type_indices,
            option_marker_positions: &option_marker_positions,
        },
    )
}

fn short_row() -> TestRow {
    TestRow {
        real_token_count: 5,
        question_type_index: 0,
        option_marker_positions: vec![1, 3],
    }
}

fn long_row() -> TestRow {
    TestRow {
        real_token_count: 20,
        question_type_index: 1,
        option_marker_positions: vec![2, 5, 9, 12, 15],
    }
}

#[test]
#[serial]
fn should_score_a_padded_row_as_it_scores_that_row_alone() {
    let runtime = initialize_test_runtime();
    let configuration = synthetic_configuration(2);
    let tensors = load_synthetic_weights(&runtime, &synthetic_head_tensors(&runtime, 2));
    let batched_logits = head_logits(
        &runtime,
        &tensors,
        &configuration,
        &[short_row(), long_row()],
    )
    .expect("the head should score the batch");
    let short_alone = head_logits(&runtime, &tensors, &configuration, &[short_row()])
        .expect("the head should score the short row");
    let long_alone = head_logits(&runtime, &tensors, &configuration, &[long_row()])
        .expect("the head should score the long row");
    assert!(largest_difference(&batched_logits[0], &short_alone[0]) <= LOGIT_TOLERANCE);
    assert!(largest_difference(&batched_logits[1], &long_alone[0]) <= LOGIT_TOLERANCE);
}

#[test]
#[serial]
fn should_return_each_row_exactly_its_own_option_count() {
    let runtime = initialize_test_runtime();
    let configuration = synthetic_configuration(2);
    let tensors = load_synthetic_weights(&runtime, &synthetic_head_tensors(&runtime, 2));
    let single_option_row = TestRow {
        real_token_count: 5,
        question_type_index: 0,
        option_marker_positions: vec![2],
    };
    let batched_logits = head_logits(
        &runtime,
        &tensors,
        &configuration,
        &[short_row(), long_row(), single_option_row],
    )
    .expect("the head should score the batch");
    assert_eq!(batched_logits[0].len(), 2);
    assert_eq!(batched_logits[1].len(), 5);
    assert_eq!(batched_logits[2].len(), 1);
    for row_logits in &batched_logits {
        for option_logit in row_logits {
            assert!(option_logit.is_finite());
        }
    }
}

#[test]
#[serial]
fn should_score_the_same_row_differently_for_a_different_question_type() {
    let runtime = initialize_test_runtime();
    let configuration = synthetic_configuration(2);
    let tensors = load_synthetic_weights(&runtime, &synthetic_head_tensors(&runtime, 2));
    let mut other_type_row = short_row();
    other_type_row.question_type_index = 2;
    let choice_logits = head_logits(&runtime, &tensors, &configuration, &[short_row()])
        .expect("the head should score a choice row");
    let noul_logits = head_logits(&runtime, &tensors, &configuration, &[other_type_row])
        .expect("the head should score a noul row");
    assert!(largest_difference(&choice_logits[0], &noul_logits[0]) > 0.0);
}

#[test]
#[serial]
fn should_score_markers_with_no_head_layers() {
    let runtime = initialize_test_runtime();
    let configuration = synthetic_configuration(0);
    let tensors = load_synthetic_weights(&runtime, &synthetic_head_tensors(&runtime, 0));
    let headless_logits = head_logits(&runtime, &tensors, &configuration, &[short_row()])
        .expect("a headless checkpoint should still score");
    assert_eq!(headless_logits[0].len(), 2);
}

#[test]
#[serial]
fn should_name_the_head_tensor_an_artifact_is_missing() {
    let runtime = initialize_test_runtime();
    let configuration = synthetic_configuration(2);
    let mut named_tensors = synthetic_head_tensors(&runtime, 2);
    named_tensors.retain(|(tensor_name, _)| tensor_name != "scorer.layers.3.bias");
    let tensors = load_synthetic_weights(&runtime, &named_tensors);
    let Err(ModernBertTrunkError::UnavailableTensor { tensor_name, .. }) =
        head_logits(&runtime, &tensors, &configuration, &[short_row()])
    else {
        panic!("a missing head tensor must be reported by name");
    };
    assert_eq!(tensor_name, "scorer.layers.3.bias");
}
