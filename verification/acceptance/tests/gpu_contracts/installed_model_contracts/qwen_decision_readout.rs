//! The reads a Qwen3.5-based decision model makes from the backbone, checked
//! on real weights.
//!
//! Kev and Clef do not sample from Qwen3.5. They read its final hidden
//! states, Clef also reads output-projection rows, and Kev answers each
//! question as a row continuing one shared state. This journey checks each
//! read against something the engine already trusts:
//!
//! * the last final hidden state dotted with the output rows reproduces the
//!   logits the vocabulary head computes, which proves both the
//!   normalization and the row read;
//! * a question row continued from a prefilled state reads as the same
//!   tokens do in one uninterrupted pass, up to the rounding any other split
//!   of that pass also introduces (a split changes how the gated-delta layers
//!   chunk their scan, so it can never be bit-identical);
//! * after a restore, a second question reads exactly as if the first had
//!   never run.
//!
//! Run it with `cargo test-qwen-decision-readout --configuration PATH`,
//! where the configuration's `model_directory` is an MLX Qwen3.5 model. Run
//! it once with a quantized build and once with a bfloat16 one: their
//! output projections are read along different paths.

use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5ModelChunkingConfiguration, Qwen3_5ResidentModel,
    Qwen3_5Tokenizer, RequestDecoderStateStack,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use astronomical_test_harness::test_run::TestRunContext;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

/// A 0.8B model needs about 2 GB in bfloat16; this leaves ample workspace.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 8 * 1024 * 1024 * 1024;
const MAXIMUM_OUTPUT_TOKENS: u32 = 1_024;
const FULL_ATTENTION_KV_GROWTH_TOKENS: u32 = 256;
/// The largest difference accepted between a logit the vocabulary head
/// computes and the same logit rebuilt from the readout.
///
/// The head multiplies in the model's own precision, while the rebuilt logit
/// is a float32 dot product of the same normalized state and row, so the two
/// differ by the head's rounding. Measured when this journey was added, on
/// Qwen3.5-0.8B: 0.031 with the 4-bit build and 0.029 with the bfloat16
/// one; the bound leaves room for another machine's rounding.
const MAXIMUM_LOGIT_DIFFERENCE: f32 = 0.25;

struct LoadedBackbone {
    model: Qwen3_5ResidentModel,
    tokenizer: Qwen3_5Tokenizer,
}

fn load_backbone() -> LoadedBackbone {
    let model_directory = TestRunContext::current().required_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, MAXIMUM_OUTPUT_TOKENS)
        .expect("the Qwen3.5 model should pass production validation");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the tokenizer should load from the validated artifact");
    let model_chunking =
        Qwen3_5ModelChunkingConfiguration::new(FULL_ATTENTION_KV_GROWTH_TOKENS, 0, 1, 3)
            .expect("the standard chunking tuple should be valid");
    let memory_limits =
        MlxMemoryLimits::new(TEST_MLX_MEMORY_CEILING_BYTES, TEST_MLX_MEMORY_CEILING_BYTES)
            .expect("the test memory limits should be valid");
    let runtime = MlxRuntime::initialize(memory_limits).expect("the MLX runtime should start");
    let mut model = Qwen3_5ResidentModel::load(
        runtime,
        validated_artifact,
        &model_directory,
        model_chunking,
    )
    .expect("the resident model should load");
    model
        .materialize_complete_weights_matching_production()
        .expect("the resident model should materialize its weights");
    LoadedBackbone { model, tokenizer }
}

fn fresh_state(model: &Qwen3_5ResidentModel) -> RequestDecoderStateStack {
    model
        .empty_request_decoder_state_matching_loaded_weights(FULL_ATTENTION_KV_GROWTH_TOKENS as i32)
        .expect("a decoder state should allocate")
}

fn shakespeare_tokens(tokenizer: &Qwen3_5Tokenizer, word_count: usize) -> Vec<u32> {
    tokenizer
        .encode_prompt(&romeo_and_juliet::excerpt(word_count))
        .expect("the Shakespeare excerpt should encode")
}

fn question_tokens(tokenizer: &Qwen3_5Tokenizer, question_text: &str) -> Vec<u32> {
    tokenizer
        .encode_prompt(question_text)
        .expect("the question should encode")
}

/// The readout of `token_ids` continuing `state` from `starting_position`, on the host.
fn hidden_rows(
    model: &Qwen3_5ResidentModel,
    token_ids: &[u32],
    starting_position: u32,
    state: &mut RequestDecoderStateStack,
) -> Vec<f32> {
    model
        .final_normalized_hidden_states(token_ids, starting_position, state)
        .expect("the readout forward should run")
        .to_vec_f32()
        .expect("the readout should read back")
}

fn largest_difference(left_values: &[f32], right_values: &[f32]) -> f32 {
    assert_eq!(left_values.len(), right_values.len());
    left_values
        .iter()
        .zip(right_values)
        .map(|(left_value, right_value)| (left_value - right_value).abs())
        .fold(0.0, f32::max)
}

#[ignore = "requires an MLX Qwen3.5 model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_rebuild_the_vocabulary_head_logits_from_the_readout() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let backbone = load_backbone();
    let token_ids = shakespeare_tokens(&backbone.tokenizer, 60);
    let hidden_size = backbone.model.hidden_size() as usize;

    let readout = hidden_rows(
        &backbone.model,
        &token_ids,
        0,
        &mut fresh_state(&backbone.model),
    );
    let last_hidden_state = &readout[readout.len() - hidden_size..];
    let head_output = backbone
        .model
        .forward_chunk_with_pre_final_normalization_hidden_states(
            &token_ids,
            0,
            &mut fresh_state(&backbone.model),
        )
        .expect("the vocabulary-head forward should run");
    let head_logits = head_output
        .final_logits()
        .to_vec_f32()
        .expect("the head logits should read back");
    // The ten most likely next tokens, and two that are not.
    let mut ranked_token_ids: Vec<u32> = (0..head_logits.len() as u32).collect();
    ranked_token_ids.sort_by(|left, right| {
        head_logits[*right as usize].total_cmp(&head_logits[*left as usize])
    });
    let mut compared_token_ids = ranked_token_ids[..10].to_vec();
    compared_token_ids.extend([0, ranked_token_ids[ranked_token_ids.len() / 2]]);
    let output_rows = backbone
        .model
        .output_embedding_rows(&compared_token_ids)
        .expect("the output rows should read")
        .to_vec_f32()
        .expect("the output rows should read back");

    let mut largest_logit_difference = 0.0_f32;
    for (row_index, token_id) in compared_token_ids.iter().enumerate() {
        let output_row = &output_rows[row_index * hidden_size..(row_index + 1) * hidden_size];
        let rebuilt_logit: f32 = output_row
            .iter()
            .zip(last_hidden_state)
            .map(|(row_value, hidden_value)| row_value * hidden_value)
            .sum();
        largest_logit_difference =
            largest_logit_difference.max((rebuilt_logit - head_logits[*token_id as usize]).abs());
    }
    eprintln!(
        "[qwen-decision-readout] tokens={} largest_logit_difference={largest_logit_difference:.5}",
        token_ids.len()
    );
    assert!(largest_logit_difference <= MAXIMUM_LOGIT_DIFFERENCE);
}

/// Splitting one uninterrupted prefill at these distances before the state's
/// end measures how far an ordinary split moves the last rows' rounding.
const CONTROL_SPLIT_DISTANCES_TOKENS: [usize; 3] = [37, 61, 113];

#[ignore = "requires an MLX Qwen3.5 model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_read_a_question_row_from_a_shared_state_as_if_it_followed_the_state() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let backbone = load_backbone();
    let state_tokens = shakespeare_tokens(&backbone.tokenizer, 120);
    let row_tokens = question_tokens(&backbone.tokenizer, "Is the speaker in love with Juliet?");
    let hidden_size = backbone.model.hidden_size() as usize;
    let mut straight_tokens = state_tokens.clone();
    straight_tokens.extend(&row_tokens);
    let straight_readout = hidden_rows(
        &backbone.model,
        &straight_tokens,
        0,
        &mut fresh_state(&backbone.model),
    );
    let straight_row_readout = &straight_readout[state_tokens.len() * hidden_size..];

    let branched_row_readout =
        row_readout_after_split(&backbone.model, &straight_tokens, state_tokens.len());
    let mut control_split_differences = Vec::new();
    for split_distance in CONTROL_SPLIT_DISTANCES_TOKENS {
        let split_point = state_tokens.len() - split_distance;
        let control_readout =
            row_readout_after_split(&backbone.model, &straight_tokens, split_point);
        control_split_differences.push(largest_difference(
            straight_row_readout,
            &control_readout[split_distance * hidden_size..],
        ));
    }

    let branch_difference = largest_difference(straight_row_readout, &branched_row_readout);
    let largest_control_difference = control_split_differences
        .iter()
        .copied()
        .fold(0.0, f32::max);
    let largest_magnitude = straight_row_readout
        .iter()
        .map(|hidden_value| hidden_value.abs())
        .fold(0.0, f32::max);
    eprintln!(
        "[qwen-decision-readout] state_tokens={} row_tokens={} branch_difference={branch_difference} control_split_differences={control_split_differences:?} largest_magnitude={largest_magnitude}",
        state_tokens.len(),
        row_tokens.len()
    );
    assert!(
        branch_difference <= largest_control_difference + bfloat16_step_at(largest_magnitude),
        "a branched row must be no further from the uninterrupted pass than an ordinary split is"
    );
}

/// The readout of `straight_tokens[split_point..]` after prefilling the
/// tokens before `split_point`: how a question row continues a shared state.
fn row_readout_after_split(
    model: &Qwen3_5ResidentModel,
    straight_tokens: &[u32],
    split_point: usize,
) -> Vec<f32> {
    let mut split_state = fresh_state(model);
    model
        .prefill_chunk(&straight_tokens[..split_point], 0, &mut split_state)
        .expect("the part before the split should prefill");
    hidden_rows(
        model,
        &straight_tokens[split_point..],
        split_point as u32,
        &mut split_state,
    )
}

/// The distance between adjacent bfloat16 values at `magnitude`: the
/// backbone computes in bfloat16 before the readout widens to float32, so
/// two runs that round once differently differ by about this much.
fn bfloat16_step_at(magnitude: f32) -> f32 {
    const BFLOAT16_FRACTION_BITS: i32 = 7;
    if magnitude <= 0.0 {
        return 0.0;
    }
    2.0_f32.powi(magnitude.log2().floor() as i32 - BFLOAT16_FRACTION_BITS)
}

#[ignore = "requires an MLX Qwen3.5 model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_read_each_question_after_a_restore_as_if_no_other_question_had_run() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let backbone = load_backbone();
    let state_tokens = shakespeare_tokens(&backbone.tokenizer, 120);
    let first_row = question_tokens(&backbone.tokenizer, "Is the speaker in love with Juliet?");
    let second_row = question_tokens(&backbone.tokenizer, "Does the passage take place at night?");
    let state_length = state_tokens.len() as u32;
    let mut shared_state = fresh_state(&backbone.model);
    backbone
        .model
        .prefill_chunk(&state_tokens, 0, &mut shared_state)
        .expect("the state should prefill");

    let state_checkpoint = shared_state
        .checkpoint()
        .expect("the state should checkpoint");
    let first_row_readout =
        hidden_rows(&backbone.model, &first_row, state_length, &mut shared_state);
    shared_state
        .restore_checkpoint(state_checkpoint)
        .expect("the state should restore");
    let state_checkpoint = shared_state
        .checkpoint()
        .expect("the state should checkpoint");
    let second_row_readout = hidden_rows(
        &backbone.model,
        &second_row,
        state_length,
        &mut shared_state,
    );
    shared_state
        .restore_checkpoint(state_checkpoint)
        .expect("the state should restore");
    let first_row_again = hidden_rows(&backbone.model, &first_row, state_length, &mut shared_state);

    let mut lone_state = fresh_state(&backbone.model);
    backbone
        .model
        .prefill_chunk(&state_tokens, 0, &mut lone_state)
        .expect("the state should prefill");
    let second_row_alone = hidden_rows(&backbone.model, &second_row, state_length, &mut lone_state);
    assert_eq!(
        second_row_readout, second_row_alone,
        "a question after a restore reads exactly as on a state no other question touched"
    );
    assert_eq!(
        first_row_readout, first_row_again,
        "the same question on the same restored state reads exactly the same"
    );
}

#[ignore = "requires an MLX Qwen3.5 model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_refuse_reads_of_no_tokens_or_rows_outside_the_vocabulary() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let backbone = load_backbone();
    let vocabulary_size = backbone.model.vocabulary_size();

    assert!(backbone.model.output_embedding_rows(&[]).is_err());
    assert!(
        backbone
            .model
            .final_normalized_hidden_states(&[], 0, &mut fresh_state(&backbone.model))
            .is_err(),
        "a readout of no tokens must be refused, not return an empty array"
    );
    assert!(
        backbone
            .model
            .output_embedding_rows(&[vocabulary_size])
            .is_err()
    );
    let last_row = backbone
        .model
        .output_embedding_rows(&[vocabulary_size - 1])
        .expect("the last row of the projection should read");
    assert_eq!(
        last_row.shape(),
        vec![1, backbone.model.hidden_size() as i32]
    );
}
