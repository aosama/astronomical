use model_primitives::synthetic_weights_test_support::initialize_test_runtime as test_runtime;
use runtime_integration::MlxRuntime;
use serial_test::file_serial;

use super::KeptDraftedRows;
use crate::qwen4_exp::growing_sequence::GrowingSequence;
use crate::qwen4_exp::request_state::{IndexerLayerState, Qwen4ExpRequestState};

const WINDOW_LENGTH: i32 = 9;

fn window_of(runtime: &MlxRuntime, first_value: f32) -> mlx_bridge::MlxArray {
    let values: Vec<f32> = (0..WINDOW_LENGTH)
        .map(|row| first_value + row as f32)
        .collect();
    runtime
        .array_from_f32(&values, &[1, WINDOW_LENGTH, 1])
        .expect("a window")
}

fn keys_of(runtime: &MlxRuntime, row_count: i32) -> GrowingSequence {
    let values: Vec<f32> = (0..row_count).map(|row| row as f32).collect();
    let rows = runtime
        .array_from_f32(&values, &[1, row_count, 1])
        .expect("keys");
    GrowingSequence::starting_with(runtime, &rows, 1).expect("a key sequence")
}

/// A state after a four-row forward: the window slid by four rows and the
/// one indexer layer holds `key_count` keys and their complete blocks.
fn state_after_four_rows(runtime: &MlxRuntime, key_count: i32) -> Qwen4ExpRequestState {
    let mut state = Qwen4ExpRequestState::new(2, vec![7, 8]);
    // Rows 4..=12: the five surviving old rows, then the four forwarded ones.
    state.ngram_convolution_window = Some(window_of(runtime, 4.0));
    let block_values: Vec<f32> = (0..key_count / 4).map(|block| block as f32).collect();
    let block_rows = runtime
        .array_from_f32(&block_values, &[1, 1, key_count / 4, 1])
        .expect("block keys");
    state.indexer_by_layer[1] = Some(IndexerLayerState {
        keys: keys_of(runtime, key_count),
        block_keys: Some(GrowingSequence::starting_with(runtime, &block_rows, 2).expect("blocks")),
    });
    state
}

fn state_before(runtime: &MlxRuntime) -> Qwen4ExpRequestState {
    let mut state = Qwen4ExpRequestState::new(2, vec![5, 6]);
    // Rows 0..=8.
    state.ngram_convolution_window = Some(window_of(runtime, 0.0));
    state
}

fn four_rows_keeping(kept_row_count: i32) -> KeptDraftedRows {
    KeptDraftedRows {
        forwarded_row_count: 4,
        kept_row_count,
        tokens_per_indexer_block: 4,
    }
}

#[test]
#[file_serial]
fn should_rebuild_the_window_from_the_rows_before_and_the_kept_rows() {
    let runtime = test_runtime();
    let mut state = state_after_four_rows(&runtime, 20);
    state
        .rewind_to_kept_drafted_rows(
            &runtime,
            &state_before(&runtime),
            vec![9, 9],
            four_rows_keeping(2),
        )
        .expect("the rewind");
    let window = state.ngram_convolution_window.expect("a window");
    // Old rows 2..=8, then forwarded rows 9 and 10.
    assert_eq!(
        window.to_vec_f32().expect("window values"),
        vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    );
    assert_eq!(state.ngram_context, vec![9, 9]);
}

#[test]
#[file_serial]
fn should_drop_refused_keys_and_the_block_they_completed() {
    let runtime = test_runtime();
    // Twenty keys make five blocks; dropping two leaves eighteen keys, four blocks.
    let mut state = state_after_four_rows(&runtime, 20);
    state
        .rewind_to_kept_drafted_rows(
            &runtime,
            &state_before(&runtime),
            vec![1, 2],
            four_rows_keeping(2),
        )
        .expect("the rewind");
    let indexer = state.indexer_by_layer[1]
        .as_ref()
        .expect("the indexer layer");
    assert_eq!(indexer.keys.used_rows(), 18);
    assert_eq!(indexer.block_keys.as_ref().expect("blocks").used_rows(), 4);
}

#[test]
#[file_serial]
fn should_change_nothing_when_every_row_is_kept() {
    let runtime = test_runtime();
    let mut state = state_after_four_rows(&runtime, 20);
    state
        .rewind_to_kept_drafted_rows(
            &runtime,
            &state_before(&runtime),
            vec![1, 2],
            four_rows_keeping(4),
        )
        .expect("keeping every row");
    assert_eq!(
        state.ngram_context,
        vec![7, 8],
        "the history after the forward stays"
    );
    assert_eq!(
        state.indexer_by_layer[1]
            .as_ref()
            .expect("indexer")
            .keys
            .used_rows(),
        20
    );
}

#[test]
#[file_serial]
fn should_start_from_zeros_when_the_forward_was_the_first() {
    let runtime = test_runtime();
    let mut state = state_after_four_rows(&runtime, 20);
    let mut fresh_before = Qwen4ExpRequestState::new(2, vec![0, 0]);
    fresh_before.ngram_convolution_window = None;
    state
        .rewind_to_kept_drafted_rows(&runtime, &fresh_before, vec![1, 2], four_rows_keeping(1))
        .expect("the rewind");
    let window = state.ngram_convolution_window.expect("a window");
    assert_eq!(
        window.to_vec_f32().expect("window values"),
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]
    );
}

#[test]
#[file_serial]
fn should_refuse_keeping_no_row_more_rows_than_forwarded_or_a_block_past_the_window() {
    let runtime = test_runtime();
    let before = state_before(&runtime);
    for refused_rows in [
        four_rows_keeping(0),
        four_rows_keeping(5),
        KeptDraftedRows {
            forwarded_row_count: WINDOW_LENGTH + 1,
            kept_row_count: 1,
            tokens_per_indexer_block: 4,
        },
    ] {
        let mut state = state_after_four_rows(&runtime, 20);
        assert!(
            state
                .rewind_to_kept_drafted_rows(&runtime, &before, vec![1, 2], refused_rows)
                .is_err(),
            "{refused_rows:?}"
        );
    }
}
