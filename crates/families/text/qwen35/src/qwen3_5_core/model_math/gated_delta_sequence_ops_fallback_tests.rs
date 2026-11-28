//! The one-token operations route of the gated-delta recurrence over a batch
//! of rows. Serving reads one row at a time, but training scores several rows
//! in one batch, and where the worker demoted the fused kernel (MLX's CUDA
//! backend runs no Metal kernel) training's forward is this route: every row
//! must get what it gets alone.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::MlxRuntime;
use serial_test::file_serial;

use super::gated_delta_inputs::GatedDeltaInputs;
use super::gated_delta_sequence::qwen3_5_gated_delta_sequence_ops_fallback;
use super::gated_delta_test_inputs::{RecurrenceInputs, gated_delta_test_runtime as test_runtime};

const KEY_HEAD_COUNT: i32 = 2;
const VALUE_HEAD_COUNT: i32 = 4;
const KEY_HEAD_DIMENSION: i32 = 128;
const VALUE_HEAD_DIMENSION: i32 = 32;
// One seed per input. MLX's random draws depend only on the key, so inputs
// of the same shape drawn from one key come out equal: queries would equal
// keys and decays would equal update rates, and a route that swapped either
// pair would still match its row run alone.
const QUERY_SEED: u64 = 2699;
const KEY_SEED: u64 = 2700;
const VALUE_SEED: u64 = 2701;
const DECAY_SEED: u64 = 2702;
const UPDATE_RATE_SEED: u64 = 2703;
const STATE_SEED: u64 = 2704;

impl RecurrenceInputs {
    /// Inputs for `row_count` rows of `token_count` tokens.
    fn batch(runtime: &MlxRuntime, row_count: i32, token_count: i32) -> Self {
        let key_shape = [row_count, token_count, KEY_HEAD_COUNT, KEY_HEAD_DIMENSION];
        let value_shape = [
            row_count,
            token_count,
            VALUE_HEAD_COUNT,
            VALUE_HEAD_DIMENSION,
        ];
        let rate_shape = [row_count, token_count, VALUE_HEAD_COUNT];
        let state_shape = [
            row_count,
            VALUE_HEAD_COUNT,
            VALUE_HEAD_DIMENSION,
            KEY_HEAD_DIMENSION,
        ];
        Self {
            queries: normal_values(runtime, QUERY_SEED, &key_shape, 0.1),
            keys: normal_values(runtime, KEY_SEED, &key_shape, 0.1),
            values: normal_values(runtime, VALUE_SEED, &value_shape, 1.0),
            decays: unit_interval_values(runtime, DECAY_SEED, &rate_shape),
            update_rates: unit_interval_values(runtime, UPDATE_RATE_SEED, &rate_shape),
            recurrent_state: normal_values(runtime, STATE_SEED, &state_shape, 0.5),
        }
    }

    /// Row `row_index` alone, as a batch of one.
    fn batch_row_alone(&self, runtime: &MlxRuntime, row_index: i32) -> Self {
        Self {
            queries: batch_row(runtime, &self.queries, row_index),
            keys: batch_row(runtime, &self.keys, row_index),
            values: batch_row(runtime, &self.values, row_index),
            decays: batch_row(runtime, &self.decays, row_index),
            update_rates: batch_row(runtime, &self.update_rates, row_index),
            recurrent_state: batch_row(runtime, &self.recurrent_state, row_index),
        }
    }
}

fn normal_values(runtime: &MlxRuntime, seed: u64, shape: &[i32], scale: f32) -> MlxArray {
    let key = runtime.random_key(seed).expect("a key builds");
    runtime
        .random_normal(shape, MlxDtype::Float32, 0.0, scale, &key)
        .expect("random inputs build")
}

/// Rates between 0 and 1, as the layer's sigmoid gives them.
fn unit_interval_values(runtime: &MlxRuntime, seed: u64, shape: &[i32]) -> MlxArray {
    let rate_logits = normal_values(runtime, seed, shape, 1.0);
    runtime.sigmoid(&rate_logits).expect("rates build")
}

/// The leading-axis row `row_index` of `batch_array`, kept as a batch of one.
fn batch_row(runtime: &MlxRuntime, batch_array: &MlxArray, row_index: i32) -> MlxArray {
    let shape = batch_array.shape();
    let mut row_start = vec![0; shape.len()];
    row_start[0] = row_index;
    let mut row_end = shape.clone();
    row_end[0] = row_index + 1;
    runtime
        .slice(batch_array, &row_start, &row_end, &vec![1; shape.len()])
        .expect("a row slices")
}

fn read_values(array: &MlxArray) -> Vec<f32> {
    array.to_vec_f32().expect("the array reads back")
}

#[test]
#[file_serial]
fn should_give_every_row_of_a_batch_what_it_gets_alone() {
    const ROW_COUNT: i32 = 3;
    let runtime = test_runtime();
    let batch_inputs = RecurrenceInputs::batch(&runtime, ROW_COUNT, 5);
    let (batch_outputs, batch_state) =
        qwen3_5_gated_delta_sequence_ops_fallback(&runtime, &batch_inputs.as_gated_delta_inputs())
            .expect("a batch of rows runs");
    assert_eq!(batch_outputs.shape()[0], ROW_COUNT);
    for row_index in 0..ROW_COUNT {
        let row_inputs = batch_inputs.batch_row_alone(&runtime, row_index);
        let (row_outputs, row_state) = qwen3_5_gated_delta_sequence_ops_fallback(
            &runtime,
            &row_inputs.as_gated_delta_inputs(),
        )
        .expect("one row runs");
        assert_eq!(
            read_values(&batch_row(&runtime, &batch_outputs, row_index)),
            read_values(&row_outputs),
            "row {row_index}'s outputs differ from its run alone"
        );
        assert_eq!(
            read_values(&batch_row(&runtime, &batch_state, row_index)),
            read_values(&row_state),
            "row {row_index}'s final state differs from its run alone"
        );
    }
}

#[test]
#[file_serial]
fn should_run_a_single_row_of_a_single_token() {
    let runtime = test_runtime();
    let single_inputs = RecurrenceInputs::batch(&runtime, 1, 1);
    let (outputs, next_state) =
        qwen3_5_gated_delta_sequence_ops_fallback(&runtime, &single_inputs.as_gated_delta_inputs())
            .expect("one token runs");
    assert_eq!(
        outputs.shape(),
        vec![1, 1, VALUE_HEAD_COUNT, VALUE_HEAD_DIMENSION]
    );
    assert_eq!(
        next_state.shape(),
        vec![
            1,
            VALUE_HEAD_COUNT,
            VALUE_HEAD_DIMENSION,
            KEY_HEAD_DIMENSION
        ]
    );
}

#[test]
#[file_serial]
fn should_refuse_a_state_for_fewer_rows_than_the_batch() {
    let runtime = test_runtime();
    let batch_inputs = RecurrenceInputs::batch(&runtime, 2, 3);
    let one_row_state = batch_row(&runtime, &batch_inputs.recurrent_state, 0);
    let mismatched_inputs = GatedDeltaInputs {
        recurrent_state: &one_row_state,
        ..batch_inputs.as_gated_delta_inputs()
    };
    assert!(
        qwen3_5_gated_delta_sequence_ops_fallback(&runtime, &mismatched_inputs).is_err(),
        "a state for one row must be refused for a batch of two"
    );
}
