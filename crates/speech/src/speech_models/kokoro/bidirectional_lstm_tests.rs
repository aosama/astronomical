//! The LSTM against a hand-computed step, and its two directions against
//! each other.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;
use serial_test::serial;

use super::BidirectionalLstm;
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_test_support::{
    assert_close, checkpoint_from_arrays, initialize_test_runtime, read_back, shakespeare_values,
    upload,
};

const HIDDEN_WIDTH: i32 = 2;
const INPUT_WIDTH: i32 = 3;

/// Both directions get the same weights, drawn from the play, so the backward
/// half must equal a forward run over the reversed sequence.
fn mirrored_lstm(runtime: &MlxRuntime) -> BidirectionalLstm {
    let gate_rows = 4 * HIDDEN_WIDTH;
    let input_weights = shakespeare_values((gate_rows * INPUT_WIDTH) as usize, 0);
    let recurrent_weights = shakespeare_values((gate_rows * HIDDEN_WIDTH) as usize, 100);
    let biases = shakespeare_values(gate_rows as usize, 200);
    let mut named_arrays: Vec<(String, MlxArray)> = Vec::new();
    for suffix in ["", "_reverse"] {
        named_arrays.push((
            format!("lstm.weight_ih_l0{suffix}"),
            upload(runtime, &input_weights, &[gate_rows, INPUT_WIDTH]),
        ));
        named_arrays.push((
            format!("lstm.weight_hh_l0{suffix}"),
            upload(runtime, &recurrent_weights, &[gate_rows, HIDDEN_WIDTH]),
        ));
        named_arrays.push((
            format!("lstm.bias_ih_l0{suffix}"),
            upload(runtime, &biases, &[gate_rows]),
        ));
        named_arrays.push((
            format!("lstm.bias_hh_l0{suffix}"),
            upload(runtime, &vec![0.0; gate_rows as usize], &[gate_rows]),
        ));
    }
    let tensors = checkpoint_from_arrays(runtime, &named_arrays);
    let checkpoint = CheckpointTensors {
        runtime,
        tensors: &tensors,
    };
    BidirectionalLstm::load(&checkpoint, "lstm").expect("the LSTM loads")
}

/// One gate's pre-activation for the first step: its weight row times the
/// input, plus its bias (the recurrent part is zero with no history).
fn first_step_gate(gate_row: usize, input_values: &[f32]) -> f32 {
    let input_weights = shakespeare_values((4 * HIDDEN_WIDTH * INPUT_WIDTH) as usize, 0);
    let biases = shakespeare_values((4 * HIDDEN_WIDTH) as usize, 200);
    let row_weights =
        &input_weights[gate_row * INPUT_WIDTH as usize..(gate_row + 1) * INPUT_WIDTH as usize];
    row_weights
        .iter()
        .zip(input_values)
        .map(|(weight, input)| weight * input)
        .sum::<f32>()
        + biases[gate_row]
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[test]
#[serial]
fn should_match_a_hand_computed_first_step() {
    let runtime = initialize_test_runtime();
    let lstm = mirrored_lstm(&runtime);
    let input_values = shakespeare_values(INPUT_WIDTH as usize, 300);
    let output = read_back(
        &runtime,
        &lstm
            .forward(
                &runtime,
                &upload(&runtime, &input_values, &[1, INPUT_WIDTH]),
            )
            .expect("forward"),
    );
    let hidden_width = HIDDEN_WIDTH as usize;
    let expected_hidden: Vec<f32> = (0..hidden_width)
        .map(|unit| {
            let cell = sigmoid(first_step_gate(unit, &input_values))
                * first_step_gate(2 * hidden_width + unit, &input_values).tanh();
            sigmoid(first_step_gate(3 * hidden_width + unit, &input_values)) * cell.tanh()
        })
        .collect();
    // With one step both directions see the same single input.
    assert_close(&output[..hidden_width], &expected_hidden, 1e-5);
    assert_close(&output[hidden_width..], &expected_hidden, 1e-5);
}

#[test]
#[serial]
fn should_read_the_backward_direction_as_a_forward_run_over_the_reversed_sequence() {
    let runtime = initialize_test_runtime();
    let lstm = mirrored_lstm(&runtime);
    let sequence_values = shakespeare_values((4 * INPUT_WIDTH) as usize, 400);
    let reversed_values: Vec<f32> = sequence_values
        .chunks(INPUT_WIDTH as usize)
        .rev()
        .flatten()
        .copied()
        .collect();
    let output = read_back(
        &runtime,
        &lstm
            .forward(
                &runtime,
                &upload(&runtime, &sequence_values, &[4, INPUT_WIDTH]),
            )
            .expect("forward"),
    );
    let reversed_output = read_back(
        &runtime,
        &lstm
            .forward(
                &runtime,
                &upload(&runtime, &reversed_values, &[4, INPUT_WIDTH]),
            )
            .expect("forward"),
    );
    let row_width = (2 * HIDDEN_WIDTH) as usize;
    let hidden_width = HIDDEN_WIDTH as usize;
    for step_index in 0..4 {
        let backward_half =
            &output[step_index * row_width + hidden_width..(step_index + 1) * row_width];
        let mirrored_step = 3 - step_index;
        let forward_half_of_reversed =
            &reversed_output[mirrored_step * row_width..mirrored_step * row_width + hidden_width];
        assert_close(backward_half, forward_half_of_reversed, 1e-5);
    }
}
