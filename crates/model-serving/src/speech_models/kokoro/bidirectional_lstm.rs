//! PyTorch's one-layer bidirectional LSTM, run step by step on MLX.
//!
//! MLX has no recurrent primitive, so each time step is a few small graph
//! operations. Two choices keep that cheap: the input projection for every
//! step is one matrix product up front, and the two directions run as a
//! batch of two (the backward direction reads the sequence reversed), which
//! halves the number of steps the graph holds. Gate order is PyTorch's:
//! input, forget, cell, output.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_model_error::SpeechModelError;

const FORWARD_SUFFIX: &str = "";
const BACKWARD_SUFFIX: &str = "_reverse";
const DIRECTION_COUNT: i32 = 2;

pub(crate) struct BidirectionalLstm {
    /// `[2, in, 4 * hidden]`: forward then backward input weights, transposed.
    input_weights: MlxArray,
    /// `[2, hidden, 4 * hidden]`: forward then backward recurrent weights, transposed.
    recurrent_weights: MlxArray,
    /// `[2, 1, 4 * hidden]`: both biases of each direction, summed.
    gate_biases: MlxArray,
    hidden_width: i32,
}

/// The two directions' hidden and cell states, `[2, 1, hidden]` each.
struct RecurrentState {
    hidden: MlxArray,
    cell: MlxArray,
}

impl BidirectionalLstm {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
    ) -> Result<Self, SpeechModelError> {
        let runtime = checkpoint.runtime;
        let recurrent_weights = stacked_transposed_weights(checkpoint, prefix, "weight_hh")?;
        let forward_bias = summed_bias(checkpoint, prefix, FORWARD_SUFFIX)?;
        let backward_bias = summed_bias(checkpoint, prefix, BACKWARD_SUFFIX)?;
        let stacked_biases = runtime.stack_axis(&[&forward_bias, &backward_bias], 0)?;
        Ok(Self {
            input_weights: stacked_transposed_weights(checkpoint, prefix, "weight_ih")?,
            hidden_width: recurrent_weights.shape()[1],
            recurrent_weights,
            gate_biases: runtime.expand_dims(&stacked_biases, 1)?,
        })
    }

    /// `[length, in]` to `[length, 2 * hidden]`: each step's forward state
    /// followed by its backward state.
    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        sequence: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let sequence_length = sequence.shape()[0];
        let reversed_order: Vec<i32> = (0..sequence_length).rev().collect();
        let reversed_indices = runtime.array_from_i32(&reversed_order, &[sequence_length])?;
        let reversed_sequence = runtime.take_axis(sequence, &reversed_indices, 0)?;
        let both_directions = runtime.stack_axis(&[sequence, &reversed_sequence], 0)?;
        let projected_inputs = runtime.add(
            &runtime.matmul(&both_directions, &self.input_weights)?,
            &self.gate_biases,
        )?;
        let step_states = self.run_steps(runtime, &projected_inputs, sequence_length)?;
        let state_refs: Vec<&MlxArray> = step_states.iter().collect();
        // `[2, length, hidden]`, the backward half still in reversed order.
        let stacked_states = runtime.concatenate_axis(&state_refs, 1)?;
        let forward_states = self.direction(runtime, &stacked_states, 0, sequence_length)?;
        let backward_reversed = self.direction(runtime, &stacked_states, 1, sequence_length)?;
        let backward_states = runtime.take_axis(&backward_reversed, &reversed_indices, 1)?;
        let joined = runtime.concatenate_axis(&[&forward_states, &backward_states], 2)?;
        Ok(runtime.squeeze_axis(&joined, 0)?)
    }

    fn direction(
        &self,
        runtime: &MlxRuntime,
        stacked_states: &MlxArray,
        direction_index: i32,
        sequence_length: i32,
    ) -> Result<MlxArray, SpeechModelError> {
        Ok(runtime.slice(
            stacked_states,
            &[direction_index, 0, 0],
            &[direction_index + 1, sequence_length, self.hidden_width],
            &[1, 1, 1],
        )?)
    }

    /// The hidden state after each step, `[2, 1, hidden]` each.
    fn run_steps(
        &self,
        runtime: &MlxRuntime,
        projected_inputs: &MlxArray,
        sequence_length: i32,
    ) -> Result<Vec<MlxArray>, SpeechModelError> {
        let state_shape = [DIRECTION_COUNT, 1, self.hidden_width];
        let mut state = RecurrentState {
            hidden: runtime.zeros(&state_shape, MlxDtype::Float32)?,
            cell: runtime.zeros(&state_shape, MlxDtype::Float32)?,
        };
        let mut step_states = Vec::with_capacity(sequence_length as usize);
        for step_index in 0..sequence_length {
            let gate_width = 4 * self.hidden_width;
            let step_input = runtime.slice(
                projected_inputs,
                &[0, step_index, 0],
                &[DIRECTION_COUNT, step_index + 1, gate_width],
                &[1, 1, 1],
            )?;
            state = self.lstm_step(runtime, &step_input, &state)?;
            step_states.push(state.hidden.retain()?);
        }
        Ok(step_states)
    }

    fn lstm_step(
        &self,
        runtime: &MlxRuntime,
        step_input: &MlxArray,
        state: &RecurrentState,
    ) -> Result<RecurrentState, SpeechModelError> {
        let gates = runtime.add(
            step_input,
            &runtime.matmul(&state.hidden, &self.recurrent_weights)?,
        )?;
        let squashed_gates = runtime.sigmoid(&gates)?;
        let input_gate = self.gate(runtime, &squashed_gates, 0)?;
        let forget_gate = self.gate(runtime, &squashed_gates, 1)?;
        let candidate_cell = runtime.tanh(&self.gate(runtime, &gates, 2)?)?;
        let output_gate = self.gate(runtime, &squashed_gates, 3)?;
        let cell = runtime.add(
            &runtime.multiply(&forget_gate, &state.cell)?,
            &runtime.multiply(&input_gate, &candidate_cell)?,
        )?;
        let hidden = runtime.multiply(&output_gate, &runtime.tanh(&cell)?)?;
        Ok(RecurrentState { hidden, cell })
    }

    /// The `gate_index`-th hidden-wide slice of `[2, 1, 4 * hidden]` gates.
    fn gate(
        &self,
        runtime: &MlxRuntime,
        gates: &MlxArray,
        gate_index: i32,
    ) -> Result<MlxArray, SpeechModelError> {
        let gate_start = gate_index * self.hidden_width;
        Ok(runtime.slice(
            gates,
            &[0, 0, gate_start],
            &[DIRECTION_COUNT, 1, gate_start + self.hidden_width],
            &[1, 1, 1],
        )?)
    }
}

/// `[2, in, 4 * hidden]` from the forward and backward `[4 * hidden, in]` weights.
fn stacked_transposed_weights(
    checkpoint: &CheckpointTensors<'_>,
    prefix: &str,
    weight_kind: &str,
) -> Result<MlxArray, SpeechModelError> {
    let forward_weight =
        checkpoint.tensor(&format!("{prefix}.{weight_kind}_l0{FORWARD_SUFFIX}"))?;
    let backward_weight =
        checkpoint.tensor(&format!("{prefix}.{weight_kind}_l0{BACKWARD_SUFFIX}"))?;
    let stacked = checkpoint
        .runtime
        .stack_axis(&[&forward_weight, &backward_weight], 0)?;
    Ok(checkpoint.runtime.transpose_axes(&stacked, &[0, 2, 1])?)
}

/// PyTorch keeps an input bias and a recurrent bias; they are always added
/// together, so they are summed once here.
fn summed_bias(
    checkpoint: &CheckpointTensors<'_>,
    prefix: &str,
    direction_suffix: &str,
) -> Result<MlxArray, SpeechModelError> {
    let input_bias = checkpoint.tensor(&format!("{prefix}.bias_ih_l0{direction_suffix}"))?;
    let recurrent_bias = checkpoint.tensor(&format!("{prefix}.bias_hh_l0{direction_suffix}"))?;
    Ok(checkpoint.runtime.add(&input_bias, &recurrent_bias)?)
}

#[cfg(test)]
#[path = "bidirectional_lstm_tests.rs"]
mod tests;
