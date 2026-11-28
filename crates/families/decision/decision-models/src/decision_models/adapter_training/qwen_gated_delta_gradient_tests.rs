//! The trainable Gated DeltaNet recurrence against serving's two routes (the
//! fused kernel, on Metal only, and the one-token operations, on every
//! backend), and its borrowed gradient against the loss's own change along
//! a direction. It lives with decision training, whose loss it checks the
//! gradient against, because the Qwen 3.5 crate sits below decision models
//! (issue #2340).

use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError};
use runtime_integration::MlxRuntime;
use serial_test::file_serial;

use super::loss_gradient::{TrainableLoss, loss_gradients};
use crate::qwen3_5_core::model_math::differentiable_gated_delta::DifferentiableGatedDelta;
use crate::{GatedDeltaInputs, qwen3_5_gated_delta_kernel, qwen3_5_gated_delta_sequence};
use model_primitives::synthetic_weights_test_support::{
    initialize_test_runtime, shakespeare_values,
};

/// Clef's linear-attention geometry: 16 key heads and 32 value heads of 128.
const KEY_HEAD_COUNT: i32 = 16;
const VALUE_HEAD_COUNT: i32 = 32;
const HEAD_DIMENSION: i32 = 128;
/// Long enough for the kernel's pipelined blocks and its tail block.
const COMPARED_TOKEN_COUNT: i32 = 160;

/// The recurrence's inputs for `token_count` tokens, decays near the
/// model's typical 0.9 and write strengths near 0.5.
struct RecurrenceInputs {
    queries: MlxArray,
    keys: MlxArray,
    values: MlxArray,
    decays: MlxArray,
    update_rates: MlxArray,
    recurrent_state: MlxArray,
}

impl RecurrenceInputs {
    fn new(runtime: &MlxRuntime, token_count: i32, byte_offset: usize, decay_center: f32) -> Self {
        let key_shape = [1, token_count, KEY_HEAD_COUNT, HEAD_DIMENSION];
        let value_shape = [1, token_count, VALUE_HEAD_COUNT, HEAD_DIMENSION];
        let rate_shape = [1, token_count, VALUE_HEAD_COUNT];
        let state_shape = [1, VALUE_HEAD_COUNT, HEAD_DIMENSION, HEAD_DIMENSION];
        let rate_center = decay_center * 5.0 / 9.0;
        Self {
            queries: shakespeare_values(runtime, &key_shape, byte_offset, 0.0),
            keys: shakespeare_values(runtime, &key_shape, byte_offset + 7_000, 0.0),
            values: shakespeare_values(runtime, &value_shape, byte_offset + 13_000, 0.0),
            decays: shakespeare_values(runtime, &rate_shape, byte_offset + 21_000, decay_center),
            update_rates: shakespeare_values(
                runtime,
                &rate_shape,
                byte_offset + 23_000,
                rate_center,
            ),
            recurrent_state: shakespeare_values(runtime, &state_shape, byte_offset + 29_000, 0.0),
        }
    }

    /// The five differentiated inputs, in the loss's order.
    fn trainable(&self) -> Vec<MlxArray> {
        let mut trainable_inputs = Vec::with_capacity(5);
        for input in [
            &self.queries,
            &self.keys,
            &self.values,
            &self.decays,
            &self.update_rates,
        ] {
            trainable_inputs.push(input.retain().unwrap());
        }
        trainable_inputs
    }
}

fn largest_gap(left: &MlxArray, right: &MlxArray) -> f32 {
    let left_values = left.to_vec_f32().unwrap();
    let right_values = right.to_vec_f32().unwrap();
    assert_eq!(left_values.len(), right_values.len());
    let mut largest = 0.0_f32;
    for (left_value, right_value) in left_values.iter().zip(&right_values) {
        largest = largest.max((left_value - right_value).abs());
    }
    largest
}

fn largest_magnitude(array: &MlxArray) -> f32 {
    let mut largest = 0.0_f32;
    for value in array.to_vec_f32().unwrap() {
        largest = largest.max(value.abs());
    }
    largest
}

/// The largest output and state gaps between the trainable recurrence and
/// serving's route over the same inputs, and the served outputs' largest
/// magnitude, which keeps the comparison from being vacuous.
fn served_and_trainable_gaps(serving_uses_fused_kernel: bool) -> (f32, f32, f32) {
    let runtime = initialize_test_runtime();
    let inputs = RecurrenceInputs::new(&runtime, COMPARED_TOKEN_COUNT, 0, 0.9);
    let served_kernel = if serving_uses_fused_kernel {
        Some(qwen3_5_gated_delta_kernel().unwrap())
    } else {
        None
    };
    let gated_delta_inputs = GatedDeltaInputs {
        queries: &inputs.queries,
        keys: &inputs.keys,
        values: &inputs.values,
        decays: &inputs.decays,
        update_rates: &inputs.update_rates,
        recurrent_state: &inputs.recurrent_state,
    };
    let (served_outputs, served_state) =
        qwen3_5_gated_delta_sequence(&runtime, served_kernel.as_ref(), &gated_delta_inputs)
            .unwrap();
    let (trainable_outputs, trainable_state) =
        DifferentiableGatedDelta::new(serving_uses_fused_kernel)
            .unwrap()
            .sequence(&gated_delta_inputs)
            .unwrap();
    let output_gap = largest_gap(&served_outputs, &trainable_outputs);
    let state_gap = largest_gap(&served_state, &trainable_state);
    eprintln!(
        "[differentiable-gated-delta] fused_kernel={serving_uses_fused_kernel} output_gap={output_gap:e} state_gap={state_gap:e}"
    );
    (output_gap, state_gap, largest_magnitude(&served_outputs))
}

#[test]
#[file_serial]
fn should_score_exactly_as_the_served_fused_kernel() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_score_exactly_as_the_served_fused_kernel",
    ) {
        return;
    }
    let (output_gap, state_gap, served_magnitude) = served_and_trainable_gaps(true);
    // Both run serving's kernel on the same inputs: exactly equal.
    assert_eq!(output_gap, 0.0);
    assert_eq!(state_gap, 0.0);
    assert!(served_magnitude > 0.0, "the comparison must not be vacuous");
}

/// Where the worker demoted the fused kernel (MLX's CUDA backend runs no
/// Metal kernel), serving falls back to one-token operations, and training
/// must score exactly as that route does. It runs on every backend.
#[test]
#[file_serial]
fn should_score_exactly_as_the_served_operations_when_serving_demoted_the_kernel() {
    let (output_gap, state_gap, served_magnitude) = served_and_trainable_gaps(false);
    assert_eq!(output_gap, 0.0);
    assert_eq!(state_gap, 0.0);
    assert!(served_magnitude > 0.0, "the comparison must not be vacuous");
}

#[test]
#[file_serial]
fn should_refuse_inputs_whose_heads_do_not_match_and_name_the_recurrence() {
    let runtime = initialize_test_runtime();
    let inputs = RecurrenceInputs::new(&runtime, 1, 0, 0.9);
    let mismatched_values = shakespeare_values(&runtime, &[1, 1, 3, HEAD_DIMENSION], 0, 0.0);

    let Err(refusal) = DifferentiableGatedDelta::new(false)
        .unwrap()
        .sequence(&GatedDeltaInputs {
            queries: &inputs.queries,
            keys: &inputs.keys,
            values: &mismatched_values,
            decays: &inputs.decays,
            update_rates: &inputs.update_rates,
            recurrent_state: &inputs.recurrent_state,
        })
    else {
        panic!("values with the wrong head count must be refused");
    };
    assert_eq!(refusal.operation, "apply the gated-delta recurrence");
}

/// Every output and the final state, weighted by fixed values and summed,
/// so each input's gradient is non-trivial.
struct RecurrenceLoss {
    recurrence: DifferentiableGatedDelta,
    recurrent_state: MlxArray,
    output_weights: MlxArray,
    state_weights: MlxArray,
}

impl RecurrenceLoss {
    fn new(
        runtime: &MlxRuntime,
        inputs: &RecurrenceInputs,
        token_count: i32,
        serving_uses_fused_kernel: bool,
    ) -> Self {
        Self {
            recurrence: DifferentiableGatedDelta::new(serving_uses_fused_kernel).unwrap(),
            recurrent_state: inputs.recurrent_state.retain().unwrap(),
            output_weights: shakespeare_values(
                runtime,
                &[1, token_count, VALUE_HEAD_COUNT, HEAD_DIMENSION],
                31_000,
                0.0,
            ),
            state_weights: shakespeare_values(
                runtime,
                &[1, VALUE_HEAD_COUNT, HEAD_DIMENSION, HEAD_DIMENSION],
                37_000,
                0.0,
            ),
        }
    }
}

impl TrainableLoss for RecurrenceLoss {
    fn build_loss(
        &self,
        context: &MlxBindingsContext,
        trainable_arrays: &[MlxArray],
    ) -> Result<MlxArray, MlxCError> {
        let (sequence_outputs, next_state) = self.recurrence.sequence(&GatedDeltaInputs {
            queries: &trainable_arrays[0],
            keys: &trainable_arrays[1],
            values: &trainable_arrays[2],
            decays: &trainable_arrays[3],
            update_rates: &trainable_arrays[4],
            recurrent_state: &self.recurrent_state,
        })?;
        let weighted_outputs = context.multiply(&sequence_outputs, &self.output_weights)?;
        let weighted_state = context.multiply(&next_state, &self.state_weights)?;
        context.add(
            &context.sum(&weighted_outputs, false)?,
            &context.sum(&weighted_state, false)?,
        )
    }
}

fn dot(left: &MlxArray, right: &MlxArray) -> f64 {
    let left_values = left.to_vec_f32().unwrap();
    let right_values = right.to_vec_f32().unwrap();
    let mut dot_product = 0.0_f64;
    for (left_value, right_value) in left_values.iter().zip(&right_values) {
        dot_product += f64::from(*left_value) * f64::from(*right_value);
    }
    dot_product
}

/// The loss change the borrowed gradient predicts along a fixed direction,
/// and the change a central difference measures, with the forward on
/// serving's fused kernel or on its one-token operations.
fn predicted_and_measured_change(serving_uses_fused_kernel: bool) -> (f64, f64) {
    const TOKEN_COUNT: i32 = 24;
    let runtime = initialize_test_runtime();
    let inputs = RecurrenceInputs::new(&runtime, TOKEN_COUNT, 0, 0.9);
    // Centred on zero, so a step along them keeps decays and rates in range.
    let directions = RecurrenceInputs::new(&runtime, TOKEN_COUNT, 41_000, 0.0).trainable();
    let gradients = loss_gradients(
        RecurrenceLoss::new(&runtime, &inputs, TOKEN_COUNT, serving_uses_fused_kernel),
        &inputs.trainable(),
    )
    .unwrap()
    .gradients;
    let mut predicted_change = 0.0_f64;
    for (gradient, direction) in gradients.iter().zip(&directions) {
        predicted_change += dot(gradient, direction);
    }

    let reference_loss =
        RecurrenceLoss::new(&runtime, &inputs, TOKEN_COUNT, serving_uses_fused_kernel);
    // The recurrence is smooth in every input, so a central difference at
    // 0.01 is the true derivative to about 1e-5 relative. The borrowed
    // backward pass differentiates MLX's recurrence, which differs from the
    // served fused kernel by about 8e-4 relative (measured 4.5e-4 here on
    // Metal); the gap for the one-token operations is printed beside it.
    let step = 1e-2_f32;
    let mut stepped_losses = Vec::with_capacity(2);
    for signed_step in [step, -step] {
        let mut stepped_arrays = Vec::with_capacity(5);
        for (start, direction) in inputs.trainable().iter().zip(&directions) {
            let step_offset = runtime.multiply_scalar(direction, signed_step).unwrap();
            stepped_arrays.push(runtime.add(start, &step_offset).unwrap());
        }
        let stepped_loss = reference_loss
            .build_loss(&runtime, &stepped_arrays)
            .unwrap();
        stepped_losses.push(f64::from(stepped_loss.to_vec_f32().unwrap()[0]));
    }
    let measured_change = (stepped_losses[0] - stepped_losses[1]) / (2.0 * f64::from(step));
    eprintln!(
        "[differentiable-gated-delta] fused_kernel={serving_uses_fused_kernel} predicted={predicted_change} measured={measured_change}"
    );
    (predicted_change, measured_change)
}

/// The borrowed gradient agrees with the loss's own change to within
/// `1e-3` relative, along a direction that moves the loss.
fn assert_gradient_matches(predicted_change: f64, measured_change: f64) {
    assert!(
        predicted_change.abs() > 1e-3,
        "the direction should move the loss"
    );
    let relative_gap = (predicted_change - measured_change).abs() / measured_change.abs();
    assert!(
        relative_gap < 1e-3,
        "gradient predicts {predicted_change}, loss moved {measured_change}"
    );
}

#[test]
#[file_serial]
fn should_match_the_loss_change_along_a_direction_through_the_recurrence() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_match_the_loss_change_along_a_direction_through_the_recurrence",
    ) {
        return;
    }
    let (predicted_change, measured_change) = predicted_and_measured_change(true);
    assert_gradient_matches(predicted_change, measured_change);
}

#[test]
#[file_serial]
fn should_match_the_loss_change_along_a_direction_through_the_served_operations() {
    let (predicted_change, measured_change) = predicted_and_measured_change(false);
    assert_gradient_matches(predicted_change, measured_change);
}
