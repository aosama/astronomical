//! The fused DeltaNet sequence kernel against the composed step loop it
//! replaces, on pseudo-random inputs near zero and far from it.
//!
//! The older comparison in `gated_delta.rs` uses patterned inputs no larger
//! than a few hundredths. The one-token DeltaNet kernel's wrong exponential
//! (issue #2599) agreed with the composed steps on such inputs and parted
//! from them only far from zero, so every fused kernel gets a sweep there.
//! This kernel has no exponential of its own: its decays and update rates
//! arrive computed. It adds a token's products in its own order, so it is
//! held to rounding, not to equal bits.
//!
//! No model on disk; under a second.

use model_serving::{qwen3_5_gated_delta_kernel, qwen3_5_gated_delta_sequence};
use runtime_integration::MlxRuntime;

use super::super::synthetic_values::{
    largest_relative_difference, swept_unit_interval_values, swept_values,
};
use super::gated_delta::{apply_ops_loop, gated_delta_runtime};

const TOKEN_COUNT: i32 = 3;
const KEY_HEAD_COUNT: i32 = 16;
const VALUE_HEAD_COUNT: i32 = 32;
const HEAD_DIMENSION: i32 = 128;
const SWEPT_INPUT_MAGNITUDES: [f32; 4] = [0.05, 0.5, 3.0, 20.0];
const SWEEP_TRIALS_A_MAGNITUDE: usize = 4;
/// Float32 keeps about seven digits; three tokens of 128-wide sums in a
/// different order move the last two. A wrong decay or update moves the
/// first.
const LARGEST_RELATIVE_DIFFERENCE: f32 = 1e-5;

#[tokio::test]
async fn should_match_the_composed_step_loop_to_rounding_on_inputs_far_from_zero() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = gated_delta_runtime();
    let gated_delta_kernel =
        qwen3_5_gated_delta_kernel().expect("the fused gated-delta kernel should construct");
    let mut generator_state = 0x9E37_79B9_7F4A_7C15_u64;
    for input_magnitude in SWEPT_INPUT_MAGNITUDES {
        for trial_index in 0..SWEEP_TRIALS_A_MAGNITUDE {
            let (output_difference, state_difference) = swept_trial_differences(
                &runtime,
                &gated_delta_kernel,
                input_magnitude,
                &mut generator_state,
            );
            eprintln!(
                "[gated-delta-sweep] magnitude={input_magnitude} trial={trial_index} output_relative_difference={:e} output_differing_elements={} state_relative_difference={:e} state_differing_elements={}",
                output_difference.0, output_difference.1, state_difference.0, state_difference.1
            );
            assert!(
                output_difference.0 <= LARGEST_RELATIVE_DIFFERENCE
                    && state_difference.0 <= LARGEST_RELATIVE_DIFFERENCE,
                "at magnitude {input_magnitude} the kernel parts from the composed steps by more than rounding: outputs {:e}, state {:e}",
                output_difference.0,
                state_difference.0
            );
        }
    }
}

/// One trial's largest relative differences and differing element counts,
/// for the outputs and for the recurrent state the sequence leaves.
fn swept_trial_differences(
    runtime: &MlxRuntime,
    gated_delta_kernel: &mlx_bridge::MlxMetalKernel,
    input_magnitude: f32,
    generator_state: &mut u64,
) -> ((f32, usize), (f32, usize)) {
    let key_shape = [1, TOKEN_COUNT, KEY_HEAD_COUNT, HEAD_DIMENSION];
    let value_shape = [1, TOKEN_COUNT, VALUE_HEAD_COUNT, HEAD_DIMENSION];
    let rate_shape = [1, TOKEN_COUNT, VALUE_HEAD_COUNT];
    let state_shape = [1, VALUE_HEAD_COUNT, HEAD_DIMENSION, HEAD_DIMENSION];
    let queries = swept_array(runtime, &key_shape, input_magnitude, generator_state);
    let keys = swept_array(runtime, &key_shape, input_magnitude, generator_state);
    let values = swept_array(runtime, &value_shape, input_magnitude, generator_state);
    // The kernel and the loop each take the starting state, so it is built
    // twice from the same values.
    let state_values = swept_values(
        element_count(&state_shape),
        input_magnitude,
        generator_state,
    );
    let fused_starting_state = f32_array(runtime, &state_values, &state_shape);
    let loop_starting_state = f32_array(runtime, &state_values, &state_shape);
    let decays = f32_array(
        runtime,
        &swept_unit_interval_values(element_count(&rate_shape), generator_state),
        &rate_shape,
    );
    let update_rates = f32_array(
        runtime,
        &swept_unit_interval_values(element_count(&rate_shape), generator_state),
        &rate_shape,
    );
    let (fused_outputs, fused_state) = qwen3_5_gated_delta_sequence(
        runtime,
        Some(gated_delta_kernel),
        &queries,
        &keys,
        &values,
        &decays,
        &update_rates,
        &fused_starting_state,
    )
    .expect("the fused gated-delta kernel should build a valid graph");
    let (loop_outputs, loop_state) = apply_ops_loop(
        runtime,
        &queries,
        &keys,
        &values,
        &decays,
        &update_rates,
        loop_starting_state,
        TOKEN_COUNT,
        KEY_HEAD_COUNT,
        VALUE_HEAD_COUNT,
        HEAD_DIMENSION,
    );
    (
        largest_relative_difference(&evaluated(&fused_outputs), &evaluated(&loop_outputs)),
        largest_relative_difference(&evaluated(&fused_state), &evaluated(&loop_state)),
    )
}

fn element_count(shape: &[i32]) -> usize {
    shape.iter().product::<i32>() as usize
}

fn f32_array(runtime: &MlxRuntime, array_values: &[f32], shape: &[i32]) -> mlx_bridge::MlxArray {
    runtime
        .array_from_f32(array_values, shape)
        .expect("a swept array should be valid")
}

fn swept_array(
    runtime: &MlxRuntime,
    shape: &[i32],
    input_magnitude: f32,
    generator_state: &mut u64,
) -> mlx_bridge::MlxArray {
    f32_array(
        runtime,
        &swept_values(element_count(shape), input_magnitude, generator_state),
        shape,
    )
}

fn evaluated(array: &mlx_bridge::MlxArray) -> Vec<f32> {
    array.to_vec_f32().expect("the array evaluates")
}
