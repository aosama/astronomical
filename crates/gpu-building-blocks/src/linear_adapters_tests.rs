//! The add-on path of a linear layer, on small known values.

use astronomical_decision_contract::decision_adapters_file::{DecisionAdapters, LinearAdapter};
use serial_test::serial;

use super::*;
use crate::mlx_test_runtime::initialize_test_runtime;

/// One add-on of rank 1 on a layer from 2 inputs to 3 outputs.
fn rank_one_adapter(up_values: Vec<f32>) -> DecisionAdapters {
    DecisionAdapters {
        adapters: vec![LinearAdapter {
            target: "layers.0.attn.Wo".to_owned(),
            rank: 1,
            input_width: 2,
            output_width: 3,
            down_values: vec![2.0, -1.0],
            up_values,
        }],
    }
}

#[test]
#[serial]
fn should_add_up_times_down_times_input_to_the_layer_output() {
    let runtime = initialize_test_runtime();
    let adapters =
        LinearAdapters::upload(&runtime, &rank_one_adapter(vec![1.0, 0.5, -2.0])).unwrap();
    let layer_input = runtime.array_from_f32(&[3.0, 1.0], &[1, 2]).unwrap();
    let base_output = runtime
        .array_from_f32(&[10.0, 20.0, 30.0], &[1, 3])
        .unwrap();

    let adapted = adapters
        .adapted_output(&runtime, "layers.0.attn.Wo", &layer_input, base_output)
        .unwrap();

    // down · input = 2·3 − 1·1 = 5; up · 5 = [5, 2.5, −10].
    assert_eq!(adapted.to_vec_f32().unwrap(), vec![15.0, 22.5, 20.0]);
}

#[test]
#[serial]
fn should_leave_the_layer_output_exactly_as_it_was_while_up_is_zero() {
    let runtime = initialize_test_runtime();
    let adapters = LinearAdapters::upload(&runtime, &rank_one_adapter(vec![0.0; 3])).unwrap();
    let layer_input = runtime.array_from_f32(&[3.0, 1.0], &[1, 2]).unwrap();
    let base_values = [0.1_f32, -7.25, 1.0e-30];
    let base_output = runtime.array_from_f32(&base_values, &[1, 3]).unwrap();

    let adapted = adapters
        .adapted_output(&runtime, "layers.0.attn.Wo", &layer_input, base_output)
        .unwrap();

    assert_eq!(adapted.to_vec_f32().unwrap(), base_values.to_vec());
}

#[test]
#[serial]
fn should_pass_a_layer_without_an_add_on_through_unchanged() {
    let runtime = initialize_test_runtime();
    let adapters =
        LinearAdapters::upload(&runtime, &rank_one_adapter(vec![1.0, 1.0, 1.0])).unwrap();
    let layer_input = runtime.array_from_f32(&[3.0, 1.0], &[1, 2]).unwrap();
    let base_output = runtime.array_from_f32(&[4.0, 5.0], &[1, 2]).unwrap();

    let adapted = adapters
        .adapted_output(&runtime, "layers.0.attn.Wqkv", &layer_input, base_output)
        .unwrap();

    assert_eq!(adapted.to_vec_f32().unwrap(), vec![4.0, 5.0]);
}

#[test]
#[serial]
fn should_pair_trainable_arrays_with_their_targets_in_order() {
    let runtime = initialize_test_runtime();
    let targets = vec!["first".to_owned(), "second".to_owned()];
    let trainable_arrays = vec![
        runtime.array_from_f32(&[1.0], &[1, 1]).unwrap(),
        runtime.array_from_f32(&[2.0], &[1, 1]).unwrap(),
        runtime.array_from_f32(&[3.0], &[1, 1]).unwrap(),
        runtime.array_from_f32(&[4.0], &[1, 1]).unwrap(),
    ];
    let layer_input = runtime.array_from_f32(&[1.0], &[1, 1]).unwrap();

    let adapters = LinearAdapters::from_trainable_arrays(&targets, &trainable_arrays).unwrap();
    let second_output = adapters
        .adapted_output(
            &runtime,
            "second",
            &layer_input,
            runtime.array_from_f32(&[0.0], &[1, 1]).unwrap(),
        )
        .unwrap();

    assert_eq!(
        adapters.targets().collect::<Vec<_>>(),
        vec!["first", "second"]
    );
    // The second target's down is 3 and its up is 4.
    assert_eq!(second_output.to_vec_f32().unwrap(), vec![12.0]);
}

#[test]
fn should_have_no_add_ons_when_built_empty() {
    assert!(LinearAdapters::default().is_empty());
}

#[test]
#[serial]
fn should_keep_a_bfloat16_layer_in_bfloat16_and_leave_it_exact_with_a_zero_add_on() {
    let runtime = initialize_test_runtime();
    let adapters =
        LinearAdapters::upload(&runtime, &rank_one_adapter(vec![0.0, 0.0, 0.0])).unwrap();
    let layer_input = runtime
        .astype(
            &runtime.array_from_f32(&[3.0, 1.0], &[1, 2]).unwrap(),
            mlx_bridge::MlxDtype::BFloat16,
        )
        .unwrap();
    let base_output = runtime
        .astype(
            &runtime
                .array_from_f32(&[10.125, -0.3, 7.0], &[1, 3])
                .unwrap(),
            mlx_bridge::MlxDtype::BFloat16,
        )
        .unwrap();
    let base_values = runtime
        .astype(&base_output, mlx_bridge::MlxDtype::Float32)
        .unwrap()
        .to_vec_f32()
        .unwrap();

    let adapted = adapters
        .adapted_output(&runtime, "layers.0.attn.Wo", &layer_input, base_output)
        .unwrap();

    assert_eq!(adapted.dtype(), mlx_bridge::MlxDtype::BFloat16);
    let adapted_values = runtime
        .astype(&adapted, mlx_bridge::MlxDtype::Float32)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert_eq!(adapted_values, base_values);
}
