//! The order, shapes and starting values of a whole-model training's parameters.

use serial_test::serial;

use super::*;
use crate::synthetic_weights_test_support::initialize_test_runtime;

fn small_layout() -> WholeModelLayout {
    WholeModelLayout {
        scorer_layout: ParameterLayout::new(&[
            ("scorer.layers.3.weight", vec![1, 4]),
            ("scorer.layers.3.bias", vec![1]),
        ]),
        adapter_targets: vec![
            AdapterTarget {
                name: "encoder.layers.0.attn.Wo".to_owned(),
                input_width: 4,
                output_width: 6,
            },
            AdapterTarget {
                name: "head.layers.0.linear1".to_owned(),
                input_width: 16,
                output_width: 2,
            },
        ],
        adapter_rank: 2,
    }
}

#[test]
#[serial]
fn should_start_the_scorer_as_served_and_every_add_on_switched_off() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let layout = small_layout();

    let parameters = layout
        .starting_parameters(math, &[0.5, -0.25, 1.0, 2.0, 0.125])
        .unwrap();

    assert_eq!(parameters.len(), 2 + 2 * 2);
    assert_eq!(
        parameters[0].to_vec_f32().unwrap(),
        vec![0.5, -0.25, 1.0, 2.0]
    );
    assert_eq!(parameters[2].shape(), vec![2, 4]);
    assert_eq!(parameters[3].shape(), vec![6, 2]);
    assert!(
        parameters[3]
            .to_vec_f32()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
    assert_eq!(parameters[4].shape(), vec![2, 16]);
    let down_values = parameters[4].to_vec_f32().unwrap();
    // Within ±1/√16 and not all zero, or neither half would ever move.
    assert!(down_values.iter().all(|value| value.abs() <= 0.25));
    assert!(down_values.iter().any(|value| *value != 0.0));
}

#[test]
#[serial]
fn should_start_every_training_from_the_same_add_ons() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let layout = small_layout();
    let scorer_values = [0.0_f32; 5];

    let first = layout.starting_parameters(math, &scorer_values).unwrap();
    let second = layout.starting_parameters(math, &scorer_values).unwrap();

    assert_eq!(
        first[2].to_vec_f32().unwrap(),
        second[2].to_vec_f32().unwrap()
    );
    // Each layer's add-on starts from its own values.
    assert_ne!(
        first[2].to_vec_f32().unwrap()[..8],
        first[4].to_vec_f32().unwrap()[..8]
    );
}

#[test]
#[serial]
fn should_hand_the_forward_pass_the_scorer_by_name_and_the_add_ons_by_layer() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let layout = small_layout();
    let parameters = layout.starting_parameters(math, &[0.0; 5]).unwrap();

    let trained_parts = layout.trained_parts(&parameters).unwrap();

    assert_eq!(
        trained_parts.tuned_tensors.keys().collect::<Vec<_>>(),
        vec!["scorer.layers.3.bias", "scorer.layers.3.weight"]
    );
    assert_eq!(
        trained_parts.adapters.targets().collect::<Vec<_>>(),
        vec!["encoder.layers.0.attn.Wo", "head.layers.0.linear1"]
    );
}

#[test]
fn should_export_the_scorer_under_shipped_names_and_add_ons_as_down_and_up() {
    let layout = small_layout();
    let parameter_values = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0],
        vec![0.5; 8],
        vec![0.25; 12],
        vec![0.125; 32],
        vec![-1.0; 4],
    ];

    let (scorer_tensors, adapter_tensors) = layout.exported_tensors(parameter_values);

    assert_eq!(scorer_tensors[0].name, "scorer.layers.3.weight");
    assert_eq!(scorer_tensors[1].values.0, vec![5.0]);
    let adapter_names: Vec<&str> = adapter_tensors
        .iter()
        .map(|adapter_tensor| adapter_tensor.name.as_str())
        .collect();
    assert_eq!(
        adapter_names,
        vec![
            "encoder.layers.0.attn.Wo.adapter_down",
            "encoder.layers.0.attn.Wo.adapter_up",
            "head.layers.0.linear1.adapter_down",
            "head.layers.0.linear1.adapter_up",
        ]
    );
    assert_eq!(adapter_tensors[1].shape, vec![6, 2]);
    assert_eq!(adapter_tensors[3].values.0, vec![-1.0; 4]);
}
