//! Turning a training's returned tensors into the files beside a model.

use super::*;

fn trained_tensor(name: &str, shape: Vec<usize>, values: Vec<f32>) -> TrainedTensor {
    TrainedTensor {
        name: name.to_owned(),
        shape,
        values: crate::FloatVector(values),
    }
}

#[test]
fn should_pair_each_layers_down_and_up_into_one_add_on() {
    let adapter_tensors = vec![
        trained_tensor(
            "encoder.layers.0.attn.Wo.adapter_down",
            vec![1, 2],
            vec![0.5, -0.5],
        ),
        trained_tensor(
            "encoder.layers.0.attn.Wo.adapter_up",
            vec![3, 1],
            vec![1.0, 2.0, 3.0],
        ),
    ];

    let Some(decision_adapters) = decision_adapters_of(&adapter_tensors).unwrap() else {
        panic!("two halves of one layer make one add-on");
    };

    assert_eq!(decision_adapters.adapters.len(), 1);
    let adapter = &decision_adapters.adapters[0];
    assert_eq!(adapter.target, "encoder.layers.0.attn.Wo");
    assert_eq!(
        (adapter.rank, adapter.input_width, adapter.output_width),
        (1, 2, 3)
    );
    assert_eq!(adapter.up_values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn should_write_no_add_ons_after_a_final_layer_training() {
    assert!(decision_adapters_of(&[]).unwrap().is_none());
}

#[test]
fn should_refuse_add_ons_that_do_not_pair_up() {
    let other_layers = vec![
        trained_tensor(
            "encoder.layers.0.attn.Wo.adapter_down",
            vec![1, 2],
            vec![0.5, -0.5],
        ),
        trained_tensor(
            "encoder.layers.1.attn.Wo.adapter_up",
            vec![3, 1],
            vec![1.0, 2.0, 3.0],
        ),
    ];
    let lone_half = vec![trained_tensor(
        "encoder.layers.0.attn.Wo.adapter_down",
        vec![1, 2],
        vec![0.5, -0.5],
    )];
    let different_ranks = vec![
        trained_tensor(
            "encoder.layers.0.attn.Wo.adapter_down",
            vec![1, 2],
            vec![0.5, -0.5],
        ),
        trained_tensor(
            "encoder.layers.0.attn.Wo.adapter_up",
            vec![1, 2],
            vec![1.0, 2.0],
        ),
    ];

    for unpaired in [other_layers, lone_half, different_ranks] {
        assert!(matches!(
            decision_adapters_of(&unpaired),
            Err(UnpairedAddOns)
        ));
    }
}

#[test]
fn should_keep_every_trained_tensor_under_its_shipped_name() {
    let head_tuning =
        head_tuning_of(&[trained_tensor("scorer.layers.3.bias", vec![1], vec![0.25])]);

    assert_eq!(head_tuning.tensors.len(), 1);
    assert_eq!(head_tuning.tensors[0].name, "scorer.layers.3.bias");
    assert_eq!(head_tuning.tensors[0].values, vec![0.25]);
    assert!(head_tuning_of(&[]).is_empty());
}
