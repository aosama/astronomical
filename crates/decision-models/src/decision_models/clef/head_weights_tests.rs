//! Unit tests for replacing Clef head tensors with trained ones.

use astronomical_decision_contract::head_tuning_file::TunedTensor;
use serial_test::serial;

use super::*;
use astronomical_gpu_building_blocks::synthetic_weights_test_support::initialize_test_runtime;

/// A head holding only the final mix's gate and a small residual bias.
fn small_head() -> ClefHeadWeights {
    let mut tensors_by_name = BTreeMap::new();
    tensors_by_name.insert(
        "residual_gate".to_owned(),
        MlxArray::from_f32(&[0.25], &[]).unwrap(),
    );
    tensors_by_name.insert(
        "residual_scorer.3.bias".to_owned(),
        MlxArray::from_f32(&[0.5], &[1]).unwrap(),
    );
    ClefHeadWeights { tensors_by_name }
}

fn tuning_of(tensor_name: &str, shape: Vec<usize>, values: Vec<f32>) -> HeadTuning {
    HeadTuning {
        tensors: vec![TunedTensor {
            name: tensor_name.to_owned(),
            shape,
            values,
        }],
    }
}

#[test]
#[serial]
fn should_read_a_trained_scalar_in_place_of_the_shipped_one() {
    let _runtime = initialize_test_runtime();
    let mut head_weights = small_head();
    head_weights
        .apply_tuning(&tuning_of("residual_gate", Vec::new(), vec![1.75]))
        .unwrap();
    assert_eq!(head_weights.scalar("residual_gate").unwrap(), 1.75);
    // An untouched tensor keeps its shipped value.
    let bias = head_weights
        .tensor("residual_scorer.3.bias")
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert_eq!(bias, vec![0.5]);
}

#[test]
#[serial]
fn should_refuse_a_trained_tensor_of_another_shape_or_one_the_head_lacks() {
    let _runtime = initialize_test_runtime();
    let mut head_weights = small_head();
    assert!(
        head_weights
            .apply_tuning(&tuning_of(
                "residual_scorer.3.bias",
                vec![2],
                vec![0.0, 1.0]
            ))
            .is_err()
    );
    assert!(
        head_weights
            .apply_tuning(&tuning_of("prior_logit_scale", Vec::new(), vec![1.0]))
            .is_err()
    );
    assert_eq!(head_weights.scalar("residual_gate").unwrap(), 0.25);
}

#[test]
#[serial]
fn should_change_nothing_with_an_empty_tuning() {
    let _runtime = initialize_test_runtime();
    let mut head_weights = small_head();
    head_weights.apply_tuning(&HeadTuning::default()).unwrap();
    assert_eq!(head_weights.scalar("residual_gate").unwrap(), 0.25);
}
