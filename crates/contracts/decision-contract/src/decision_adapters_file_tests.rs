//! Unit tests for the add-on weights file.

use safetensors::Dtype;
use safetensors::tensor::TensorView;

use super::*;

fn two_adapters() -> DecisionAdapters {
    DecisionAdapters {
        adapters: vec![
            LinearAdapter {
                target: "layers.0.attn.Wo".to_owned(),
                rank: 1,
                input_width: 3,
                output_width: 2,
                down_values: vec![0.5, -1.0, 2.0],
                up_values: vec![1.0e-30, -3.5],
            },
            LinearAdapter {
                target: "layers.0.attn.Wqkv".to_owned(),
                rank: 2,
                input_width: 2,
                output_width: 1,
                down_values: vec![1.0, 2.0, 3.0, 4.0],
                up_values: vec![0.0, 0.25],
            },
        ],
    }
}

fn float32_file(named_tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut tensor_bytes = Vec::new();
    for (_, _, values) in named_tensors {
        let mut little_endian_bytes = Vec::new();
        for float_value in values {
            little_endian_bytes.extend_from_slice(&float_value.to_le_bytes());
        }
        tensor_bytes.push(little_endian_bytes);
    }
    let mut tensor_views = Vec::new();
    for ((tensor_name, shape, _), little_endian_bytes) in named_tensors.iter().zip(&tensor_bytes) {
        tensor_views.push((
            *tensor_name,
            TensorView::new(Dtype::F32, shape.clone(), little_endian_bytes).unwrap(),
        ));
    }
    safetensors::serialize(tensor_views, None).unwrap()
}

#[test]
fn should_read_back_exactly_the_add_ons_it_wrote() {
    let decision_adapters = two_adapters();

    let file_bytes = decision_adapters_bytes(&decision_adapters).unwrap();

    assert_eq!(
        parse_decision_adapters(&file_bytes).unwrap(),
        decision_adapters
    );
}

#[test]
fn should_refuse_an_add_on_with_only_its_down_matrix() {
    let file_bytes = float32_file(&[("layers.2.mlp.Wi.adapter_down", vec![1, 2], vec![1.0, 2.0])]);

    assert_eq!(
        parse_decision_adapters(&file_bytes),
        Err(DecisionAdaptersFileError::MissingHalf(
            "layers.2.mlp.Wi".to_owned()
        ))
    );
}

#[test]
fn should_refuse_an_add_on_with_only_its_up_matrix() {
    let file_bytes = float32_file(&[("layers.2.mlp.Wi.adapter_up", vec![2, 1], vec![1.0, 2.0])]);

    assert_eq!(
        parse_decision_adapters(&file_bytes),
        Err(DecisionAdaptersFileError::MissingHalf(
            "layers.2.mlp.Wi".to_owned()
        ))
    );
}

#[test]
fn should_refuse_halves_that_belong_to_different_layers() {
    let file_bytes = float32_file(&[
        ("layers.1.mlp.Wi.adapter_down", vec![1, 2], vec![1.0, 2.0]),
        ("layers.1.mlp.Wo.adapter_up", vec![2, 1], vec![1.0, 2.0]),
    ]);

    assert_eq!(
        parse_decision_adapters(&file_bytes),
        Err(DecisionAdaptersFileError::MissingHalf(
            "layers.1.mlp.Wi".to_owned()
        ))
    );
}

#[test]
fn should_refuse_down_and_up_matrices_of_different_ranks() {
    let file_bytes = float32_file(&[
        ("layers.0.attn.Wo.adapter_down", vec![2, 1], vec![1.0, 2.0]),
        ("layers.0.attn.Wo.adapter_up", vec![1, 1], vec![1.0]),
    ]);

    assert_eq!(
        parse_decision_adapters(&file_bytes),
        Err(DecisionAdaptersFileError::RankMismatch(
            "layers.0.attn.Wo".to_owned()
        ))
    );
}

#[test]
fn should_refuse_a_rank_of_zero() {
    let file_bytes = float32_file(&[
        ("layers.0.attn.Wo.adapter_down", vec![0, 3], vec![]),
        ("layers.0.attn.Wo.adapter_up", vec![2, 0], vec![]),
    ]);

    assert_eq!(
        parse_decision_adapters(&file_bytes),
        Err(DecisionAdaptersFileError::RankMismatch(
            "layers.0.attn.Wo".to_owned()
        ))
    );
}

#[test]
fn should_refuse_a_tensor_that_is_not_an_add_on() {
    let file_bytes = float32_file(&[("scorer.layers.1.weight", vec![1, 1], vec![1.0])]);

    assert_eq!(
        parse_decision_adapters(&file_bytes),
        Err(DecisionAdaptersFileError::UnknownTensor(
            "scorer.layers.1.weight".to_owned()
        ))
    );
}

#[test]
fn should_refuse_a_matrix_stored_as_a_vector() {
    let file_bytes = float32_file(&[("layers.0.attn.Wo.adapter_down", vec![2], vec![1.0, 2.0])]);

    assert_eq!(
        parse_decision_adapters(&file_bytes),
        Err(DecisionAdaptersFileError::NotAMatrix(
            "layers.0.attn.Wo.adapter_down".to_owned()
        ))
    );
}

#[test]
fn should_refuse_a_value_that_is_not_finite() {
    let file_bytes = float32_file(&[
        ("layers.0.attn.Wo.adapter_down", vec![1, 1], vec![f32::NAN]),
        ("layers.0.attn.Wo.adapter_up", vec![1, 1], vec![1.0]),
    ]);

    assert_eq!(
        parse_decision_adapters(&file_bytes),
        Err(DecisionAdaptersFileError::Unreadable(
            Float32TensorsError::NonFiniteValue("layers.0.attn.Wo.adapter_down".to_owned())
        ))
    );
}

#[test]
fn should_refuse_bytes_that_are_not_a_safetensors_file() {
    assert!(matches!(
        parse_decision_adapters(b"not a weights file"),
        Err(DecisionAdaptersFileError::Unreadable(
            Float32TensorsError::NotSafetensors(_)
        ))
    ));
}
