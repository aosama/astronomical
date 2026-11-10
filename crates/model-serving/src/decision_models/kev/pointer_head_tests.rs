//! Unit contracts for Kev's pointer head and its temperature softmax.

use std::collections::HashMap;

use safetensors::tensor::TensorView;
use safetensors::{Dtype, serialize};

use super::{KevPointerHead, kev_probabilities};

fn float32_bytes(values: &[f32]) -> Vec<u8> {
    let mut encoded_bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        encoded_bytes.extend(value.to_le_bytes());
    }
    encoded_bytes
}

/// A head of pointer dimension 2 over a hidden size of 3, given its four tensors.
fn head_bytes(
    query_weight: &[f32],
    query_bias: &[f32],
    key_weight: &[f32],
    key_bias: &[f32],
) -> Vec<u8> {
    let encoded_tensors = [
        ("q.weight", float32_bytes(query_weight), vec![2, 3]),
        ("q.bias", float32_bytes(query_bias), vec![2]),
        ("k.weight", float32_bytes(key_weight), vec![2, 3]),
        ("k.bias", float32_bytes(key_bias), vec![2]),
    ];
    let mut tensor_views = HashMap::new();
    for (tensor_name, tensor_bytes, shape) in &encoded_tensors {
        tensor_views.insert(
            (*tensor_name).to_owned(),
            TensorView::new(Dtype::F32, shape.clone(), tensor_bytes)
                .expect("the test tensor should be valid"),
        );
    }
    serialize(tensor_views, None).expect("the test head should serialize")
}

#[test]
fn should_score_each_option_as_the_scaled_dot_product_of_its_projections() {
    let pointer_head = KevPointerHead::from_safetensors_bytes(
        &head_bytes(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0.0, 1.0],
            &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0],
        ),
        2,
    )
    .expect("a well-formed head should read");

    // q = [h0, h1 + 1] = [2, 4]; option k = [h2 + 1, h0]; logit = k · q / √2.
    let option_logits =
        pointer_head.option_logits(&[2.0, 3.0, 0.0], &[&[1.0, 0.0, 2.0], &[0.0, 5.0, -1.0]]);

    let scale = 1.0 / 2.0_f32.sqrt();
    assert_eq!(pointer_head.hidden_size(), 3);
    assert_eq!(option_logits, vec![(3.0 * 2.0 + 1.0 * 4.0) * scale, 0.0]);
}

#[test]
fn should_refuse_a_head_whose_shapes_disagree_or_that_is_not_float32() {
    let mismatched_bias = head_bytes(&[0.0; 6], &[0.0, 0.0], &[0.0; 6], &[0.0, 0.0]);

    assert!(KevPointerHead::from_safetensors_bytes(&mismatched_bias, 3).is_err());
    assert!(KevPointerHead::from_safetensors_bytes(b"not safetensors", 2).is_err());
}

#[test]
fn should_divide_by_the_temperature_without_clamping_it() {
    let probabilities = kev_probabilities(&[2.0, 0.0], 0.1);

    // softmax([20, 0]): a clamp to 0.5 would give softmax([4, 0]) instead.
    let expected_top = 1.0 / (1.0 + (-20.0_f64).exp());
    assert!((probabilities[0] - expected_top).abs() < 1e-12);
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn should_give_equal_logits_equal_probability_at_any_temperature() {
    assert_eq!(
        kev_probabilities(&[1.5, 1.5, 1.5], 2.35),
        vec![1.0 / 3.0; 3]
    );
    assert_eq!(kev_probabilities(&[7.0], 1.0), vec![1.0]);
}
