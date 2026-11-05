use super::*;

fn f32_bytes(float_values: &[f32]) -> Vec<u8> {
    let mut tensor_bytes = Vec::new();
    for float_value in float_values {
        tensor_bytes.extend_from_slice(&float_value.to_le_bytes());
    }
    tensor_bytes
}

fn bfloat16_bytes(float_values: &[f32]) -> Vec<u8> {
    let mut tensor_bytes = Vec::new();
    for float_value in float_values {
        tensor_bytes.extend_from_slice(&((float_value.to_bits() >> 16) as u16).to_le_bytes());
    }
    tensor_bytes
}

fn classify_f32(tensor_name: &str, first: &[f32], second: &[f32]) -> TensorAgreement {
    classify_tensor_pair(
        tensor_name,
        TensorSide {
            data_type: "F32",
            tensor_bytes: &f32_bytes(first),
        },
        TensorSide {
            data_type: "F32",
            tensor_bytes: &f32_bytes(second),
        },
    )
}

#[test]
fn should_call_equal_bytes_identical() {
    assert_eq!(
        classify_f32("a", &[0.1, 0.2], &[0.1, 0.2]),
        TensorAgreement::Identical
    );
}

#[test]
fn should_see_through_a_precision_difference() {
    let values = [0.123_f32, -4.56, 7.89];
    let agreement = classify_tensor_pair(
        "a",
        TensorSide {
            data_type: "F32",
            tensor_bytes: &f32_bytes(&values),
        },
        TensorSide {
            data_type: "BF16",
            tensor_bytes: &bfloat16_bytes(&values),
        },
    );
    assert_eq!(agreement, TensorAgreement::EqualAfterRounding);
}

#[test]
fn should_accept_the_norm_offset_in_either_direction() {
    let stored = [0.01_f32, -0.02, 0.03];
    let shifted = [1.01_f32, 0.98, 1.03];
    assert_eq!(
        classify_f32("layer.norm.weight", &shifted, &stored),
        TensorAgreement::OffsetByOne
    );
    assert_eq!(
        classify_f32("layer.norm.weight", &stored, &shifted),
        TensorAgreement::OffsetByOne
    );
}

#[test]
fn should_refuse_a_uniform_offset_outside_a_norm() {
    let agreement = classify_f32("mlp.weight", &[1.01, 0.98, 1.03], &[0.01, -0.02, 0.03]);
    assert_eq!(agreement, TensorAgreement::Different);
}

#[test]
fn should_refuse_an_offset_that_changes_direction_across_the_tensor() {
    let agreement = classify_f32("norm.weight", &[1.0, 0.0], &[0.0, 1.0]);
    assert_eq!(agreement, TensorAgreement::Different);
}

#[test]
fn should_call_different_values_different() {
    assert_eq!(
        classify_f32("a", &[0.5, 0.6], &[0.5, 0.7]),
        TensorAgreement::Different
    );
}

#[test]
fn should_not_count_constant_tensors_as_evidence() {
    assert_eq!(
        classify_f32("a", &[0.0, 0.0], &[0.0, 0.0]),
        TensorAgreement::Uninformative
    );
    assert_eq!(
        classify_f32("norm", &[1.0, 1.0], &[0.0, 0.0]),
        TensorAgreement::Uninformative
    );
    assert!(!TensorAgreement::Uninformative.is_shared());
}

#[test]
fn should_treat_a_constant_against_a_varied_tensor_as_different() {
    assert_eq!(
        classify_f32("a", &[0.0, 0.0], &[0.5, 0.6]),
        TensorAgreement::Different
    );
}

#[test]
fn should_call_mismatched_lengths_and_non_floats_different() {
    assert_eq!(
        classify_f32("a", &[0.1], &[0.1, 0.2]),
        TensorAgreement::Different
    );
    let agreement = classify_tensor_pair(
        "a",
        TensorSide {
            data_type: "U32",
            tensor_bytes: &[0; 4],
        },
        TensorSide {
            data_type: "U32",
            tensor_bytes: &[0; 4],
        },
    );
    assert_eq!(agreement, TensorAgreement::Different);
}

#[test]
fn should_never_call_not_a_number_shared() {
    let agreement = classify_tensor_pair(
        "norm",
        TensorSide {
            data_type: "F32",
            tensor_bytes: &f32_bytes(&[f32::NAN, 0.5]),
        },
        TensorSide {
            data_type: "BF16",
            tensor_bytes: &bfloat16_bytes(&[f32::NAN, 0.5]),
        },
    );
    assert!(!agreement.is_shared());
}
