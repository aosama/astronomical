//! Unit contracts for reading tensor bytes as numbers and writing them back
//! in half precision, for both element types a Laya checkpoint holds.

use super::super::half_float::f32_to_half_bits;
use super::super::safetensors_file::TensorElementType;
use super::{float32_values, half_bits_bytes, half_precision_bytes};

fn single_precision_bytes(values: &[f32]) -> Vec<u8> {
    let mut single_bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        single_bytes.extend_from_slice(&value.to_le_bytes());
    }
    single_bytes
}

#[test]
fn should_decode_half_precision_bytes_into_their_values() {
    let half_bytes = half_bits_bytes(&[0x3C00, 0xC000, 0x3800]);

    let weights = float32_values(&half_bytes, TensorElementType::Float16)
        .expect("half-precision bytes should decode");

    assert_eq!(weights, vec![1.0, -2.0, 0.5]);
}

#[test]
fn should_decode_single_precision_bytes_into_their_values() {
    let weights = float32_values(
        &single_precision_bytes(&[0.1, -3.25, 1.0e-6]),
        TensorElementType::Float32,
    )
    .expect("single-precision bytes should decode");

    assert_eq!(weights, vec![0.1, -3.25, 1.0e-6]);
}

#[test]
fn should_refuse_to_read_an_integer_tensor_as_weights() {
    assert!(float32_values(&[0; 8], TensorElementType::Unsigned32).is_err());
}

#[test]
fn should_pass_half_precision_bytes_through_unchanged() {
    let half_bytes = half_bits_bytes(&[0x3C00, 0x7BFF, 0x0001]);

    let written_bytes = half_precision_bytes(half_bytes.clone(), TensorElementType::Float16)
        .expect("half-precision bytes should pass through");

    assert_eq!(written_bytes, half_bytes);
}

#[test]
fn should_round_single_precision_values_to_the_nearest_half() {
    let source_values = [0.1_f32, -3.25, 70_000.0];

    let written_bytes = half_precision_bytes(
        single_precision_bytes(&source_values),
        TensorElementType::Float32,
    )
    .expect("single-precision bytes should convert");

    let expected_bits: Vec<u16> = source_values
        .iter()
        .map(|source_value| f32_to_half_bits(*source_value))
        .collect();
    assert_eq!(written_bytes, half_bits_bytes(&expected_bits));
    assert_eq!(
        &written_bytes[4..],
        &[0x00, 0x7C],
        "a value beyond the largest half becomes infinity"
    );
}

#[test]
fn should_write_half_bits_lowest_byte_first() {
    assert_eq!(
        half_bits_bytes(&[0x1234, 0xABCD]),
        vec![0x34, 0x12, 0xCD, 0xAB]
    );
    assert!(half_bits_bytes(&[]).is_empty());
}
