//! Unit tests for reading named tensors out of a safetensors file.

use safetensors::Dtype;
use safetensors::tensor::TensorView;

use super::*;

fn float32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut encoded_bytes = Vec::with_capacity(values.len() * 4);
    for float_value in values {
        encoded_bytes.extend_from_slice(&float_value.to_le_bytes());
    }
    encoded_bytes
}

fn half_le_bytes(half_values: &[u16]) -> Vec<u8> {
    let mut encoded_bytes = Vec::with_capacity(half_values.len() * 2);
    for half_bits in half_values {
        encoded_bytes.extend_from_slice(&half_bits.to_le_bytes());
    }
    encoded_bytes
}

fn write_weights_file(directory: &Path) -> PathBuf {
    let float32_bytes = float32_le_bytes(&[1.5, -2.0, 0.25, 8.0]);
    // 1.0, -2.0, 65504 and the smallest subnormal, in IEEE half precision.
    let float16_bytes = half_le_bytes(&[0x3c00, 0xc000, 0x7bff, 0x0001]);
    // 1.0 and -0.5 in bfloat16.
    let bfloat16_bytes = half_le_bytes(&[0x3f80, 0xbf00]);
    let integer_bytes = 7_i32.to_le_bytes().to_vec();
    let views = vec![
        (
            "scorer.layers.1.weight",
            TensorView::new(Dtype::F32, vec![2, 2], &float32_bytes).unwrap(),
        ),
        (
            "scorer.layers.0.weight",
            TensorView::new(Dtype::F16, vec![4], &float16_bytes).unwrap(),
        ),
        (
            "residual_gate",
            TensorView::new(Dtype::BF16, vec![2], &bfloat16_bytes).unwrap(),
        ),
        (
            "position_ids",
            TensorView::new(Dtype::I32, vec![1], &integer_bytes).unwrap(),
        ),
    ];
    let file_path = directory.join("model.safetensors");
    std::fs::write(&file_path, safetensors::serialize(views, None).unwrap()).unwrap();
    file_path
}

#[test]
fn should_read_only_the_named_tensors_widened_exactly() {
    let directory = tempfile::tempdir().unwrap();
    let file_path = write_weights_file(directory.path());
    let tensors = read_named_float_tensors(
        &file_path,
        &[
            "scorer.layers.0.weight",
            "residual_gate",
            "scorer.layers.1.weight",
        ],
    )
    .unwrap();
    assert_eq!(
        tensors[0].values,
        vec![1.0, -2.0, 65_504.0, 2.0_f32.powi(-24)]
    );
    assert_eq!(tensors[1].values, vec![1.0, -0.5]);
    assert_eq!(tensors[2].shape, vec![2, 2]);
    assert_eq!(tensors[2].values, vec![1.5, -2.0, 0.25, 8.0]);
}

#[test]
fn should_name_a_missing_tensor_and_refuse_an_integer_one() {
    let directory = tempfile::tempdir().unwrap();
    let file_path = write_weights_file(directory.path());
    assert!(matches!(
        read_named_float_tensors(&file_path, &["q.weight"]),
        Err(SafetensorsSliceError::MissingTensor { tensor_name, .. }) if tensor_name == "q.weight"
    ));
    assert!(matches!(
        read_named_float_tensors(&file_path, &["position_ids"]),
        Err(SafetensorsSliceError::UnreadableDtype { .. })
    ));
}

#[test]
fn should_refuse_a_file_that_is_not_safetensors() {
    let directory = tempfile::tempdir().unwrap();
    let file_path = directory.path().join("model.safetensors");
    std::fs::write(&file_path, b"Romeo, Romeo").unwrap();
    assert!(read_named_float_tensors(&file_path, &["q.weight"]).is_err());
    assert!(matches!(
        read_named_float_tensors(&directory.path().join("absent.safetensors"), &[]),
        Err(SafetensorsSliceError::Io { .. })
    ));
}

#[test]
fn should_widen_half_precision_specials() {
    assert_eq!(float16_to_float32(0x0000), 0.0);
    assert!(float16_to_float32(0x8000).is_sign_negative());
    assert_eq!(float16_to_float32(0x7c00), f32::INFINITY);
    assert!(float16_to_float32(0x7e00).is_nan());
    assert_eq!(float16_to_float32(0x0200), 2.0_f32.powi(-15));
    assert_eq!(float16_to_float32(0x3555), 0.333_251_95);
}
