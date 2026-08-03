use std::collections::HashSet;
use std::io::Write;

use astronomical_model_serving::{
    TensorDtype, TensorProfile, validate_bounded_safetensors_with_partial_profiles,
};

#[test]
fn should_validate_profiled_tensors_and_accepted_extras_with_partial_profiles() {
    let profiled_tensor = TensorProfile {
        name: "language_model.weight".to_owned(),
        dtype: TensorDtype::Float32,
        shape: vec![2, 2],
    };
    let accepted_extra_names: HashSet<&str> = ["vision_tower.weight"].into_iter().collect();
    let weights_bytes = safetensors_bytes_with_multiple_tensors(&[
        ("language_model.weight", "F32", "[2,2]", &[0_u8; 16]),
        ("vision_tower.weight", "F32", "[1,1]", &[0_u8; 4]),
    ]);

    let metadata = validate_bounded_safetensors_with_partial_profiles(
        &file_from_bytes(&weights_bytes),
        weights_bytes.len() as u64,
        "model.safetensors",
        &[profiled_tensor],
        &accepted_extra_names,
    )
    .expect("profiled tensors with accepted extras should validate with partial profiles");

    assert_eq!(metadata.total_payload_bytes, 20);
}

#[test]
fn should_reject_a_profiled_tensor_with_the_wrong_dtype_using_partial_profiles() {
    let profiled_tensor = TensorProfile {
        name: "language_model.weight".to_owned(),
        dtype: TensorDtype::Float32,
        shape: vec![2, 2],
    };
    let accepted_extra_names: HashSet<&str> = HashSet::new();
    let weights_bytes = safetensors_bytes_with_multiple_tensors(&[(
        "language_model.weight",
        "F16",
        "[2,2]",
        &[0_u8; 8],
    )]);

    let validation_error = validate_bounded_safetensors_with_partial_profiles(
        &file_from_bytes(&weights_bytes),
        weights_bytes.len() as u64,
        "model.safetensors",
        &[profiled_tensor],
        &accepted_extra_names,
    )
    .expect_err("a profiled tensor with the wrong dtype must fail closed");

    assert!(validation_error.to_string().contains("dtype"));
}

#[test]
fn should_accept_bfloat16_and_float32_for_a_flexible_tensor_profile() {
    for (tensor_dtype, tensor_payload_bytes) in [("BF16", &[0_u8; 8][..]), ("F32", &[0_u8; 16][..])]
    {
        let profiled_tensor = TensorProfile {
            name: "language_model.A_log".to_owned(),
            dtype: TensorDtype::BFloat16OrFloat32,
            shape: vec![2, 2],
        };
        let weights_bytes = safetensors_bytes_with_multiple_tensors(&[(
            "language_model.A_log",
            tensor_dtype,
            "[2,2]",
            tensor_payload_bytes,
        )]);

        validate_bounded_safetensors_with_partial_profiles(
            &file_from_bytes(&weights_bytes),
            weights_bytes.len() as u64,
            "model.safetensors",
            &[profiled_tensor],
            &HashSet::new(),
        )
        .unwrap_or_else(|validation_error| {
            panic!("{tensor_dtype} should satisfy the flexible tensor profile: {validation_error}")
        });
    }
}

#[test]
fn should_reject_float16_for_a_bfloat16_or_float32_tensor_profile() {
    let profiled_tensor = TensorProfile {
        name: "language_model.A_log".to_owned(),
        dtype: TensorDtype::BFloat16OrFloat32,
        shape: vec![2, 2],
    };
    let weights_bytes = safetensors_bytes_with_multiple_tensors(&[(
        "language_model.A_log",
        "F16",
        "[2,2]",
        &[0_u8; 8],
    )]);

    let validation_error = validate_bounded_safetensors_with_partial_profiles(
        &file_from_bytes(&weights_bytes),
        weights_bytes.len() as u64,
        "model.safetensors",
        &[profiled_tensor],
        &HashSet::new(),
    )
    .expect_err("float16 must not satisfy a BF16-or-F32 tensor profile");

    assert!(validation_error.to_string().contains("dtype"));
}

#[test]
fn should_reject_a_profiled_tensor_with_the_wrong_shape_using_partial_profiles() {
    let profiled_tensor = TensorProfile {
        name: "language_model.weight".to_owned(),
        dtype: TensorDtype::Float32,
        shape: vec![2, 2],
    };
    let accepted_extra_names: HashSet<&str> = HashSet::new();
    let weights_bytes = safetensors_bytes_with_multiple_tensors(&[(
        "language_model.weight",
        "F32",
        "[4,1]",
        &[0_u8; 16],
    )]);

    let validation_error = validate_bounded_safetensors_with_partial_profiles(
        &file_from_bytes(&weights_bytes),
        weights_bytes.len() as u64,
        "model.safetensors",
        &[profiled_tensor],
        &accepted_extra_names,
    )
    .expect_err("a profiled tensor with the wrong shape must fail closed");

    assert!(validation_error.to_string().contains("shape"));
}

#[test]
fn should_reject_an_unexpected_tensor_that_is_neither_profiled_nor_accepted() {
    let profiled_tensor = TensorProfile {
        name: "language_model.weight".to_owned(),
        dtype: TensorDtype::Float32,
        shape: vec![2, 2],
    };
    let accepted_extra_names: HashSet<&str> = ["vision_tower.weight"].into_iter().collect();
    let weights_bytes = safetensors_bytes_with_multiple_tensors(&[
        ("language_model.weight", "F32", "[2,2]", &[0_u8; 16]),
        ("unknown_tensor.weight", "F32", "[1,1]", &[0_u8; 4]),
    ]);

    let validation_error = validate_bounded_safetensors_with_partial_profiles(
        &file_from_bytes(&weights_bytes),
        weights_bytes.len() as u64,
        "model.safetensors",
        &[profiled_tensor],
        &accepted_extra_names,
    )
    .expect_err("a tensor that is neither profiled nor accepted must fail closed");

    assert!(
        validation_error
            .to_string()
            .contains("unknown_tensor.weight")
    );
}

#[test]
fn should_reject_a_missing_profiled_tensor_using_partial_profiles() {
    let profiled_tensor = TensorProfile {
        name: "language_model.missing.weight".to_owned(),
        dtype: TensorDtype::Float32,
        shape: vec![2, 2],
    };
    let accepted_extra_names: HashSet<&str> = ["vision_tower.weight"].into_iter().collect();
    let weights_bytes = safetensors_bytes_with_multiple_tensors(&[(
        "vision_tower.weight",
        "F32",
        "[1,1]",
        &[0_u8; 4],
    )]);

    let validation_error = validate_bounded_safetensors_with_partial_profiles(
        &file_from_bytes(&weights_bytes),
        weights_bytes.len() as u64,
        "model.safetensors",
        &[profiled_tensor],
        &accepted_extra_names,
    )
    .expect_err("a missing profiled tensor must fail closed");

    assert!(
        validation_error
            .to_string()
            .contains("language_model.missing.weight")
    );
}

fn file_from_bytes(weights_bytes: &[u8]) -> std::fs::File {
    let mut weights_file = tempfile::tempfile().expect("the test should create a temporary file");
    weights_file
        .write_all(weights_bytes)
        .expect("the test should write the safetensors file");
    weights_file
}

fn safetensors_bytes_with_multiple_tensors(
    tensor_definitions: &[(&str, &str, &str, &[u8])],
) -> Vec<u8> {
    let mut all_payload_bytes = Vec::new();
    let mut tensor_entries = Vec::new();
    let mut current_offset = 0_usize;
    for (tensor_name, tensor_dtype, tensor_shape_json, tensor_payload_bytes) in tensor_definitions {
        let start_offset = current_offset;
        let end_offset = start_offset + tensor_payload_bytes.len();
        all_payload_bytes.extend_from_slice(tensor_payload_bytes);
        tensor_entries.push(format!(
            r#""{tensor_name}":{{"dtype":"{tensor_dtype}","shape":{tensor_shape_json},"data_offsets":[{start_offset},{end_offset}]}}"#
        ));
        current_offset = end_offset;
    }
    let header = format!("{{{}}}", tensor_entries.join(","));
    let mut weights_bytes = Vec::new();
    weights_bytes.extend_from_slice(
        &u64::try_from(header.len())
            .expect("the multi-tensor header length should fit u64")
            .to_le_bytes(),
    );
    weights_bytes.extend_from_slice(header.as_bytes());
    weights_bytes.extend_from_slice(&all_payload_bytes);
    weights_bytes
}
