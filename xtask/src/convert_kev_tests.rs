//! Component contracts for `cargo xtask convert-kev`: a small synthetic
//! adapter and base model are converted end to end and the output folder is
//! read back.

use std::path::Path;

use serde_json::json;

use super::{ConvertKevInvocation, KEV_HEAD_FILE_NAME, KEV_SETTINGS_FILE_NAME, convert_checkpoint};
use crate::convert_kev::test_support::{TestHeadSettings, kev_head_archive_bytes};
use crate::model_files::bfloat16::{bfloat16_bits_to_f32, f32_to_bfloat16_bits};
use crate::model_files::safetensors_file::{
    OutputCheckpoint, PlannedTensor, SourceCheckpoint, TensorElementType,
};

const BASE_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const ADAPTED_TENSOR_NAME: &str = "model.language_model.layers.0.mlp.up_proj.weight";
const UNTOUCHED_TENSOR_NAME: &str = "model.language_model.norm.weight";

fn write_safetensors(file_path: &Path, tensors: &[(&str, TensorElementType, Vec<u64>, Vec<u8>)]) {
    let mut planned_tensors = Vec::new();
    for (tensor_name, element_type, shape, _) in tensors {
        planned_tensors.push(PlannedTensor {
            tensor_name: (*tensor_name).to_owned(),
            element_type: *element_type,
            shape: shape.clone(),
        });
    }
    let output_file = std::fs::File::create(file_path).expect("the test file should be creatable");
    let mut output_checkpoint =
        OutputCheckpoint::begin(output_file, planned_tensors).expect("the header should write");
    for (tensor_name, _, _, tensor_bytes) in tensors {
        output_checkpoint
            .append_tensor(tensor_name, tensor_bytes)
            .expect("the tensor should match its plan");
    }
    output_checkpoint.finish().expect("the file should finish");
}

fn float32_bytes(values: &[f32]) -> Vec<u8> {
    let mut encoded_bytes = Vec::new();
    for value in values {
        encoded_bytes.extend(value.to_le_bytes());
    }
    encoded_bytes
}

fn bfloat16_bytes(values: &[f32]) -> Vec<u8> {
    let mut encoded_bytes = Vec::new();
    for value in values {
        encoded_bytes.extend(f32_to_bfloat16_bits(*value).to_le_bytes());
    }
    encoded_bytes
}

/// An adapter of rank 1 and alpha 2 (scale 2) on one 2×3 module, with a
/// head of width 2 over a hidden size of 3.
fn write_synthetic_adapter(adapter_directory: &Path) {
    std::fs::create_dir_all(adapter_directory).expect("the adapter folder should be creatable");
    std::fs::write(
        adapter_directory.join("adapter_config.json"),
        json!({"peft_type": "LORA", "r": 1, "lora_alpha": 2, "bias": "none"}).to_string(),
    )
    .expect("the adapter config should be writable");
    write_safetensors(
        &adapter_directory.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.layers.0.mlp.up_proj.lora_A.weight",
                TensorElementType::Float32,
                vec![1, 3],
                float32_bytes(&[1.0, 0.0, -1.0]),
            ),
            (
                "base_model.model.layers.0.mlp.up_proj.lora_B.weight",
                TensorElementType::Float32,
                vec![2, 1],
                float32_bytes(&[0.5, 0.25]),
            ),
        ],
    );
    std::fs::write(
        adapter_directory.join("head.pt"),
        kev_head_archive_bytes(&TestHeadSettings {
            pointer_dimension: 2,
            hidden_size: 3,
            temperature: 2.0,
            base_revision: BASE_REVISION,
            option_isolation: false,
        }),
    )
    .expect("the head should be writable");
    std::fs::write(
        adapter_directory.join("tokenizer.json"),
        r#"{"version":"1.0"}"#,
    )
    .expect("the tokenizer should be writable");
    std::fs::write(
        adapter_directory.join("tokenizer_config.json"),
        json!({"eos_token": "<|endoftext|>", "cache_dir": "/fictional-home/fictional-person/cache"}).to_string(),
    )
    .expect("the tokenizer settings should be writable");
}

fn write_synthetic_base(base_directory: &Path) {
    std::fs::create_dir_all(base_directory).expect("the base folder should be creatable");
    std::fs::write(
        base_directory.join("config.json"),
        r#"{"model_type":"qwen3_5"}"#,
    )
    .expect("the base config should be writable");
    write_safetensors(
        &base_directory.join("model.safetensors"),
        &[
            (
                ADAPTED_TENSOR_NAME,
                TensorElementType::BFloat16,
                vec![2, 3],
                bfloat16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            (
                UNTOUCHED_TENSOR_NAME,
                TensorElementType::BFloat16,
                vec![3],
                bfloat16_bytes(&[0.5, 0.25, 0.125]),
            ),
        ],
    );
}

fn invocation_in(
    temporary_directory: &tempfile::TempDir,
    base_folder_name: &str,
) -> ConvertKevInvocation {
    let invocation = ConvertKevInvocation {
        adapter_directory: temporary_directory.path().join("adapter"),
        base_directory: temporary_directory.path().join(base_folder_name),
        output_directory: temporary_directory.path().join("converted"),
    };
    write_synthetic_adapter(&invocation.adapter_directory);
    write_synthetic_base(&invocation.base_directory);
    invocation
}

fn bfloat16_tensor(checkpoint: &mut SourceCheckpoint, tensor_name: &str) -> Vec<f32> {
    let tensor_record = checkpoint
        .tensor_records
        .iter()
        .find(|tensor_record| tensor_record.tensor_name == tensor_name)
        .cloned()
        .expect("the tensor should be present");
    let tensor_bytes = checkpoint
        .read_tensor_bytes(&tensor_record)
        .expect("the tensor should read");
    let (whole_values, _) = tensor_bytes.as_chunks::<2>();
    whole_values
        .iter()
        .map(|value_bytes| bfloat16_bits_to_f32(u16::from_le_bytes(*value_bytes)))
        .collect()
}

#[test]
fn should_fold_the_adapter_into_the_base_and_keep_every_other_tensor() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let invocation = invocation_in(&temporary_directory, "base");

    convert_checkpoint(&invocation).expect("the synthetic checkpoint should convert");

    let mut converted =
        SourceCheckpoint::open(&invocation.output_directory.join("model.safetensors"))
            .expect("the merged backbone should read back");
    // W + 2 · B · A with B = [0.5, 0.25]ᵀ and A = [1, 0, -1].
    assert_eq!(
        bfloat16_tensor(&mut converted, ADAPTED_TENSOR_NAME),
        vec![2.0, 2.0, 2.0, 4.5, 5.0, 5.5]
    );
    assert_eq!(
        bfloat16_tensor(&mut converted, UNTOUCHED_TENSOR_NAME),
        vec![0.5, 0.25, 0.125]
    );
}

#[test]
fn should_write_the_head_the_settings_and_the_side_files_without_local_paths() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let invocation = invocation_in(&temporary_directory, "base");

    convert_checkpoint(&invocation).expect("the synthetic checkpoint should convert");

    let head = SourceCheckpoint::open(&invocation.output_directory.join(KEV_HEAD_FILE_NAME))
        .expect("the head should read back");
    let head_names: Vec<&str> = head
        .tensor_records
        .iter()
        .map(|record| record.tensor_name.as_str())
        .collect();
    assert_eq!(head_names, ["k.bias", "k.weight", "q.bias", "q.weight"]);
    let settings: serde_json::Value = serde_json::from_slice(
        &std::fs::read(invocation.output_directory.join(KEV_SETTINGS_FILE_NAME))
            .expect("the settings should exist"),
    )
    .expect("the settings should be JSON");
    assert_eq!(settings["model_type"], "kev");
    assert_eq!(settings["temperature"], 2.0);
    assert_eq!(settings["pointer_dimension"], 2);
    assert_eq!(settings["base_revision"], BASE_REVISION);
    let tokenizer_settings =
        std::fs::read_to_string(invocation.output_directory.join("tokenizer_config.json"))
            .expect("the tokenizer settings should be copied");
    assert!(tokenizer_settings.contains("<|endoftext|>"));
    assert!(!tokenizer_settings.contains("fictional-home"));
    assert!(invocation.output_directory.join("config.json").is_file());
    assert!(invocation.output_directory.join("tokenizer.json").is_file());
}

#[test]
fn should_refuse_a_base_snapshot_of_another_revision_before_writing_anything() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let invocation = invocation_in(
        &temporary_directory,
        "fedcba9876543210fedcba9876543210fedcba98",
    );

    let conversion_outcome = convert_checkpoint(&invocation);

    assert!(matches!(conversion_outcome, Err(reason) if reason.contains("revision")));
    assert!(!invocation.output_directory.exists());
}

#[test]
fn should_accept_the_base_snapshot_of_the_adapters_revision() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let invocation = invocation_in(&temporary_directory, BASE_REVISION);

    assert!(convert_checkpoint(&invocation).is_ok());
}

#[test]
fn should_refuse_to_write_into_a_folder_that_already_holds_files() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let invocation = invocation_in(&temporary_directory, "base");
    std::fs::create_dir_all(&invocation.output_directory)
        .expect("the output folder should be creatable");
    std::fs::write(invocation.output_directory.join("config.json"), "{}")
        .expect("the existing file should be writable");

    assert!(
        matches!(convert_checkpoint(&invocation), Err(reason) if reason.contains("already holds files"))
    );
}

#[test]
fn should_parse_the_three_folders_and_refuse_anything_else() {
    let arguments: Vec<String> = ["--output", "c", "--base", "b", "--adapter", "a"]
        .iter()
        .map(|argument| (*argument).to_owned())
        .collect();
    let invocation =
        ConvertKevInvocation::parse(&arguments).expect("the three folders should parse");

    assert_eq!(invocation.adapter_directory, Path::new("a"));
    assert_eq!(invocation.base_directory, Path::new("b"));
    assert_eq!(invocation.output_directory, Path::new("c"));
    for invalid_arguments in [
        vec!["--adapter", "a"],
        vec!["--adapter"],
        vec!["--merge"],
        vec!["--adapter", "a", "--adapter", "b"],
    ] {
        let invalid_arguments: Vec<String> = invalid_arguments
            .iter()
            .map(|argument| (*argument).to_owned())
            .collect();
        assert!(ConvertKevInvocation::parse(&invalid_arguments).is_err());
    }
}
