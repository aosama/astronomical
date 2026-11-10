//! Component contracts for `cargo xtask convert-laya`: a small synthetic
//! PyTorch-layout checkpoint is converted end to end and the output folder
//! is read back.

use std::path::Path;

use serde_json::json;

use super::half_float::{f32_to_half_bits, half_bits_to_f32};
use super::safetensors_file::{
    OutputCheckpoint, PlannedTensor, SourceCheckpoint, TensorElementType,
};
use super::{ConvertLayaInvocation, convert_checkpoint};

const HIDDEN_SIZE: u64 = 32;

fn arguments(argument_texts: &[&str]) -> Vec<String> {
    argument_texts
        .iter()
        .map(|argument| (*argument).to_owned())
        .collect()
}

/// Half-precision bytes of `element_count` values that differ from each
/// other, so a shifted or reordered tensor cannot pass unnoticed.
fn distinct_half_bytes(element_count: u64, first_value: f32) -> Vec<u8> {
    let mut half_bytes = Vec::with_capacity(element_count as usize * 2);
    for element_index in 0..element_count {
        let value = first_value + element_index as f32 / 64.0;
        half_bytes.extend_from_slice(&f32_to_half_bits(value).to_le_bytes());
    }
    half_bytes
}

/// Writes a checkpoint folder in the PyTorch Laya layout with one tensor of
/// every kind the converter treats differently.
fn write_synthetic_checkpoint(checkpoint_directory: &Path) {
    let source_tensors = [
        ("encoder.layers.0.attn.Wo.weight", vec![2, HIDDEN_SIZE]),
        ("encoder.layers.0.mlp_norm.weight", vec![HIDDEN_SIZE]),
        ("head.layers.0.self_attn.in_proj_bias", vec![6]),
        (
            "head.layers.0.self_attn.in_proj_weight",
            vec![3, HIDDEN_SIZE],
        ),
        ("scorer.3.weight", vec![1, HIDDEN_SIZE]),
        ("temperature", vec![3]),
    ];
    let mut planned_tensors = Vec::new();
    for (tensor_name, shape) in &source_tensors {
        planned_tensors.push(PlannedTensor {
            tensor_name: (*tensor_name).to_owned(),
            element_type: TensorElementType::Float16,
            shape: shape.clone(),
        });
    }
    std::fs::create_dir_all(checkpoint_directory.join("encoder"))
        .expect("the encoder folder should be creatable");
    std::fs::create_dir_all(checkpoint_directory.join("tokenizer"))
        .expect("the tokenizer folder should be creatable");
    let weights_file = std::fs::File::create(checkpoint_directory.join("model.safetensors"))
        .expect("the weights file should be creatable");
    let mut source_writer =
        OutputCheckpoint::begin(weights_file, planned_tensors).expect("the header should write");
    for (tensor_position, (tensor_name, shape)) in source_tensors.iter().enumerate() {
        let element_count = shape.iter().product();
        source_writer
            .append_tensor(
                tensor_name,
                &distinct_half_bytes(element_count, tensor_position as f32 - 2.0),
            )
            .expect("each synthetic tensor should match its plan");
    }
    source_writer.finish().expect("the weights should finish");
    std::fs::write(
        checkpoint_directory.join("rl_agent_config.json"),
        json!({
            "head_layers": 1,
            "max_len": 512,
            "head_max_len": 192,
            "temperature": [2.5, 1.0, 1.0],
            "model_name": "synthetic-agent",
            "training": {"eval_data": "/fictional-home/fictional-person/eval.jsonl"}
        })
        .to_string(),
    )
    .expect("the checkpoint settings should be writable");
    std::fs::write(
        checkpoint_directory.join("encoder/config.json"),
        json!({"model_type": "modernbert", "hidden_size": HIDDEN_SIZE}).to_string(),
    )
    .expect("the encoder config should be writable");
    std::fs::write(
        checkpoint_directory.join("tokenizer/tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"BPE"}}"#,
    )
    .expect("the tokenizer should be writable");
    std::fs::write(
        checkpoint_directory.join("tokenizer/tokenizer_config.json"),
        json!({"cls_token": "[CLS]", "cache_dir": "/fictional-home/fictional-person/cache"})
            .to_string(),
    )
    .expect("the tokenizer settings should be writable");
}

fn invocation_in(temporary_directory: &tempfile::TempDir) -> ConvertLayaInvocation {
    ConvertLayaInvocation {
        checkpoint_directory: temporary_directory.path().join("checkpoint"),
        output_directory: temporary_directory.path().join("converted"),
    }
}

#[test]
fn should_convert_a_checkpoint_into_the_mlx_layout() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let invocation = invocation_in(&temporary_directory);
    write_synthetic_checkpoint(&invocation.checkpoint_directory);

    convert_checkpoint(&invocation).expect("the synthetic checkpoint should convert");

    let converted_checkpoint =
        SourceCheckpoint::open(&invocation.output_directory.join("model.safetensors"))
            .expect("the converted weights should read back");
    let converted_names: Vec<&str> = converted_checkpoint
        .tensor_records
        .iter()
        .map(|tensor_record| tensor_record.tensor_name.as_str())
        .collect();
    assert_eq!(
        converted_names,
        [
            "encoder.layers.0.attn.Wo.biases",
            "encoder.layers.0.attn.Wo.scales",
            "encoder.layers.0.attn.Wo.weight",
            "encoder.layers.0.mlp_norm.weight",
            "head.layers.0.self_attn.in_proj.bias",
            "head.layers.0.self_attn.in_proj.biases",
            "head.layers.0.self_attn.in_proj.scales",
            "head.layers.0.self_attn.in_proj.weight",
            "scorer.layers.3.weight",
        ],
        "projections are quantized, names follow MLX, and the temperature tensor is gone"
    );
    let packed_weight = &converted_checkpoint.tensor_records[2];
    assert_eq!(packed_weight.element_type, TensorElementType::Unsigned32);
    assert_eq!(packed_weight.shape, vec![2, HIDDEN_SIZE / 4]);
    assert!(
        !invocation
            .output_directory
            .join("model.safetensors.unfinished")
            .exists()
    );
}

#[test]
fn should_keep_unquantized_tensors_bit_for_bit_and_quantized_ones_within_half_a_step() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let invocation = invocation_in(&temporary_directory);
    write_synthetic_checkpoint(&invocation.checkpoint_directory);
    convert_checkpoint(&invocation).expect("the synthetic checkpoint should convert");
    let mut source_checkpoint =
        SourceCheckpoint::open(&invocation.checkpoint_directory.join("model.safetensors"))
            .expect("the source weights should read");
    let mut converted_checkpoint =
        SourceCheckpoint::open(&invocation.output_directory.join("model.safetensors"))
            .expect("the converted weights should read back");
    let source_records = source_checkpoint.tensor_records.clone();
    let converted_records = converted_checkpoint.tensor_records.clone();

    // Source name order: Wo.weight, mlp_norm, in_proj_bias, in_proj_weight, scorer, temperature.
    let source_norm_bytes = source_checkpoint
        .read_tensor_bytes(&source_records[1])
        .expect("the source norm should read");
    let converted_norm_bytes = converted_checkpoint
        .read_tensor_bytes(&converted_records[3])
        .expect("the converted norm should read");
    assert_eq!(source_norm_bytes, converted_norm_bytes);

    let source_projection_bytes = source_checkpoint
        .read_tensor_bytes(&source_records[0])
        .expect("the source projection should read");
    let bias_bytes = converted_checkpoint
        .read_tensor_bytes(&converted_records[0])
        .expect("the biases should read");
    let scale_bytes = converted_checkpoint
        .read_tensor_bytes(&converted_records[1])
        .expect("the scales should read");
    let level_bytes = converted_checkpoint
        .read_tensor_bytes(&converted_records[2])
        .expect("the packed weight should read");
    let (source_halves, _) = source_projection_bytes.as_chunks::<2>();
    for (weight_index, source_half_bytes) in source_halves.iter().enumerate() {
        let group_index = weight_index / 32;
        let scale = half_bits_to_f32(u16::from_le_bytes([
            scale_bytes[group_index * 2],
            scale_bytes[group_index * 2 + 1],
        ]));
        let bias = half_bits_to_f32(u16::from_le_bytes([
            bias_bytes[group_index * 2],
            bias_bytes[group_index * 2 + 1],
        ]));
        let source_weight = half_bits_to_f32(u16::from_le_bytes([
            source_half_bytes[0],
            source_half_bytes[1],
        ]));
        let recovered_weight = f32::from(level_bytes[weight_index]) * scale + bias;
        assert!(
            (source_weight - recovered_weight).abs() <= scale.abs() * 0.5 + f32::EPSILON,
            "weight {weight_index}: {source_weight} was recovered as {recovered_weight}"
        );
    }
}

#[test]
fn should_write_a_config_and_tokenizer_settings_free_of_local_paths() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let invocation = invocation_in(&temporary_directory);
    write_synthetic_checkpoint(&invocation.checkpoint_directory);

    convert_checkpoint(&invocation).expect("the synthetic checkpoint should convert");

    let config_text = std::fs::read_to_string(invocation.output_directory.join("config.json"))
        .expect("the config should be written");
    let tokenizer_settings_text =
        std::fs::read_to_string(invocation.output_directory.join("tokenizer_config.json"))
            .expect("the tokenizer settings should be written");
    let model_config: serde_json::Value =
        serde_json::from_str(&config_text).expect("the config should be JSON");
    assert_eq!(model_config["model_type"], "laya");
    assert_eq!(model_config["temperature"], json!([2.5, 1.0, 1.0]));
    assert_eq!(model_config["quantization"]["bits"], 8);
    assert!(!config_text.contains("fictional-home"));
    assert!(!tokenizer_settings_text.contains("fictional-home"));
    assert!(tokenizer_settings_text.contains("[CLS]"));
    assert!(invocation.output_directory.join("tokenizer.json").is_file());
}

#[test]
fn should_refuse_to_write_into_a_folder_that_already_holds_files() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let invocation = invocation_in(&temporary_directory);
    write_synthetic_checkpoint(&invocation.checkpoint_directory);
    std::fs::create_dir_all(&invocation.output_directory)
        .expect("the output folder should be creatable");
    std::fs::write(invocation.output_directory.join("config.json"), "{}")
        .expect("the existing file should be writable");

    let conversion_outcome = convert_checkpoint(&invocation);

    assert!(matches!(conversion_outcome, Err(reason) if reason.contains("already holds files")));
    assert_eq!(
        std::fs::read_to_string(invocation.output_directory.join("config.json"))
            .expect("the existing file should still be there"),
        "{}",
        "an existing model must be left untouched"
    );
}

#[test]
fn should_refuse_a_folder_that_is_not_a_laya_checkpoint_before_writing_anything() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let invocation = invocation_in(&temporary_directory);
    std::fs::create_dir_all(&invocation.checkpoint_directory)
        .expect("the empty checkpoint folder should be creatable");

    let conversion_outcome = convert_checkpoint(&invocation);

    assert!(matches!(conversion_outcome, Err(reason) if reason.contains("rl_agent_config.json")));
    assert!(!invocation.output_directory.exists());
}

#[test]
fn should_parse_both_folders_in_either_order() {
    let invocation = ConvertLayaInvocation::parse(&arguments(&[
        "--output",
        "converted",
        "--checkpoint",
        "checkpoint",
    ]))
    .expect("both folders in any order should parse");

    assert_eq!(invocation.checkpoint_directory, Path::new("checkpoint"));
    assert_eq!(invocation.output_directory, Path::new("converted"));
}

#[test]
fn should_refuse_missing_repeated_and_unknown_arguments() {
    assert!(ConvertLayaInvocation::parse(&arguments(&["--checkpoint", "a"])).is_err());
    assert!(ConvertLayaInvocation::parse(&arguments(&["--checkpoint"])).is_err());
    assert!(
        ConvertLayaInvocation::parse(&arguments(&[
            "--checkpoint",
            "a",
            "--checkpoint",
            "b",
            "--output",
            "c"
        ]))
        .is_err()
    );
    assert!(ConvertLayaInvocation::parse(&arguments(&["--force"])).is_err());
    assert!(ConvertLayaInvocation::parse(&[]).is_err());
}
