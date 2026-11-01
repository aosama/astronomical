//! Library-size facts published with a fully resident expert payload.
//!
//! Fictional tensor geometry keeps the measurement independent of model files
//! and MLX. A failed layer measurement must publish zero library bytes so
//! Observatory hides the SSD remainder instead of inventing one.

use std::path::PathBuf;

use crate::qwen3_5_core::artifacts::{
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype,
};

use super::resident_expert_library_facts;

#[test]
fn should_sum_complete_layer_payloads_and_keep_the_first_capacity() {
    let first_layer = layer_plan_with_one_tensor("language_model.model.layers.0.mlp", 4, 4, 10);
    let second_layer = layer_plan_with_one_tensor("language_model.model.layers.1.mlp", 8, 8, 5);

    let (complete_expert_payload_byte_count, expert_capacity) =
        resident_expert_library_facts(&[first_layer, second_layer]);

    assert_eq!(complete_expert_payload_byte_count, 40 + 40);
    assert_eq!(expert_capacity, 4);
}

#[test]
fn should_publish_zero_library_bytes_when_a_later_layer_measurement_fails() {
    let measured_layer = layer_plan_with_one_tensor("language_model.model.layers.0.mlp", 4, 4, 10);
    let inconsistent_layer =
        layer_plan_with_one_tensor("language_model.model.layers.1.mlp", 4, 2, 8);

    let (complete_expert_payload_byte_count, expert_capacity) =
        resident_expert_library_facts(&[measured_layer, inconsistent_layer]);

    assert_eq!(complete_expert_payload_byte_count, 0);
    assert_eq!(expert_capacity, 4);
}

#[test]
fn should_return_zero_bytes_and_zero_capacity_for_an_empty_plan_list() {
    let (complete_expert_payload_byte_count, expert_capacity) = resident_expert_library_facts(&[]);

    assert_eq!(complete_expert_payload_byte_count, 0);
    assert_eq!(expert_capacity, 0);
}

#[test]
fn should_saturate_an_expert_capacity_that_does_not_fit_in_u32() {
    let oversized_capacity_layer = layer_plan_with_one_tensor(
        "language_model.model.layers.0.mlp",
        usize::MAX,
        usize::MAX,
        0,
    );

    let (complete_expert_payload_byte_count, expert_capacity) =
        resident_expert_library_facts(&[oversized_capacity_layer]);

    assert_eq!(complete_expert_payload_byte_count, 0);
    assert_eq!(expert_capacity, u32::MAX);
}

fn layer_plan_with_one_tensor(
    layer_prefix: &str,
    plan_expert_capacity: usize,
    source_expert_capacity: usize,
    bytes_per_expert: usize,
) -> QuantizedExpertLayerPlan {
    let tensor_source = QuantizedTensorSource {
        tensor_name: format!("{layer_prefix}.switch_mlp.gate_proj.weight"),
        projection_name: "gate_proj".to_owned(),
        parameter_name: "weight".to_owned(),
        quantization_bits: 4,
        quantization_group_size: 64,
        source_file: PathBuf::from("fictional-model-shard.safetensors"),
        source_file_size_bytes: 0,
        dtype: SafetensorsDtype::Uint32,
        full_shape: vec![source_expert_capacity, 1, 1],
        tensor_payload_offset: 0,
        bytes_per_expert,
        expert_capacity: source_expert_capacity,
    };
    QuantizedExpertLayerPlan {
        layer_prefix: layer_prefix.to_owned(),
        tensor_sources: vec![tensor_source],
        jangtq2_projection_rotations: Vec::new(),
        expert_capacity: plan_expert_capacity,
        quantization_bits: 4,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    }
}
