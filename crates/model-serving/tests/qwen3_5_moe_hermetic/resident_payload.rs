//! Hermetic arithmetic contracts for complete sparse-expert payload admission.
//!
//! These tests use fictional paths and geometry because admission must be exact
//! before files are opened or MLX is initialized. Real-artifact acceptance
//! separately prove that the computed inventory can be materialized.

use std::path::PathBuf;

use astronomical_model_serving::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource,
    SafetensorsDtype, maximum_resident_gate_up_fusion_transient_payload_bytes,
};

#[test]
fn should_count_every_complete_expert_tensor_in_a_resident_layer() {
    let resident_layer_plan =
        synthetic_layer_plan("language_model.model.layers.0.mlp", 4, &[128, 64, 32]);

    assert_eq!(
        resident_layer_plan
            .complete_expert_payload_byte_count()
            .expect("the complete resident payload should fit u64"),
        (128 + 64 + 32) * 4,
    );
}

#[test]
fn should_preserve_mixed_tensor_payload_widths_in_the_resident_inventory() {
    let resident_layer_plan = synthetic_layer_plan(
        "language_model.model.layers.1.mlp",
        256,
        &[1_024, 512, 256, 128, 64, 32, 16, 8, 4],
    );

    assert_eq!(
        resident_layer_plan
            .complete_expert_payload_byte_count()
            .expect("mixed affine tensor widths should have an exact payload"),
        523_264,
    );
}

#[test]
fn should_reject_a_complete_expert_payload_that_exceeds_u64() {
    let overflowing_layer_plan =
        synthetic_layer_plan("language_model.mtp.layers.0.mlp", 2, &[usize::MAX]);

    assert!(matches!(
        overflowing_layer_plan.complete_expert_payload_byte_count(),
        Err(ExpertManifestError::CompleteExpertPayloadByteCountOverflow { .. })
    ));
}

#[test]
fn should_reserve_one_compatible_native_gate_up_layer_transient() {
    let resident_layer_plan = synthetic_gate_up_fusion_layer_plan(
        QuantizationMode::NativeBfloat16,
        vec![4, 16, 32],
        vec![4, 16, 32],
        0,
        0,
        128,
    );

    assert_eq!(
        maximum_resident_gate_up_fusion_transient_payload_bytes(&[resident_layer_plan])
            .expect("the native fusion transient should fit u64"),
        1_024,
    );
}

#[test]
fn should_reserve_every_affine_parameter_in_one_layer_fusion_transient() {
    let resident_layer_plan = synthetic_gate_up_fusion_layer_plan(
        QuantizationMode::Affine,
        vec![4, 16, 4],
        vec![4, 16, 4],
        4,
        4,
        128,
    );

    assert_eq!(
        maximum_resident_gate_up_fusion_transient_payload_bytes(&[resident_layer_plan])
            .expect("the affine fusion transient should fit u64"),
        2_048,
    );
}

#[test]
fn should_keep_incompatible_gate_up_shapes_separate_without_fusion_headroom() {
    let resident_layer_plan = synthetic_gate_up_fusion_layer_plan(
        QuantizationMode::NativeBfloat16,
        vec![4, 16, 32],
        vec![4, 24, 32],
        0,
        0,
        128,
    );

    assert_eq!(
        maximum_resident_gate_up_fusion_transient_payload_bytes(&[resident_layer_plan])
            .expect("shape incompatibility should remain a valid separate plan"),
        0,
    );
}

#[test]
fn should_keep_incompatible_quantization_bit_widths_separate_without_fusion_headroom() {
    let resident_layer_plan = synthetic_gate_up_fusion_layer_plan(
        QuantizationMode::Affine,
        vec![4, 16, 4],
        vec![4, 16, 4],
        4,
        8,
        128,
    );

    assert_eq!(
        maximum_resident_gate_up_fusion_transient_payload_bytes(&[resident_layer_plan])
            .expect("quantization incompatibility should remain a valid separate plan"),
        0,
    );
}

#[test]
fn should_keep_mixed_native_and_affine_gate_up_separate_without_fusion_headroom() {
    let mut resident_layer_plan = synthetic_gate_up_fusion_layer_plan(
        QuantizationMode::Affine,
        vec![4, 16, 4],
        vec![4, 16, 4],
        4,
        4,
        128,
    );
    resident_layer_plan.tensor_sources.retain(|tensor_source| {
        tensor_source.projection_name != "gate_proj" || tensor_source.parameter_name == "weight"
    });

    assert_eq!(
        maximum_resident_gate_up_fusion_transient_payload_bytes(&[resident_layer_plan])
            .expect("mixed native and affine gate/up should remain a valid separate plan"),
        0,
    );
}

#[test]
fn should_reject_a_gate_up_fusion_transient_that_exceeds_u64() {
    let resident_layer_plan = synthetic_gate_up_fusion_layer_plan(
        QuantizationMode::NativeBfloat16,
        vec![2, 16, 32],
        vec![2, 16, 32],
        0,
        0,
        usize::MAX,
    );

    assert!(
        maximum_resident_gate_up_fusion_transient_payload_bytes(&[resident_layer_plan]).is_err(),
        "fusion admission must reject an unrepresentable temporary payload"
    );
}

fn synthetic_layer_plan(
    layer_prefix: &str,
    expert_capacity: usize,
    tensor_bytes_per_expert: &[usize],
) -> QuantizedExpertLayerPlan {
    let tensor_sources = tensor_bytes_per_expert
        .iter()
        .enumerate()
        .map(|(tensor_index, bytes_per_expert)| QuantizedTensorSource {
            tensor_name: format!("{layer_prefix}.switch_mlp.tensor_{tensor_index}"),
            projection_name: "gate_proj".to_owned(),
            parameter_name: "weight".to_owned(),
            quantization_bits: 4,
            quantization_group_size: 64,
            source_file: PathBuf::from("fictional-model-shard.safetensors"),
            source_file_size_bytes: u64::MAX,
            dtype: SafetensorsDtype::Uint32,
            full_shape: vec![expert_capacity, 1, 1],
            tensor_payload_offset: 0,
            bytes_per_expert: *bytes_per_expert,
            expert_capacity,
        })
        .collect();
    QuantizedExpertLayerPlan {
        layer_prefix: layer_prefix.to_owned(),
        tensor_sources,
        expert_capacity,
        quantization_bits: 4,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    }
}

fn synthetic_gate_up_fusion_layer_plan(
    quantization_mode: QuantizationMode,
    gate_weight_shape: Vec<usize>,
    up_weight_shape: Vec<usize>,
    gate_quantization_bits: i32,
    up_quantization_bits: i32,
    weight_bytes_per_expert: usize,
) -> QuantizedExpertLayerPlan {
    let expert_capacity = gate_weight_shape[0];
    // Fusion planning uses only validated tensor metadata, so fictional source
    // intervals keep this acceptance fixture independent of files and MLX.
    let mut tensor_sources = Vec::new();
    let mut append_projection =
        |projection_name: &str, weight_shape: Vec<usize>, quantization_bits: i32| {
            tensor_sources.push(QuantizedTensorSource {
                tensor_name: format!("language_model.model.layers.0.mlp.{projection_name}.weight"),
                projection_name: projection_name.to_owned(),
                parameter_name: "weight".to_owned(),
                quantization_bits,
                quantization_group_size: 64,
                source_file: PathBuf::from("fictional-model-shard.safetensors"),
                source_file_size_bytes: u64::MAX,
                dtype: match quantization_mode {
                    QuantizationMode::NativeBfloat16 => SafetensorsDtype::BFloat16,
                    QuantizationMode::Affine => SafetensorsDtype::Uint32,
                },
                full_shape: weight_shape,
                tensor_payload_offset: 0,
                bytes_per_expert: weight_bytes_per_expert,
                expert_capacity,
            });
            if quantization_mode == QuantizationMode::Affine {
                for parameter_name in ["scales", "biases"] {
                    tensor_sources.push(QuantizedTensorSource {
                        tensor_name: format!(
                            "language_model.model.layers.0.mlp.{projection_name}.{parameter_name}"
                        ),
                        projection_name: projection_name.to_owned(),
                        parameter_name: parameter_name.to_owned(),
                        quantization_bits,
                        quantization_group_size: 64,
                        source_file: PathBuf::from("fictional-model-shard.safetensors"),
                        source_file_size_bytes: u64::MAX,
                        dtype: SafetensorsDtype::Float16,
                        full_shape: vec![expert_capacity, 16, 1],
                        tensor_payload_offset: 0,
                        bytes_per_expert: weight_bytes_per_expert / 2,
                        expert_capacity,
                    });
                }
            }
        };
    append_projection("gate_proj", gate_weight_shape, gate_quantization_bits);
    append_projection("up_proj", up_weight_shape, up_quantization_bits);

    QuantizedExpertLayerPlan {
        layer_prefix: "language_model.model.layers.0.mlp".to_owned(),
        tensor_sources,
        expert_capacity,
        quantization_bits: gate_quantization_bits,
        quantization_group_size: 64,
        quantization_mode,
    }
}
