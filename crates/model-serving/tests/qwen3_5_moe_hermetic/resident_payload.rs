//! Hermetic arithmetic contracts for complete sparse-expert payload admission.
//!
//! These tests use fictional paths and geometry because admission must be exact
//! before files are opened or MLX is initialized. Real-artifact qualifications
//! separately prove that the computed inventory can be materialized.

use std::path::PathBuf;

use astronomical_model_serving::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource,
    SafetensorsDtype,
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
