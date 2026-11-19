use std::collections::HashMap;
use std::path::PathBuf;

use astronomical_expert_atlas::{AtlasExpertLayer, AtlasRecordSlot};

use super::{apply_record_strides, synthesize_layer_header};
use crate::expert_paging::quantized_expert_manifest::{
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource,
};
use crate::expert_paging::safetensors_header::SafetensorsDtype;

fn atlas_layer(slots: Vec<AtlasRecordSlot>) -> AtlasExpertLayer {
    let record_byte_count = slots.iter().map(|slot| slot.bytes_per_expert).sum();
    AtlasExpertLayer {
        layer_index: 7,
        file_name: "experts/layer-007.atlas".to_owned(),
        expert_count: 4,
        record_byte_count,
        record_stride_bytes: 16_384,
        slots,
    }
}

fn slot(
    stored_name: &str,
    dtype: &str,
    offset_in_record: u64,
    bytes_per_expert: u64,
) -> AtlasRecordSlot {
    AtlasRecordSlot {
        tensor_name: stored_name.to_owned(),
        dtype: dtype.to_owned(),
        shape: vec![4, 8],
        offset_in_record,
        bytes_per_expert,
    }
}

#[test]
fn should_name_each_slot_with_the_canonical_tensor_name_the_planner_asks_for() {
    let layer = atlas_layer(vec![
        slot(
            "model.layers.7.mlp.switch_mlp.gate_proj.weight",
            "U32",
            0,
            1_000,
        ),
        slot(
            "model.layers.7.mlp.switch_mlp.gate_proj.scales",
            "BF16",
            1_000,
            64,
        ),
    ]);
    let synthesized_header = synthesize_layer_header(&layer).expect("the header synthesizes");
    assert_eq!(
        synthesized_header.tensor_entries[0].tensor_name,
        "language_model.model.layers.7.mlp.switch_mlp.gate_proj.weight"
    );
    assert_eq!(
        synthesized_header.tensor_entries[1].dtype,
        SafetensorsDtype::BFloat16
    );
    assert_eq!(
        synthesized_header.tensor_entries[1].data_start_offset,
        1_000
    );
    assert_eq!(
        synthesized_header.tensor_entries[1].data_end_offset,
        1_000 + 4 * 64
    );
    assert_eq!(synthesized_header.total_file_size_bytes, 4 * 16_384);
}

#[test]
fn should_refuse_a_slot_that_is_not_a_routed_expert_tensor() {
    let layer = atlas_layer(vec![slot(
        "model.layers.7.mlp.shared_expert.gate_proj.weight",
        "U32",
        0,
        1_000,
    )]);
    assert!(synthesize_layer_header(&layer).is_err());
}

#[test]
fn should_refuse_a_slot_with_an_unknown_dtype() {
    let layer = atlas_layer(vec![slot(
        "model.layers.7.mlp.switch_mlp.gate_proj.weight",
        "Q4",
        0,
        1_000,
    )]);
    assert!(synthesize_layer_header(&layer).is_err());
}

fn layer_plan_reading(source_files: &[&str]) -> QuantizedExpertLayerPlan {
    QuantizedExpertLayerPlan {
        layer_prefix: "language_model.model.layers.7.mlp".to_owned(),
        tensor_sources: source_files
            .iter()
            .map(|source_file| QuantizedTensorSource {
                tensor_name: format!("{source_file}.tensor"),
                projection_name: "gate_proj".to_owned(),
                parameter_name: "weight".to_owned(),
                quantization_bits: 4,
                quantization_group_size: 64,
                source_file: PathBuf::from(source_file),
                source_file_size_bytes: 1 << 20,
                dtype: SafetensorsDtype::Uint32,
                full_shape: vec![4, 8],
                tensor_payload_offset: 0,
                bytes_per_expert: 1_000,
                expert_stride_bytes: 1_000,
                expert_capacity: 4,
            })
            .collect(),
        jangtq2_projection_rotations: Vec::new(),
        expert_capacity: 4,
        quantization_bits: 4,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    }
}

#[test]
fn should_set_the_record_stride_only_on_tensors_read_from_atlas_layer_files() {
    let mut layer_plan = layer_plan_reading(&[
        "/model/experts/layer-007.atlas",
        "/model/resident.safetensors",
    ]);
    let record_stride_by_layer_file =
        HashMap::from([(PathBuf::from("/model/experts/layer-007.atlas"), 16_384_u64)]);
    apply_record_strides(&mut layer_plan, &record_stride_by_layer_file).expect("strides apply");
    assert_eq!(layer_plan.tensor_sources[0].expert_stride_bytes, 16_384);
    assert_eq!(layer_plan.tensor_sources[1].expert_stride_bytes, 1_000);
}

#[test]
fn should_leave_a_plan_with_no_atlas_files_unchanged() {
    let mut layer_plan = layer_plan_reading(&["/model/model-00001.safetensors"]);
    apply_record_strides(&mut layer_plan, &HashMap::new()).expect("strides apply");
    assert_eq!(layer_plan.tensor_sources[0].expert_stride_bytes, 1_000);
}
