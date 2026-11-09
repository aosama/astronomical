use super::{expert_layer_file_name, plan_layer_records};
use crate::atlas_manifest::ATLAS_RECORD_ALIGNMENT_BYTES;
use crate::source_checkpoint::{SourceExpertLayer, SourceTensor};

fn expert_tensor(parameter_path: &str, payload_byte_count: u64) -> SourceTensor {
    SourceTensor {
        tensor_name: format!("language_model.model.layers.3.mlp.switch_mlp.{parameter_path}"),
        shard_file_name: "model-00001.safetensors".to_owned(),
        dtype: "U32".to_owned(),
        shape: vec![4, 16],
        payload_file_offset: 0,
        payload_byte_count,
    }
}

#[test]
fn should_order_slots_gate_up_down_then_weight_scales_biases() {
    let source_layer = SourceExpertLayer {
        layer_index: 3,
        expert_tensors: vec![
            expert_tensor("down_proj.biases", 400),
            expert_tensor("gate_proj.scales", 400),
            expert_tensor("up_proj.weight", 4_000),
            expert_tensor("gate_proj.weight", 4_000),
        ],
    };
    let planned_layer = plan_layer_records(&source_layer, 4);
    let slot_suffixes: Vec<&str> = planned_layer
        .slots
        .iter()
        .map(|slot| {
            slot.tensor_name
                .rsplit("switch_mlp.")
                .next()
                .unwrap_or_default()
        })
        .collect();
    assert_eq!(
        slot_suffixes,
        [
            "gate_proj.weight",
            "gate_proj.scales",
            "up_proj.weight",
            "down_proj.biases"
        ]
    );
}

#[test]
fn should_pack_slots_back_to_back_and_round_the_stride_up_to_a_page() {
    let source_layer = SourceExpertLayer {
        layer_index: 3,
        expert_tensors: vec![
            expert_tensor("gate_proj.weight", 4_000),
            expert_tensor("gate_proj.scales", 400),
        ],
    };
    let planned_layer = plan_layer_records(&source_layer, 4);
    assert_eq!(planned_layer.slots[1].offset_in_record, 1_000);
    assert_eq!(planned_layer.record_byte_count, 1_100);
    assert_eq!(
        planned_layer.record_stride_bytes,
        ATLAS_RECORD_ALIGNMENT_BYTES
    );
    assert_eq!(planned_layer.file_name, expert_layer_file_name(3));
}

#[test]
fn should_keep_a_record_that_is_already_page_sized_at_its_own_size() {
    let page_bytes_per_expert = ATLAS_RECORD_ALIGNMENT_BYTES;
    let source_layer = SourceExpertLayer {
        layer_index: 0,
        expert_tensors: vec![expert_tensor("gate_proj.weight", 4 * page_bytes_per_expert)],
    };
    let planned_layer = plan_layer_records(&source_layer, 4);
    assert_eq!(planned_layer.record_stride_bytes, page_bytes_per_expert);
    assert_eq!(planned_layer.file_byte_count(), 4 * page_bytes_per_expert);
}

#[test]
fn should_place_unknown_parameters_after_the_known_ones() {
    let source_layer = SourceExpertLayer {
        layer_index: 0,
        expert_tensors: vec![
            expert_tensor("gate_proj.codebook", 40),
            expert_tensor("gate_proj.weight", 4_000),
        ],
    };
    let planned_layer = plan_layer_records(&source_layer, 4);
    assert!(
        planned_layer.slots[0]
            .tensor_name
            .ends_with("gate_proj.weight")
    );
    assert!(
        planned_layer.slots[1]
            .tensor_name
            .ends_with("gate_proj.codebook")
    );
}
