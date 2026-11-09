use std::collections::{BTreeMap, BTreeSet};

use super::{
    ShardIndexDocument, SourceExpertLayer, SourceTensor, VISION_BUNDLE_FILE_NAME,
    check_expert_layer, payload_span, routed_expert_layer_index, vision_sidecar_file_names,
};

fn shard_index_of(tensor_files: &[(&str, &str)]) -> ShardIndexDocument {
    ShardIndexDocument {
        weight_map: tensor_files
            .iter()
            .map(|(tensor_name, shard_file_name)| {
                ((*tensor_name).to_owned(), (*shard_file_name).to_owned())
            })
            .collect::<BTreeMap<_, _>>(),
    }
}

#[test]
fn should_read_the_layer_of_a_routed_expert_tensor() {
    assert_eq!(
        routed_expert_layer_index("language_model.model.layers.12.mlp.switch_mlp.gate_proj.weight"),
        Some(12)
    );
    assert_eq!(
        routed_expert_layer_index("model.layers.0.mlp.switch_mlp.down_proj.biases"),
        Some(0)
    );
}

#[test]
fn should_treat_shared_experts_and_routers_as_resident() {
    for resident_tensor_name in [
        "language_model.model.layers.3.mlp.shared_expert.gate_proj.weight",
        "language_model.model.layers.3.mlp.gate.weight",
        "language_model.model.embed_tokens.weight",
    ] {
        assert_eq!(
            routed_expert_layer_index(resident_tensor_name),
            None,
            "{resident_tensor_name}"
        );
    }
}

#[test]
fn should_refuse_a_routed_expert_name_without_a_layer_number() {
    assert_eq!(
        routed_expert_layer_index("language_model.model.layers.x.mlp.switch_mlp.gate_proj.weight"),
        None
    );
}

#[test]
fn should_carry_the_vision_bundle_when_it_holds_the_whole_tower() {
    let shard_index = shard_index_of(&[
        (
            "language_model.model.embed_tokens.weight",
            "model-00001.safetensors",
        ),
        (
            "vision_tower.blocks.0.attn.proj.bias",
            VISION_BUNDLE_FILE_NAME,
        ),
        ("model.visual.merger.norm.bias", VISION_BUNDLE_FILE_NAME),
    ]);

    assert_eq!(
        vision_sidecar_file_names(&shard_index, true),
        Ok(BTreeSet::from([VISION_BUNDLE_FILE_NAME.to_owned()]))
    );
}

#[test]
fn should_refuse_a_vision_tower_inside_a_language_shard() {
    let shard_index = shard_index_of(&[
        (
            "language_model.model.embed_tokens.weight",
            "model-00001.safetensors",
        ),
        (
            "vision_tower.blocks.0.attn.proj.bias",
            "model-00001.safetensors",
        ),
    ]);

    let refusal = vision_sidecar_file_names(&shard_index, true)
        .expect_err("serving cannot load a tower outside the vision bundle");
    assert!(
        refusal.contains("vision_tower.blocks.0.attn.proj.bias"),
        "{refusal}"
    );
}

#[test]
fn should_refuse_a_vision_tower_in_its_own_regular_shard() {
    let shard_index = shard_index_of(&[
        (
            "language_model.model.embed_tokens.weight",
            "model-00001.safetensors",
        ),
        (
            "vision_tower.blocks.0.attn.proj.bias",
            "model-00002.safetensors",
        ),
    ]);

    assert!(vision_sidecar_file_names(&shard_index, true).is_err());
}

#[test]
fn should_refuse_a_language_tensor_inside_the_vision_bundle() {
    let shard_index = shard_index_of(&[
        (
            "language_model.model.embed_tokens.weight",
            VISION_BUNDLE_FILE_NAME,
        ),
        (
            "vision_tower.blocks.0.attn.proj.bias",
            VISION_BUNDLE_FILE_NAME,
        ),
    ]);

    assert!(vision_sidecar_file_names(&shard_index, true).is_err());
}

#[test]
fn should_refuse_a_declared_vision_tower_with_no_vision_bundle() {
    let shard_index = shard_index_of(&[(
        "language_model.model.embed_tokens.weight",
        "model-00001.safetensors",
    )]);

    assert!(vision_sidecar_file_names(&shard_index, true).is_err());
}

#[test]
fn should_find_no_sidecar_in_a_text_only_checkpoint() {
    let shard_index = shard_index_of(&[(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
        "model-00001.safetensors",
    )]);

    assert_eq!(
        vision_sidecar_file_names(&shard_index, false),
        Ok(BTreeSet::new())
    );
}

#[test]
fn should_place_a_payload_after_the_header() {
    assert_eq!(payload_span(1_000, [16, 48]), Some((1_016, 32)));
}

#[test]
fn should_accept_an_empty_payload_span_at_the_boundary() {
    assert_eq!(payload_span(1_000, [16, 16]), Some((1_016, 0)));
}

#[test]
fn should_refuse_data_offsets_that_end_before_they_start() {
    assert_eq!(payload_span(1_000, [48, 16]), None);
}

#[test]
fn should_refuse_data_offsets_that_overflow_the_file_offset() {
    assert_eq!(payload_span(1_000, [u64::MAX - 10, u64::MAX]), None);
}

#[test]
fn should_refuse_an_empty_routed_expert_tensor() {
    let empty_expert_layer = SourceExpertLayer {
        layer_index: 0,
        expert_tensors: vec![SourceTensor {
            tensor_name: "model.layers.0.mlp.switch_mlp.gate_proj.weight".to_owned(),
            shard_file_name: "model-00001.safetensors".to_owned(),
            dtype: "U32".to_owned(),
            shape: vec![4, 0],
            payload_file_offset: 1_024,
            payload_byte_count: 0,
        }],
    };

    assert!(check_expert_layer(&empty_expert_layer, 4).is_err());
}
