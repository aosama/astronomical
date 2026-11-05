use super::*;

#[test]
fn should_recognize_towers_in_every_convention() {
    for tower_tensor in [
        "vision_tower.blocks.0.attn.proj.weight",
        "model.visual.pos_embed.weight",
        "visual.merger.norm.bias",
        "vision_model.encoder.layers.0.weight",
    ] {
        assert!(is_vision_tower_tensor(tower_tensor), "{tower_tensor}");
    }
}

#[test]
fn should_not_mistake_text_or_audio_tensors_for_a_tower() {
    for other_tensor in [
        "language_model.model.layers.0.mlp.up_proj.weight",
        "model.language_model.visual_gate.weight",
        "audio_tower.blocks.0.weight",
        "lm_head.weight",
        "",
    ] {
        assert!(!is_vision_tower_tensor(other_tensor), "{other_tensor}");
    }
}

#[test]
fn should_report_the_tower_prefix_as_written() {
    assert_eq!(tower_name_prefix("vision_tower.blocks.0.x"), "vision_tower");
    assert_eq!(tower_name_prefix("model.visual.blocks.0.x"), "model.visual");
}

#[test]
fn should_find_a_block_index_only_when_one_is_there() {
    assert_eq!(
        block_index("vision_tower.blocks.26.attn.proj.weight"),
        Some(26)
    );
    assert_eq!(block_index("vision_model.encoder.layers.3.weight"), Some(3));
    assert_eq!(block_index("vision_tower.merger.norm.bias"), None);
    assert_eq!(block_index("vision_tower.blocks"), None);
    assert_eq!(block_index("vision_tower.blocks.first.weight"), None);
}

#[test]
fn should_reduce_both_naming_conventions_to_one_name() {
    assert_eq!(
        canonical_tensor_name("model.language_model.layers.0.norm.weight"),
        "layers.0.norm.weight"
    );
    assert_eq!(
        canonical_tensor_name("language_model.model.layers.0.norm.weight"),
        "layers.0.norm.weight"
    );
    assert_eq!(
        canonical_tensor_name("language_model.lm_head.weight"),
        "lm_head.weight"
    );
    assert_eq!(canonical_tensor_name("lm_head.weight"), "lm_head.weight");
}

#[test]
fn should_identify_quantization_tensors() {
    assert!(is_quantization_companion("layer.scales"));
    assert!(is_quantization_companion("layer.biases"));
    assert!(!is_quantization_companion("layer.weight"));
    assert_eq!(
        scales_name_for_weight("layer.weight").as_deref(),
        Some("layer.scales")
    );
    assert_eq!(scales_name_for_weight("layer.bias"), None);
}
