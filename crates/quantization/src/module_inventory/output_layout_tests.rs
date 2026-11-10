use std::collections::BTreeMap;

use super::{OutputDestination, OutputLayoutRules, OutputValueTransform, is_multimodal_tensor};

fn hub_release_rules(config_text: &str) -> OutputLayoutRules {
    let config_value: serde_json::Value =
        serde_json::from_str(config_text).expect("the test config parses");
    let mut shape_by_tensor_name = BTreeMap::new();
    shape_by_tensor_name.insert(
        "model.language_model.layers.0.linear_attn.conv1d.weight".to_owned(),
        vec![6144, 1, 4],
    );
    OutputLayoutRules::from_config(&config_value, &shape_by_tensor_name)
}

const VISION_QWEN_CONFIG: &str = r#"{
    "model_type": "qwen3_5",
    "vision_config": {"depth": 12},
    "text_config": {"model_type": "qwen3_5_text", "tie_word_embeddings": true}
}"#;

#[test]
fn should_rename_a_hub_language_tensor_to_the_mlx_lm_spelling() {
    let rules = hub_release_rules(VISION_QWEN_CONFIG);

    let layout = rules.layout_for("model.language_model.layers.3.mlp.up_proj.weight", &[8, 64]);

    assert_eq!(
        layout.output_name,
        "language_model.model.layers.3.mlp.up_proj.weight"
    );
    assert_eq!(layout.destination, OutputDestination::ModelShards);
    assert_eq!(layout.value_transform, OutputValueTransform::Unchanged);
}

#[test]
fn should_move_the_depthwise_convolution_into_mlx_axis_order() {
    let rules = hub_release_rules(VISION_QWEN_CONFIG);

    let layout = rules.layout_for(
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        &[6144, 1, 4],
    );

    assert_eq!(
        layout.value_transform,
        OutputValueTransform::PermuteAxes(vec![0, 2, 1])
    );
}

#[test]
fn should_offset_a_norm_weight_only_when_the_convolution_is_in_pytorch_layout() {
    let hub_rules = hub_release_rules(VISION_QWEN_CONFIG);
    let config_value: serde_json::Value =
        serde_json::from_str(VISION_QWEN_CONFIG).expect("the test config parses");
    let mut converted_shapes = BTreeMap::new();
    converted_shapes.insert(
        "language_model.model.layers.0.linear_attn.conv1d.weight".to_owned(),
        vec![6144, 4, 1],
    );
    let converted_rules = OutputLayoutRules::from_config(&config_value, &converted_shapes);

    let hub_layout = hub_rules.layout_for(
        "model.language_model.layers.0.input_layernorm.weight",
        &[1024],
    );
    let converted_layout = converted_rules.layout_for(
        "language_model.model.layers.0.input_layernorm.weight",
        &[1024],
    );

    assert_eq!(hub_layout.value_transform, OutputValueTransform::AddOne);
    assert_eq!(
        converted_layout.value_transform,
        OutputValueTransform::Unchanged
    );
}

#[test]
fn should_leave_the_linear_attention_norm_without_an_offset() {
    let rules = hub_release_rules(VISION_QWEN_CONFIG);

    let layout = rules.layout_for(
        "model.language_model.layers.0.linear_attn.norm.weight",
        &[128],
    );

    assert_eq!(layout.value_transform, OutputValueTransform::Unchanged);
}

#[test]
fn should_send_a_declared_tower_to_the_sidecar_under_the_front_end_name() {
    let rules = hub_release_rules(VISION_QWEN_CONFIG);

    let layout = rules.layout_for("model.visual.patch_embed.proj.weight", &[768, 3, 2, 16, 16]);

    assert_eq!(layout.destination, OutputDestination::VisionSidecar);
    assert_eq!(layout.output_name, "vision_tower.patch_embed.proj.weight");
    assert_eq!(
        layout.value_transform,
        OutputValueTransform::PermuteAxes(vec![0, 2, 3, 4, 1])
    );
}

#[test]
fn should_leave_an_already_channel_last_patch_embedding_alone() {
    let rules = hub_release_rules(VISION_QWEN_CONFIG);

    let layout = rules.layout_for("vision_tower.patch_embed.proj.weight", &[768, 2, 16, 16, 3]);

    assert_eq!(layout.value_transform, OutputValueTransform::Unchanged);
}

#[test]
fn should_drop_a_tower_the_qwen_config_does_not_declare() {
    let rules = hub_release_rules(r#"{"model_type": "qwen3_5"}"#);

    let layout = rules.layout_for("model.visual.blocks.0.attn.qkv.weight", &[2304, 768]);

    assert_eq!(layout.destination, OutputDestination::Dropped);
}

#[test]
fn should_send_the_multi_token_prediction_head_to_its_sidecar_under_its_own_names() {
    let rules = hub_release_rules(VISION_QWEN_CONFIG);

    let prediction_head = rules.layout_for("mtp.layers.0.self_attn.q_proj.weight", &[64, 64]);

    assert_eq!(prediction_head.destination, OutputDestination::MtpSidecar);
    assert_eq!(
        prediction_head.output_name,
        "mtp.layers.0.self_attn.q_proj.weight"
    );
    assert!(prediction_head.keeps_stored_dtype);
}

#[test]
fn should_drop_a_tied_output_head() {
    let rules = hub_release_rules(VISION_QWEN_CONFIG);

    let tied_head = rules.layout_for("lm_head.weight", &[248_320, 1024]);

    assert_eq!(tied_head.destination, OutputDestination::Dropped);
}

#[test]
fn should_keep_an_untied_output_head_under_the_language_model() {
    let rules = hub_release_rules(
        r#"{"model_type": "qwen3_5", "text_config": {"tie_word_embeddings": false}}"#,
    );

    let layout = rules.layout_for("lm_head.weight", &[248_320, 1024]);

    assert_eq!(layout.destination, OutputDestination::ModelShards);
    assert_eq!(layout.output_name, "language_model.lm_head.weight");
}

#[test]
fn should_keep_the_stored_width_of_the_gated_delta_decay_parameter() {
    let rules = hub_release_rules(VISION_QWEN_CONFIG);

    let decay = rules.layout_for("model.language_model.layers.0.linear_attn.A_log", &[16]);
    let step_bias = rules.layout_for("model.language_model.layers.0.linear_attn.dt_bias", &[16]);

    assert!(decay.keeps_stored_dtype);
    assert!(!step_bias.keeps_stored_dtype);
}

#[test]
fn should_change_nothing_for_a_family_without_layout_rules() {
    let rules = hub_release_rules(r#"{"model_type": "gemma4"}"#);

    let layout = rules.layout_for("model.layers.0.self_attn.q_proj.weight", &[64, 64]);

    assert_eq!(layout.output_name, "model.layers.0.self_attn.q_proj.weight");
    assert_eq!(layout.destination, OutputDestination::ModelShards);
    assert_eq!(
        rules.output_module_name("model.layers.0.self_attn.q_proj"),
        "model.layers.0.self_attn.q_proj"
    );
    assert!(!rules.quantizes_router_linears());
}

#[test]
fn should_recognize_towers_inside_the_model_and_encoder_containers() {
    assert!(is_multimodal_tensor(
        "model.encoder.vision_tower.blocks.0.weight"
    ));
    assert!(is_multimodal_tensor("model.merger.mlp.0.weight"));
    assert!(!is_multimodal_tensor(
        "model.language_model.layers.0.mlp.up_proj.weight"
    ));
    assert!(!is_multimodal_tensor(
        "language_model.model.embed_tokens.weight"
    ));
}
