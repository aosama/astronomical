//! Beside-source coverage for the Qwen3-VL text encoder's expected tensors. The builder is a
//! long table of tensor names, so these tests pin the total, the order of each section and the
//! shapes at its start, middle and end, which is what a refactor of the builder must preserve.

use serde_json::json;
use test_harness::fixtures::image_pipeline_fixture::qwen_image_21_text_encoder_config;

use super::text_encoder_tensor_profiles;
use crate::configuration::QwenImage21TextEncoderConfig;
use crate::tensor_profiles::profile_test_support::{
    assert_every_name_is_unique, assert_names_appear_in_order, assert_profile_at,
    assert_quantized_linear_at, has_tensor, position_of,
};

fn reviewed_text_encoder_config() -> QwenImage21TextEncoderConfig {
    // The shared fixture describes the text geometry and quantization only; the strict parser
    // also requires the upstream version, the streaming token identifiers and the vision tower.
    let mut reviewed_document = qwen_image_21_text_encoder_config();
    reviewed_document["transformers_version"] = json!("4.57.1");
    reviewed_document["image_token_id"] = json!(151_655);
    reviewed_document["video_token_id"] = json!(151_656);
    reviewed_document["vision_start_token_id"] = json!(151_652);
    reviewed_document["vision_end_token_id"] = json!(151_653);
    reviewed_document["text_config"]["bos_token_id"] = json!(151_643);
    reviewed_document["text_config"]["eos_token_id"] = json!(151_645);
    reviewed_document["text_config"]["initializer_range"] = json!(0.02);
    reviewed_document["vision_config"] = json!({
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    });
    let reviewed_document_bytes = serde_json::to_vec(&reviewed_document)
        .expect("the reviewed text encoder document serializes");
    QwenImage21TextEncoderConfig::parse(&reviewed_document_bytes)
        .expect("the reviewed text encoder document parses")
}

/// Tensors that do not depend on the layer or block counts: head, embedding, final norm, patch
/// embedding (2), position embedding (3) and four mergers of ten tensors each.
const TENSORS_WITHOUT_LAYERS_OR_BLOCKS: usize = 6 + 1 + 2 + 3 + 40;
const TENSORS_PER_LANGUAGE_LAYER: usize = 25;
const TENSORS_PER_VISION_BLOCK: usize = 18;

#[test]
fn should_list_the_1438_reviewed_text_encoder_tensors_once_each() {
    let profiles = text_encoder_tensor_profiles(&reviewed_text_encoder_config());

    assert_eq!(profiles.len(), 1438);
    assert_every_name_is_unique(&profiles);
}

#[test]
fn should_start_with_the_language_head_and_embedding_tables() {
    let profiles = text_encoder_tensor_profiles(&reviewed_text_encoder_config());

    assert_quantized_linear_at(&profiles, 0, "language_model.lm_head", 151_936, 512, 64);
    assert_quantized_linear_at(
        &profiles,
        3,
        "language_model.model.embed_tokens",
        151_936,
        512,
        64,
    );
}

#[test]
fn should_start_a_language_layer_with_grouped_query_attention_projections() {
    let profiles = text_encoder_tensor_profiles(&reviewed_text_encoder_config());
    let layer = "language_model.model.layers.0";

    assert_quantized_linear_at(
        &profiles,
        6,
        &format!("{layer}.self_attn.q_proj"),
        4096,
        512,
        64,
    );
    // Grouped-query attention: key and value projections are 8 heads of 128, not 32.
    assert_quantized_linear_at(
        &profiles,
        9,
        &format!("{layer}.self_attn.k_proj"),
        1024,
        512,
        64,
    );
    assert_quantized_linear_at(
        &profiles,
        12,
        &format!("{layer}.self_attn.v_proj"),
        1024,
        512,
        64,
    );
    assert_quantized_linear_at(
        &profiles,
        15,
        &format!("{layer}.self_attn.o_proj"),
        4096,
        512,
        64,
    );
}

#[test]
fn should_follow_the_attention_projections_with_four_full_precision_norms() {
    let profiles = text_encoder_tensor_profiles(&reviewed_text_encoder_config());
    let layer = "language_model.model.layers.0";

    assert_profile_at(
        &profiles,
        18,
        &format!("{layer}.self_attn.q_norm.weight"),
        "BF16",
        &[128],
    );
    assert_profile_at(
        &profiles,
        19,
        &format!("{layer}.self_attn.k_norm.weight"),
        "BF16",
        &[128],
    );
    assert_profile_at(
        &profiles,
        20,
        &format!("{layer}.input_layernorm.weight"),
        "BF16",
        &[4096],
    );
    assert_profile_at(
        &profiles,
        21,
        &format!("{layer}.post_attention_layernorm.weight"),
        "BF16",
        &[4096],
    );
}

#[test]
fn should_finish_a_language_layer_with_the_gate_up_and_down_projections() {
    let profiles = text_encoder_tensor_profiles(&reviewed_text_encoder_config());
    let layer = "language_model.model.layers.0";

    assert_quantized_linear_at(
        &profiles,
        22,
        &format!("{layer}.mlp.gate_proj"),
        12288,
        512,
        64,
    );
    assert_quantized_linear_at(
        &profiles,
        25,
        &format!("{layer}.mlp.up_proj"),
        12288,
        512,
        64,
    );
    assert_quantized_linear_at(
        &profiles,
        28,
        &format!("{layer}.mlp.down_proj"),
        4096,
        1536,
        192,
    );
}

#[test]
fn should_place_the_final_language_norm_before_the_vision_tower() {
    let profiles = text_encoder_tensor_profiles(&reviewed_text_encoder_config());

    assert_profile_at(
        &profiles,
        906,
        "language_model.model.norm.weight",
        "BF16",
        &[4096],
    );
    assert_profile_at(
        &profiles,
        907,
        "vision_tower.patch_embed.proj.weight",
        "BF16",
        &[1152, 3, 2, 16, 16],
    );
    assert_profile_at(
        &profiles,
        908,
        "vision_tower.patch_embed.proj.bias",
        "BF16",
        &[1152],
    );
    assert_quantized_linear_at(&profiles, 909, "vision_tower.pos_embed", 2304, 144, 18);
}

#[test]
fn should_describe_the_first_vision_block_with_a_full_precision_second_projection() {
    let profiles = text_encoder_tensor_profiles(&reviewed_text_encoder_config());
    let block = "vision_tower.blocks.0";

    assert_quantized_linear_at(&profiles, 912, &format!("{block}.attn.qkv"), 3456, 144, 18);
    assert_profile_at(
        &profiles,
        915,
        &format!("{block}.attn.qkv.bias"),
        "BF16",
        &[3456],
    );
    assert_quantized_linear_at(
        &profiles,
        920,
        &format!("{block}.mlp.linear_fc1"),
        4304,
        144,
        18,
    );
    assert_profile_at(
        &profiles,
        924,
        &format!("{block}.mlp.linear_fc2.weight"),
        "BF16",
        &[1152, 4304],
    );
    assert_profile_at(
        &profiles,
        929,
        &format!("{block}.norm2.bias"),
        "BF16",
        &[1152],
    );
}

#[test]
fn should_not_quantize_the_vision_second_projection() {
    let profiles = text_encoder_tensor_profiles(&reviewed_text_encoder_config());

    assert!(!has_tensor(
        &profiles,
        "vision_tower.blocks.0.mlp.linear_fc2.scales"
    ));
    assert!(!has_tensor(
        &profiles,
        "vision_tower.blocks.26.mlp.linear_fc2.biases"
    ));
}

#[test]
fn should_end_with_three_deepstack_mergers_then_the_spatial_merger() {
    let profiles = text_encoder_tensor_profiles(&reviewed_text_encoder_config());

    assert_names_appear_in_order(
        &profiles,
        &[
            "vision_tower.blocks.26.norm2.bias",
            "vision_tower.deepstack_merger_list.0.linear_fc1.weight",
            "vision_tower.deepstack_merger_list.1.linear_fc1.weight",
            "vision_tower.deepstack_merger_list.2.linear_fc1.weight",
            "vision_tower.merger.linear_fc1.weight",
        ],
    );
    assert_quantized_linear_at(
        &profiles,
        1428,
        "vision_tower.merger.linear_fc1",
        4608,
        576,
        72,
    );
    assert_profile_at(
        &profiles,
        1431,
        "vision_tower.merger.linear_fc1.bias",
        "BF16",
        &[4608],
    );
    assert_quantized_linear_at(
        &profiles,
        1432,
        "vision_tower.merger.linear_fc2",
        4096,
        576,
        72,
    );
    assert_eq!(
        position_of(&profiles, "vision_tower.merger.norm.bias"),
        1437
    );
}

#[test]
fn should_give_the_spatial_merger_the_vision_width_norm_and_the_deepstack_mergers_the_merged_width()
{
    let profiles = text_encoder_tensor_profiles(&reviewed_text_encoder_config());

    assert_profile_at(
        &profiles,
        1426,
        "vision_tower.deepstack_merger_list.2.norm.weight",
        "BF16",
        &[4608],
    );
    assert_profile_at(
        &profiles,
        1436,
        "vision_tower.merger.norm.weight",
        "BF16",
        &[1152],
    );
    assert_profile_at(
        &profiles,
        1437,
        "vision_tower.merger.norm.bias",
        "BF16",
        &[1152],
    );
}

#[test]
fn should_emit_only_the_fixed_tensors_for_a_model_without_layers_or_blocks() {
    let config = QwenImage21TextEncoderConfig {
        num_hidden_layers: 0,
        vision_depth: 0,
        ..reviewed_text_encoder_config()
    };

    let profiles = text_encoder_tensor_profiles(&config);

    assert_eq!(profiles.len(), TENSORS_WITHOUT_LAYERS_OR_BLOCKS);
    assert_names_appear_in_order(
        &profiles,
        &[
            "language_model.lm_head.weight",
            "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight",
            "vision_tower.patch_embed.proj.weight",
            "vision_tower.pos_embed.weight",
            "vision_tower.deepstack_merger_list.0.linear_fc1.weight",
            "vision_tower.merger.norm.bias",
        ],
    );
}

#[test]
fn should_add_a_fixed_number_of_tensors_for_each_layer_and_vision_block() {
    let config = QwenImage21TextEncoderConfig {
        num_hidden_layers: 3,
        vision_depth: 2,
        ..reviewed_text_encoder_config()
    };

    let profiles = text_encoder_tensor_profiles(&config);

    assert_eq!(
        profiles.len(),
        TENSORS_WITHOUT_LAYERS_OR_BLOCKS
            + 3 * TENSORS_PER_LANGUAGE_LAYER
            + 2 * TENSORS_PER_VISION_BLOCK
    );
    assert!(has_tensor(
        &profiles,
        "language_model.model.layers.2.mlp.down_proj.weight"
    ));
    assert!(!has_tensor(
        &profiles,
        "language_model.model.layers.3.mlp.down_proj.weight"
    ));
    assert!(!has_tensor(&profiles, "vision_tower.blocks.2.norm1.weight"));
}
