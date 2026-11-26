//! Beside-source coverage for the transformer's expected tensors: the shared projections, then
//! 23 tensors per block, then the output projections. The tests pin the total, the order and the
//! shapes at the start, middle and end of that sequence.

use serde_json::json;
use test_harness::fixtures::image_pipeline_fixture::qwen_image_21_transformer_config;

use super::transformer_tensor_profiles;
use crate::configuration::QwenImage21TransformerConfig;
use crate::tensor_profiles::profile_test_support::{
    assert_every_name_is_unique, assert_names_appear_in_order, assert_profile_at,
    assert_quantized_linear_at, has_tensor, position_of,
};

fn reviewed_transformer_config() -> QwenImage21TransformerConfig {
    // The shared fixture omits the diffusers version the strict parser requires.
    let mut reviewed_document = qwen_image_21_transformer_config();
    reviewed_document["_diffusers_version"] = json!("0.37.0.dev0");
    let reviewed_document_bytes = serde_json::to_vec(&reviewed_document)
        .expect("the reviewed transformer document serializes");
    QwenImage21TransformerConfig::parse(&reviewed_document_bytes)
        .expect("the reviewed transformer document parses")
}

/// Nineteen tensors ahead of the blocks (six projections and one norm), six after them (the two
/// output projections).
const TENSORS_AROUND_THE_BLOCKS: usize = 19 + 6;
const TENSORS_PER_BLOCK: usize = 23;

#[test]
fn should_list_the_761_reviewed_transformer_tensors_once_each() {
    let profiles = transformer_tensor_profiles(&reviewed_transformer_config());

    assert_eq!(profiles.len(), 761);
    assert_every_name_is_unique(&profiles);
}

#[test]
fn should_start_with_the_shared_projections_before_any_block() {
    let profiles = transformer_tensor_profiles(&reviewed_transformer_config());

    assert_quantized_linear_at(&profiles, 0, "img_in", 4096, 8, 1);
    assert_quantized_linear_at(&profiles, 3, "txt_in.in_layer", 4096, 512, 64);
    assert_quantized_linear_at(&profiles, 6, "txt_in.out_layer", 4096, 512, 64);
    assert_profile_at(&profiles, 9, "txt_in.text_norm.weight", "BF16", &[4096]);
    // The sinusoidal timestep projection is 256 channels wide, not the inner width.
    assert_quantized_linear_at(&profiles, 10, "time_text_embed.linear_1", 4096, 32, 4);
    assert_quantized_linear_at(&profiles, 13, "time_text_embed.linear_2", 4096, 512, 64);
    // One shared modulation projection feeds every block: inner width to four times inner width.
    assert_quantized_linear_at(&profiles, 16, "modulation.0", 16384, 512, 64);
}

#[test]
fn should_describe_a_block_as_attention_then_norms_then_feed_forward() {
    let profiles = transformer_tensor_profiles(&reviewed_transformer_config());
    let block = "transformer_blocks.0";

    assert_quantized_linear_at(&profiles, 19, &format!("{block}.attn.to_q"), 4096, 512, 64);
    assert_quantized_linear_at(&profiles, 22, &format!("{block}.attn.to_k"), 4096, 512, 64);
    assert_quantized_linear_at(&profiles, 25, &format!("{block}.attn.to_v"), 4096, 512, 64);
    assert_quantized_linear_at(
        &profiles,
        28,
        &format!("{block}.attn.to_out.0"),
        4096,
        512,
        64,
    );
    assert_profile_at(
        &profiles,
        31,
        &format!("{block}.attn.norm_q.weight"),
        "BF16",
        &[128],
    );
    assert_profile_at(
        &profiles,
        32,
        &format!("{block}.attn.norm_k.weight"),
        "BF16",
        &[128],
    );
    assert_quantized_linear_at(
        &profiles,
        33,
        &format!("{block}.img_mlp.gate_layer"),
        12288,
        512,
        64,
    );
    assert_quantized_linear_at(
        &profiles,
        36,
        &format!("{block}.img_mlp.proj"),
        12288,
        512,
        64,
    );
    assert_quantized_linear_at(
        &profiles,
        39,
        &format!("{block}.img_mlp.out"),
        4096,
        1536,
        192,
    );
}

#[test]
fn should_end_with_the_last_block_then_the_output_projections() {
    let profiles = transformer_tensor_profiles(&reviewed_transformer_config());

    assert_quantized_linear_at(
        &profiles,
        752,
        "transformer_blocks.31.img_mlp.out",
        4096,
        1536,
        192,
    );
    assert_quantized_linear_at(&profiles, 755, "norm_out.linear", 4096, 512, 64);
    assert_quantized_linear_at(&profiles, 758, "proj_out", 64, 512, 64);
    assert_eq!(position_of(&profiles, "proj_out.biases"), 760);
}

#[test]
fn should_emit_only_the_shared_and_output_projections_for_a_model_without_blocks() {
    let config = QwenImage21TransformerConfig {
        num_layers: 0,
        ..reviewed_transformer_config()
    };

    let profiles = transformer_tensor_profiles(&config);

    assert_eq!(profiles.len(), TENSORS_AROUND_THE_BLOCKS);
    assert!(!has_tensor(
        &profiles,
        "transformer_blocks.0.attn.to_q.weight"
    ));
    assert_names_appear_in_order(
        &profiles,
        &[
            "modulation.0.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
        ],
    );
}

#[test]
fn should_add_the_same_number_of_tensors_for_every_block() {
    let config = QwenImage21TransformerConfig {
        num_layers: 2,
        ..reviewed_transformer_config()
    };

    let profiles = transformer_tensor_profiles(&config);

    assert_eq!(
        profiles.len(),
        TENSORS_AROUND_THE_BLOCKS + 2 * TENSORS_PER_BLOCK
    );
    assert!(has_tensor(
        &profiles,
        "transformer_blocks.1.img_mlp.out.weight"
    ));
    assert!(!has_tensor(
        &profiles,
        "transformer_blocks.2.img_mlp.out.weight"
    ));
}

#[test]
fn should_widen_the_patch_projections_with_the_patch_size() {
    let config = QwenImage21TransformerConfig {
        patch_size: 2,
        num_layers: 0,
        ..reviewed_transformer_config()
    };

    let profiles = transformer_tensor_profiles(&config);

    // 64 channels in 2x2 patches: 256 input features pack into 32 words, in 4 groups of 64.
    assert_quantized_linear_at(&profiles, 0, "img_in", 4096, 32, 4);
    assert_quantized_linear_at(&profiles, 19, "norm_out.linear", 4096, 512, 64);
    assert_quantized_linear_at(&profiles, 22, "proj_out", 256, 512, 64);
}
