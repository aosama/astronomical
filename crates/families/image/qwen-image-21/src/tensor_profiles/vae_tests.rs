//! Beside-source coverage for the 3D causal VAE's expected tensors. The builder walks the
//! decoder, then the encoder, then the posterior projections, so these tests pin the total, the
//! section order and the width ladder at the points where the reviewed topology is irregular:
//! which blocks carry a shortcut and which blocks change the frame count.

use serde_json::json;
use test_harness::fixtures::image_pipeline_fixture::qwen_image_21_vae_config;

use super::vae_tensor_profiles;
use crate::configuration::QwenImage21VaeConfig;
use crate::tensor_profiles::profile_test_support::{
    assert_every_name_is_unique, assert_names_appear_in_order, assert_profile_at, has_tensor,
};

fn reviewed_vae_config() -> QwenImage21VaeConfig {
    // The shared fixture omits the diffusers version the strict parser requires.
    let mut reviewed_document = qwen_image_21_vae_config();
    reviewed_document["_diffusers_version"] = json!("0.37.0.dev0");
    let reviewed_document_bytes =
        serde_json::to_vec(&reviewed_document).expect("the reviewed VAE document serializes");
    QwenImage21VaeConfig::parse(&reviewed_document_bytes).expect("the reviewed VAE document parses")
}

fn vae_config_with_ladder(num_res_blocks: usize, dim_mult: [u32; 5]) -> QwenImage21VaeConfig {
    let mut config = reviewed_vae_config();
    config.num_res_blocks = num_res_blocks;
    config.dim_mult = dim_mult;
    config
}

#[test]
fn should_list_the_238_reviewed_vae_tensors_once_each_in_full_precision() {
    let profiles = vae_tensor_profiles(&reviewed_vae_config());

    assert_eq!(profiles.len(), 238);
    assert_every_name_is_unique(&profiles);
    assert!(profiles.iter().all(|profile| profile.dtype == "F32"));
}

#[test]
fn should_start_with_the_decoder_input_convolution_and_first_resnet() {
    let profiles = vae_tensor_profiles(&reviewed_vae_config());

    assert_profile_at(
        &profiles,
        0,
        "decoder.conv_in.weight",
        "F32",
        &[1152, 3, 3, 64],
    );
    assert_profile_at(&profiles, 1, "decoder.conv_in.bias", "F32", &[1152]);
    assert_profile_at(
        &profiles,
        2,
        "decoder.up_blocks.0.resnets.0.conv1.weight",
        "F32",
        &[1152, 3, 3, 1152],
    );
}

#[test]
fn should_add_a_decoder_shortcut_only_where_the_block_width_changes() {
    let profiles = vae_tensor_profiles(&reviewed_vae_config());

    assert!(!has_tensor(
        &profiles,
        "decoder.up_blocks.0.resnets.0.conv_shortcut.weight"
    ));
    assert!(!has_tensor(
        &profiles,
        "decoder.up_blocks.1.resnets.0.conv_shortcut.weight"
    ));
    assert_profile_at(
        &profiles,
        52,
        "decoder.up_blocks.2.resnets.0.conv_shortcut.weight",
        "F32",
        &[576, 1, 1, 1152],
    );
    assert!(!has_tensor(
        &profiles,
        "decoder.up_blocks.2.resnets.1.conv_shortcut.weight"
    ));
    assert!(has_tensor(
        &profiles,
        "decoder.up_blocks.3.resnets.0.conv_shortcut.weight"
    ));
}

#[test]
fn should_always_give_the_final_decoder_block_a_shortcut_to_the_base_width() {
    let profiles = vae_tensor_profiles(&reviewed_vae_config());

    assert_profile_at(
        &profiles,
        98,
        "decoder.up_blocks.4.resnets.0.conv_shortcut.weight",
        "F32",
        &[144, 1, 1, 288],
    );
    assert!(!has_tensor(
        &profiles,
        "decoder.up_blocks.4.upsampler.resample.1.weight"
    ));
}

#[test]
fn should_upsample_in_time_only_in_the_first_three_decoder_blocks() {
    let profiles = vae_tensor_profiles(&reviewed_vae_config());

    assert_profile_at(
        &profiles,
        20,
        "decoder.up_blocks.0.upsampler.time_conv.weight",
        "F32",
        &[2304, 1, 1, 1152],
    );
    assert!(has_tensor(
        &profiles,
        "decoder.up_blocks.1.upsampler.time_conv.weight"
    ));
    assert!(has_tensor(
        &profiles,
        "decoder.up_blocks.2.upsampler.time_conv.weight"
    ));
    assert!(!has_tensor(
        &profiles,
        "decoder.up_blocks.3.upsampler.time_conv.weight"
    ));
    assert!(has_tensor(
        &profiles,
        "decoder.up_blocks.3.upsampler.resample.1.weight"
    ));
}

#[test]
fn should_place_the_decoder_mid_block_after_the_up_blocks_and_before_the_encoder() {
    let profiles = vae_tensor_profiles(&reviewed_vae_config());

    assert_names_appear_in_order(
        &profiles,
        &[
            "decoder.up_blocks.3.upsampler.resample.1.weight",
            "decoder.up_blocks.4.resnets.2.norm2.gamma",
            "decoder.mid_block.resnets.0.conv1.weight",
            "decoder.mid_block.attentions.0.proj.bias",
            "decoder.norm_out.gamma",
            "decoder.conv_out.weight",
            "encoder.conv_in.weight",
        ],
    );
    assert_profile_at(
        &profiles,
        112,
        "decoder.mid_block.resnets.0.conv1.weight",
        "F32",
        &[1152, 3, 3, 1152],
    );
    assert_profile_at(&profiles, 129, "decoder.norm_out.gamma", "F32", &[144]);
    assert_profile_at(
        &profiles,
        130,
        "decoder.conv_out.weight",
        "F32",
        &[4, 3, 3, 144],
    );
}

#[test]
fn should_downsample_in_time_only_in_encoder_blocks_one_to_three() {
    let profiles = vae_tensor_profiles(&reviewed_vae_config());

    assert!(has_tensor(
        &profiles,
        "encoder.down_blocks.0.downsampler.resample.1.weight"
    ));
    assert!(!has_tensor(
        &profiles,
        "encoder.down_blocks.0.downsampler.time_conv.weight"
    ));
    assert_profile_at(
        &profiles,
        164,
        "encoder.down_blocks.1.downsampler.time_conv.weight",
        "F32",
        &[192, 1, 1, 192],
    );
    assert!(has_tensor(
        &profiles,
        "encoder.down_blocks.3.downsampler.time_conv.weight"
    ));
}

#[test]
fn should_leave_the_last_encoder_block_without_a_downsampler_or_shortcut() {
    let profiles = vae_tensor_profiles(&reviewed_vae_config());

    assert!(has_tensor(
        &profiles,
        "encoder.down_blocks.4.resnets.1.conv2.weight"
    ));
    assert!(!has_tensor(
        &profiles,
        "encoder.down_blocks.4.downsampler.resample.1.weight"
    ));
    assert!(!has_tensor(
        &profiles,
        "encoder.down_blocks.4.resnets.0.conv_shortcut.weight"
    ));
}

#[test]
fn should_end_with_the_encoder_mid_block_and_the_posterior_projections() {
    let profiles = vae_tensor_profiles(&reviewed_vae_config());

    assert_profile_at(&profiles, 231, "encoder.norm_out.gamma", "F32", &[768]);
    assert_profile_at(
        &profiles,
        232,
        "encoder.conv_out.weight",
        "F32",
        &[128, 3, 3, 768],
    );
    assert_profile_at(
        &profiles,
        234,
        "quant_conv.weight",
        "F32",
        &[128, 1, 1, 128],
    );
    assert_profile_at(
        &profiles,
        236,
        "post_quant_conv.weight",
        "F32",
        &[64, 1, 1, 64],
    );
    assert_profile_at(&profiles, 237, "post_quant_conv.bias", "F32", &[64]);
}

#[test]
fn should_use_one_encoder_resnet_and_two_decoder_resnets_when_the_config_asks_for_one() {
    let profiles = vae_tensor_profiles(&vae_config_with_ladder(1, [1, 2, 4, 8, 8]));

    assert_eq!(profiles.len(), 166);
    assert!(has_tensor(
        &profiles,
        "decoder.up_blocks.0.resnets.1.conv1.weight"
    ));
    assert!(!has_tensor(
        &profiles,
        "decoder.up_blocks.0.resnets.2.conv1.weight"
    ));
    assert!(has_tensor(
        &profiles,
        "encoder.down_blocks.0.resnets.0.conv1.weight"
    ));
    assert!(!has_tensor(
        &profiles,
        "encoder.down_blocks.0.resnets.1.conv1.weight"
    ));
}

#[test]
fn should_skip_every_shortcut_but_the_final_decoder_one_on_a_flat_width_ladder() {
    let profiles = vae_tensor_profiles(&vae_config_with_ladder(3, [2, 2, 2, 2, 2]));

    assert_eq!(profiles.len(), 300);
    assert!(!has_tensor(
        &profiles,
        "decoder.up_blocks.2.resnets.0.conv_shortcut.weight"
    ));
    assert!(!has_tensor(
        &profiles,
        "encoder.down_blocks.2.resnets.0.conv_shortcut.weight"
    ));
    assert!(has_tensor(
        &profiles,
        "decoder.up_blocks.4.resnets.0.conv_shortcut.weight"
    ));
}
