//! The VAE's expected tensors, all unquantized F32.

use super::QwenImage21TensorProfile;
use crate::qwen_image_21::configuration::QwenImage21VaeConfig;

fn conv3d(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    prefix: &str,
    out: usize,
    in_channels: usize,
    kernel: [usize; 2],
) {
    profiles.push(QwenImage21TensorProfile::f32(
        &format!("{prefix}.weight"),
        vec![out, kernel[0], kernel[1], in_channels],
    ));
    profiles.push(QwenImage21TensorProfile::f32(
        &format!("{prefix}.bias"),
        vec![out],
    ));
}

fn conv1x1(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    prefix: &str,
    out: usize,
    in_channels: usize,
) {
    conv3d(profiles, prefix, out, in_channels, [1, 1]);
}

fn resnet(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    has_shortcut: bool,
) {
    conv3d(
        profiles,
        &format!("{prefix}.conv1"),
        out_channels,
        in_channels,
        [3, 3],
    );
    conv3d(
        profiles,
        &format!("{prefix}.conv2"),
        out_channels,
        out_channels,
        [3, 3],
    );
    profiles.push(QwenImage21TensorProfile::f32(
        &format!("{prefix}.norm1.gamma"),
        vec![in_channels],
    ));
    profiles.push(QwenImage21TensorProfile::f32(
        &format!("{prefix}.norm2.gamma"),
        vec![out_channels],
    ));
    if has_shortcut {
        conv1x1(
            profiles,
            &format!("{prefix}.conv_shortcut"),
            out_channels,
            in_channels,
        );
    }
}

fn attention(profiles: &mut Vec<QwenImage21TensorProfile>, prefix: &str, channels: usize) {
    profiles.push(QwenImage21TensorProfile::f32(
        &format!("{prefix}.norm.gamma"),
        vec![channels],
    ));
    conv1x1(
        profiles,
        &format!("{prefix}.to_qkv"),
        3 * channels,
        channels,
    );
    conv1x1(profiles, &format!("{prefix}.proj"), channels, channels);
}

/// Expected physical tensors of the 3D causal VAE (238 tensors, all F32).
#[must_use]
pub fn vae_tensor_profiles(config: &QwenImage21VaeConfig) -> Vec<QwenImage21TensorProfile> {
    // Channel ladders. Decoder block `i` operates at `decoder_base_dim * dim_mult[4 - i]`
    // (1152, 1152, 576, 288, 144); encoder block `i` at `base_dim * dim_mult[i]`
    // (96, 192, 384, 768, 768). Block inputs are the previous block's operating width (or the
    // conv_in output for block 0).
    let decoder_width =
        |index: usize| config.decoder_base_dim * config.dim_mult[4 - index] as usize;
    let encoder_width = |index: usize| config.base_dim * config.dim_mult[index] as usize;
    let encoder_latent_width = 2 * config.z_dim;
    let mut profiles = Vec::new();

    // Decoder: conv_in lifts the `z_dim` latent channels into the first up-block width.
    conv3d(
        &mut profiles,
        "decoder.conv_in",
        decoder_width(0),
        config.z_dim,
        [3, 3],
    );
    // Up blocks 0..3 spatially upsample; blocks 0..2 also temporally upsample via `time_conv`
    // (the reviewed artifact topology — not the mirror of the encoder's 1..3 layout). The
    // upsampler runs at the block's operating width; `time_conv` doubles the channels.
    for block_index in 0..4 {
        let prefix = format!("decoder.up_blocks.{block_index}");
        let operating = decoder_width(block_index);
        let input = if block_index == 0 {
            decoder_width(0)
        } else {
            decoder_width(block_index - 1)
        };
        let has_shortcut = input != operating;
        for resnet_index in 0..config.num_res_blocks + 1 {
            resnet(
                &mut profiles,
                &format!("{prefix}.resnets.{resnet_index}"),
                if resnet_index == 0 { input } else { operating },
                operating,
                resnet_index == 0 && has_shortcut,
            );
        }
        let upsampler = format!("{prefix}.upsampler");
        if block_index < 3 {
            conv1x1(
                &mut profiles,
                &format!("{upsampler}.time_conv"),
                2 * operating,
                operating,
            );
        }
        conv3d(
            &mut profiles,
            &format!("{upsampler}.resample.1"),
            operating,
            operating,
            [3, 3],
        );
    }
    // Final up block: no upsampler, channel change to the decoder base width.
    let final_prefix = "decoder.up_blocks.4";
    let final_operating = decoder_width(4);
    let final_input = decoder_width(3);
    for resnet_index in 0..config.num_res_blocks + 1 {
        resnet(
            &mut profiles,
            &format!("{final_prefix}.resnets.{resnet_index}"),
            if resnet_index == 0 {
                final_input
            } else {
                final_operating
            },
            final_operating,
            resnet_index == 0,
        );
    }
    let mid = "decoder.mid_block";
    for resnet_index in 0..config.num_res_blocks {
        resnet(
            &mut profiles,
            &format!("{mid}.resnets.{resnet_index}"),
            decoder_width(0),
            decoder_width(0),
            false,
        );
    }
    attention(
        &mut profiles,
        &format!("{mid}.attentions.0"),
        decoder_width(0),
    );
    profiles.push(QwenImage21TensorProfile::f32(
        "decoder.norm_out.gamma",
        vec![final_operating],
    ));
    conv3d(
        &mut profiles,
        "decoder.conv_out",
        config.out_channels,
        final_operating,
        [3, 3],
    );

    // Encoder: conv_in from pixel channels into the first down-block width.
    conv3d(
        &mut profiles,
        "encoder.conv_in",
        encoder_width(0),
        config.in_channels,
        [3, 3],
    );
    for block_index in 0..5 {
        let prefix = format!("encoder.down_blocks.{block_index}");
        let operating = encoder_width(block_index);
        let input = if block_index == 0 {
            encoder_width(0)
        } else {
            encoder_width(block_index - 1)
        };
        let has_shortcut = input != operating;
        for resnet_index in 0..config.num_res_blocks {
            resnet(
                &mut profiles,
                &format!("{prefix}.resnets.{resnet_index}"),
                if resnet_index == 0 { input } else { operating },
                operating,
                resnet_index == 0 && has_shortcut,
            );
        }
        if block_index < 4 {
            let downsampler = format!("{prefix}.downsampler");
            conv3d(
                &mut profiles,
                &format!("{downsampler}.resample.1"),
                operating,
                operating,
                [3, 3],
            );
            // Temporal downsampling in the reviewed artifact sits in down blocks 1..3, matching
            // `temperal_downsample = [false, true, true, true]`.
            if block_index > 0 {
                conv1x1(
                    &mut profiles,
                    &format!("{downsampler}.time_conv"),
                    operating,
                    operating,
                );
            }
        }
    }
    let mid = "encoder.mid_block";
    let mid_channels = encoder_width(4);
    for resnet_index in 0..config.num_res_blocks {
        resnet(
            &mut profiles,
            &format!("{mid}.resnets.{resnet_index}"),
            mid_channels,
            mid_channels,
            false,
        );
    }
    attention(&mut profiles, &format!("{mid}.attentions.0"), mid_channels);
    profiles.push(QwenImage21TensorProfile::f32(
        "encoder.norm_out.gamma",
        vec![mid_channels],
    ));
    conv3d(
        &mut profiles,
        "encoder.conv_out",
        encoder_latent_width,
        mid_channels,
        [3, 3],
    );

    // The encoder latent width is projected down to `z_dim` for the KL posterior.
    conv1x1(
        &mut profiles,
        "quant_conv",
        encoder_latent_width,
        encoder_latent_width,
    );
    conv1x1(&mut profiles, "post_quant_conv", config.z_dim, config.z_dim);

    profiles
}
