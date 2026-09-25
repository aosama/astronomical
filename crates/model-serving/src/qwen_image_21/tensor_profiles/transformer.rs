//! The transformer's expected tensors: shared projections, then one section per block.

use super::{QwenImage21TensorProfile, quantized_linear};
use crate::qwen_image_21::configuration::QwenImage21TransformerConfig;

/// Expected physical tensors of the 32-block single-stream transformer (761 tensors).
#[must_use]
pub fn transformer_tensor_profiles(
    config: &QwenImage21TransformerConfig,
) -> Vec<QwenImage21TensorProfile> {
    let inner = config.inner_dim();
    let mlp_hidden = config.mlp_hidden_dim();
    let bits = config.quantization_bits;
    let group = config.quantization_group_size;
    let mut profiles = Vec::new();

    quantized_linear(
        &mut profiles,
        "img_in",
        inner,
        config.in_channels * config.patch_size * config.patch_size,
        bits,
        group,
    );
    quantized_linear(
        &mut profiles,
        "txt_in.in_layer",
        inner,
        config.context_in_dim,
        bits,
        group,
    );
    quantized_linear(&mut profiles, "txt_in.out_layer", inner, inner, bits, group);
    profiles.push(QwenImage21TensorProfile::bf16(
        "txt_in.text_norm.weight",
        vec![config.context_in_dim],
    ));
    // Sinusoidal timestep projection width: 256 channels into the inner dim, then the inner dim.
    quantized_linear(
        &mut profiles,
        "time_text_embed.linear_1",
        inner,
        256,
        bits,
        group,
    );
    quantized_linear(
        &mut profiles,
        "time_text_embed.linear_2",
        inner,
        inner,
        bits,
        group,
    );
    // One shared modulation projection: inner -> 4 * inner (mod1.scale, mod1.gate, mod2.scale, mod2.gate).
    quantized_linear(&mut profiles, "modulation.0", 4 * inner, inner, bits, group);

    for block_index in 0..config.num_layers {
        let block = format!("transformer_blocks.{block_index}");
        for projection in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
            quantized_linear(
                &mut profiles,
                &format!("{block}.{projection}"),
                inner,
                inner,
                bits,
                group,
            );
        }
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{block}.attn.norm_q.weight"),
            vec![config.attention_head_dim],
        ));
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{block}.attn.norm_k.weight"),
            vec![config.attention_head_dim],
        ));
        for projection in ["img_mlp.gate_layer", "img_mlp.proj"] {
            quantized_linear(
                &mut profiles,
                &format!("{block}.{projection}"),
                mlp_hidden,
                inner,
                bits,
                group,
            );
        }
        quantized_linear(
            &mut profiles,
            &format!("{block}.img_mlp.out"),
            inner,
            mlp_hidden,
            bits,
            group,
        );
    }

    quantized_linear(&mut profiles, "norm_out.linear", inner, inner, bits, group);
    quantized_linear(
        &mut profiles,
        "proj_out",
        config.out_channels * config.patch_size * config.patch_size,
        inner,
        bits,
        group,
    );
    profiles
}
