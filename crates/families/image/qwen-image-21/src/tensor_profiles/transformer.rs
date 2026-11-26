//! The transformer's expected tensors: shared projections, then one section per block.

use super::QwenImage21TensorProfile;
use crate::configuration::QwenImage21TransformerConfig;

/// Expected physical tensors of the 32-block single-stream transformer (761 tensors).
#[must_use]
pub fn transformer_tensor_profiles(
    config: &QwenImage21TransformerConfig,
) -> Vec<QwenImage21TensorProfile> {
    let mut profiles = Vec::new();

    push_shared_projections(&mut profiles, config);
    for block_index in 0..config.num_layers {
        push_transformer_block(&mut profiles, config, block_index);
    }
    push_output_projections(&mut profiles, config);

    profiles
}

/// The tensors ahead of the first block: the image and text input projections with the text norm,
/// the timestep embedding, and the one modulation projection, which lives outside every
/// `transformer_blocks.N` prefix.
fn push_shared_projections(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    config: &QwenImage21TransformerConfig,
) {
    let inner = config.inner_dim();
    let bits = config.quantization_bits;
    let group = config.quantization_group_size;

    super::quantized_linear(
        profiles,
        "img_in",
        inner,
        config.in_channels * config.patch_size * config.patch_size,
        bits,
        group,
    );
    super::quantized_linear(
        profiles,
        "txt_in.in_layer",
        inner,
        config.context_in_dim,
        bits,
        group,
    );
    super::quantized_linear(profiles, "txt_in.out_layer", inner, inner, bits, group);
    profiles.push(QwenImage21TensorProfile::bf16(
        "txt_in.text_norm.weight",
        vec![config.context_in_dim],
    ));
    // Sinusoidal timestep projection width: 256 channels into the inner dim, then the inner dim.
    super::quantized_linear(
        profiles,
        "time_text_embed.linear_1",
        inner,
        256,
        bits,
        group,
    );
    super::quantized_linear(
        profiles,
        "time_text_embed.linear_2",
        inner,
        inner,
        bits,
        group,
    );
    // One shared modulation projection: inner -> 4 * inner (mod1.scale, mod1.gate, mod2.scale, mod2.gate).
    super::quantized_linear(profiles, "modulation.0", 4 * inner, inner, bits, group);
}

/// One block: attention projections and query/key norms, then the gated feed-forward.
fn push_transformer_block(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    config: &QwenImage21TransformerConfig,
    block_index: usize,
) {
    let inner = config.inner_dim();
    let mlp_hidden = config.mlp_hidden_dim();
    let bits = config.quantization_bits;
    let group = config.quantization_group_size;
    let block = format!("transformer_blocks.{block_index}");

    for projection in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
        super::quantized_linear(
            profiles,
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
        super::quantized_linear(
            profiles,
            &format!("{block}.{projection}"),
            mlp_hidden,
            inner,
            bits,
            group,
        );
    }
    super::quantized_linear(
        profiles,
        &format!("{block}.img_mlp.out"),
        inner,
        mlp_hidden,
        bits,
        group,
    );
}

/// The projections after the last block: `norm_out.linear`, then `proj_out`, which writes
/// `out_channels` values for every pixel of a patch.
fn push_output_projections(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    config: &QwenImage21TransformerConfig,
) {
    let inner = config.inner_dim();
    let bits = config.quantization_bits;
    let group = config.quantization_group_size;

    super::quantized_linear(profiles, "norm_out.linear", inner, inner, bits, group);
    super::quantized_linear(
        profiles,
        "proj_out",
        config.out_channels * config.patch_size * config.patch_size,
        inner,
        bits,
        group,
    );
}

#[cfg(test)]
#[path = "transformer_tests.rs"]
mod transformer_tests;
