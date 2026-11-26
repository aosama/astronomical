//! The Qwen3-VL text encoder's expected tensors: language model plus the validated vision
//! tower. The T2I path executes only the language model; the tower is still part of the
//! reviewed file, so the profile covers both.

use super::QwenImage21TensorProfile;
use crate::configuration::QwenImage21TextEncoderConfig;

/// The four mergers that project vision features into the text width: three deepstack mergers
/// and, last, the single spatial merger.
const VISION_MERGER_PREFIXES: [&str; 4] = [
    "vision_tower.deepstack_merger_list.0",
    "vision_tower.deepstack_merger_list.1",
    "vision_tower.deepstack_merger_list.2",
    "vision_tower.merger",
];

/// Expected physical tensors of the Qwen3-VL text encoder (1438 tensors), language model and
/// vision tower. The T2I path executes only the language model; the vision tower is validated so
/// the artifact contract covers the full file.
#[must_use]
pub fn text_encoder_tensor_profiles(
    config: &QwenImage21TextEncoderConfig,
) -> Vec<QwenImage21TensorProfile> {
    let mut profiles = Vec::new();

    push_language_head_and_embeddings(&mut profiles, config);
    for layer_index in 0..config.num_hidden_layers {
        push_language_layer(&mut profiles, config, layer_index);
    }
    profiles.push(QwenImage21TensorProfile::bf16(
        "language_model.model.norm.weight",
        vec![config.hidden_size],
    ));

    push_vision_embeddings(&mut profiles, config);
    for block_index in 0..config.vision_depth {
        push_vision_block(&mut profiles, config, block_index);
    }
    for merger_prefix in VISION_MERGER_PREFIXES {
        push_vision_merger(&mut profiles, config, merger_prefix);
    }

    profiles
}

/// Adds one quantized linear under the text encoder's own quantization settings.
fn push_quantized_linear(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    config: &QwenImage21TextEncoderConfig,
    prefix: &str,
    out_features: usize,
    in_features: usize,
) {
    super::quantized_linear(
        profiles,
        prefix,
        out_features,
        in_features,
        config.quantization_bits,
        config.quantization_group_size,
    );
}

fn push_language_head_and_embeddings(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    config: &QwenImage21TextEncoderConfig,
) {
    for prefix in [
        "language_model.lm_head",
        "language_model.model.embed_tokens",
    ] {
        push_quantized_linear(
            profiles,
            config,
            prefix,
            config.vocab_size,
            config.hidden_size,
        );
    }
}

/// One language-model layer: grouped-query attention, its norms, then the SwiGLU MLP.
fn push_language_layer(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    config: &QwenImage21TextEncoderConfig,
    layer_index: usize,
) {
    let hidden = config.hidden_size;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let layer = format!("language_model.model.layers.{layer_index}");
    for (projection, out_features) in [
        ("q_proj", hidden),
        ("k_proj", kv_dim),
        ("v_proj", kv_dim),
        ("o_proj", hidden),
    ] {
        push_quantized_linear(
            profiles,
            config,
            &format!("{layer}.self_attn.{projection}"),
            out_features,
            hidden,
        );
    }
    for (norm, width) in [
        ("self_attn.q_norm", config.head_dim),
        ("self_attn.k_norm", config.head_dim),
        ("input_layernorm", hidden),
        ("post_attention_layernorm", hidden),
    ] {
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{layer}.{norm}.weight"),
            vec![width],
        ));
    }
    for projection in ["mlp.gate_proj", "mlp.up_proj"] {
        push_quantized_linear(
            profiles,
            config,
            &format!("{layer}.{projection}"),
            config.intermediate_size,
            hidden,
        );
    }
    push_quantized_linear(
        profiles,
        config,
        &format!("{layer}.mlp.down_proj"),
        hidden,
        config.intermediate_size,
    );
}

/// Vision tower. `patch_embed.proj` is a full-precision Conv3d kernel
/// [out, in, t, h, w]; `pos_embed` is a quantized embedding of the position count.
fn push_vision_embeddings(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    config: &QwenImage21TextEncoderConfig,
) {
    let vision_hidden = config.vision_hidden_size;
    profiles.push(QwenImage21TensorProfile::bf16(
        "vision_tower.patch_embed.proj.weight",
        vec![
            vision_hidden,
            3,
            config.vision_temporal_patch_size as usize,
            config.vision_patch_size as usize,
            config.vision_patch_size as usize,
        ],
    ));
    profiles.push(QwenImage21TensorProfile::bf16(
        "vision_tower.patch_embed.proj.bias",
        vec![vision_hidden],
    ));
    push_quantized_linear(
        profiles,
        config,
        "vision_tower.pos_embed",
        config.vision_num_position_embeddings,
        vision_hidden,
    );
}

/// One vision block: fused QKV and output projection, the two-layer MLP, then both norms.
fn push_vision_block(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    config: &QwenImage21TextEncoderConfig,
    block_index: usize,
) {
    let vision_hidden = config.vision_hidden_size;
    let vision_qkv = 3 * vision_hidden;
    let vision_intermediate = config.vision_intermediate_size;
    let block = format!("vision_tower.blocks.{block_index}");
    // Quantized linears in the tower keep their original linear bias (`.bias`) in addition
    // to the per-group dequantization bias (`.biases`).
    for (projection, out_features, in_features) in [
        ("attn.qkv", vision_qkv, vision_hidden),
        ("attn.proj", vision_hidden, vision_hidden),
        ("mlp.linear_fc1", vision_intermediate, vision_hidden),
    ] {
        push_quantized_linear(
            profiles,
            config,
            &format!("{block}.{projection}"),
            out_features,
            in_features,
        );
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{block}.{projection}.bias"),
            vec![out_features],
        ));
    }
    // fc2 stays full-precision in the reviewed artifact.
    profiles.push(QwenImage21TensorProfile::bf16(
        &format!("{block}.mlp.linear_fc2.weight"),
        vec![vision_hidden, vision_intermediate],
    ));
    profiles.push(QwenImage21TensorProfile::bf16(
        &format!("{block}.mlp.linear_fc2.bias"),
        vec![vision_hidden],
    ));
    for norm in ["norm1", "norm2"] {
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{block}.{norm}.weight"),
            vec![vision_hidden],
        ));
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{block}.{norm}.bias"),
            vec![vision_hidden],
        ));
    }
}

/// One vision merger, a deepstack merger or the spatial merger: quantized 4:1 spatial-merge
/// projections from the vision hidden dim to the text hidden dim.
fn push_vision_merger(
    profiles: &mut Vec<QwenImage21TensorProfile>,
    config: &QwenImage21TextEncoderConfig,
    merger_prefix: &str,
) {
    let vision_hidden = config.vision_hidden_size;
    let merger_input = vision_hidden
        * config.vision_spatial_merge_size as usize
        * config.vision_spatial_merge_size as usize;
    let hidden = config.hidden_size;
    push_quantized_linear(
        profiles,
        config,
        &format!("{merger_prefix}.linear_fc1"),
        merger_input,
        merger_input,
    );
    profiles.push(QwenImage21TensorProfile::bf16(
        &format!("{merger_prefix}.linear_fc1.bias"),
        vec![merger_input],
    ));
    push_quantized_linear(
        profiles,
        config,
        &format!("{merger_prefix}.linear_fc2"),
        hidden,
        merger_input,
    );
    profiles.push(QwenImage21TensorProfile::bf16(
        &format!("{merger_prefix}.linear_fc2.bias"),
        vec![hidden],
    ));
    let norm_width = if merger_prefix.starts_with("vision_tower.merger") {
        vision_hidden
    } else {
        merger_input
    };
    profiles.push(QwenImage21TensorProfile::bf16(
        &format!("{merger_prefix}.norm.weight"),
        vec![norm_width],
    ));
    profiles.push(QwenImage21TensorProfile::bf16(
        &format!("{merger_prefix}.norm.bias"),
        vec![norm_width],
    ));
}

#[cfg(test)]
#[path = "text_encoder_tests.rs"]
mod text_encoder_tests;
