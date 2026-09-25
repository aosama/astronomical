//! The Qwen3-VL text encoder's expected tensors: language model plus the validated vision
//! tower. The T2I path executes only the language model; the tower is still part of the
//! reviewed file, so the profile covers both.

use super::{QwenImage21TensorProfile, quantized_linear};
use crate::qwen_image_21::configuration::QwenImage21TextEncoderConfig;

/// Expected physical tensors of the Qwen3-VL text encoder (1438 tensors), language model and
/// vision tower. The T2I path executes only the language model; the vision tower is validated so
/// the artifact contract covers the full file.
#[must_use]
pub fn text_encoder_tensor_profiles(
    config: &QwenImage21TextEncoderConfig,
) -> Vec<QwenImage21TensorProfile> {
    let hidden = config.hidden_size;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let bits = config.quantization_bits;
    let group = config.quantization_group_size;
    let vision_hidden = config.vision_hidden_size;
    let vision_qkv = 3 * vision_hidden;
    let merger_input = vision_hidden
        * config.vision_spatial_merge_size as usize
        * config.vision_spatial_merge_size as usize;
    let mut profiles = Vec::new();

    quantized_linear(
        &mut profiles,
        "language_model.lm_head",
        config.vocab_size,
        hidden,
        bits,
        group,
    );
    quantized_linear(
        &mut profiles,
        "language_model.model.embed_tokens",
        config.vocab_size,
        hidden,
        bits,
        group,
    );

    for layer_index in 0..config.num_hidden_layers {
        let layer = format!("language_model.model.layers.{layer_index}");
        quantized_linear(
            &mut profiles,
            &format!("{layer}.self_attn.q_proj"),
            hidden,
            hidden,
            bits,
            group,
        );
        quantized_linear(
            &mut profiles,
            &format!("{layer}.self_attn.k_proj"),
            kv_dim,
            hidden,
            bits,
            group,
        );
        quantized_linear(
            &mut profiles,
            &format!("{layer}.self_attn.v_proj"),
            kv_dim,
            hidden,
            bits,
            group,
        );
        quantized_linear(
            &mut profiles,
            &format!("{layer}.self_attn.o_proj"),
            hidden,
            hidden,
            bits,
            group,
        );
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{layer}.self_attn.q_norm.weight"),
            vec![config.head_dim],
        ));
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{layer}.self_attn.k_norm.weight"),
            vec![config.head_dim],
        ));
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{layer}.input_layernorm.weight"),
            vec![hidden],
        ));
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{layer}.post_attention_layernorm.weight"),
            vec![hidden],
        ));
        for projection in ["mlp.gate_proj", "mlp.up_proj"] {
            quantized_linear(
                &mut profiles,
                &format!("{layer}.{projection}"),
                config.intermediate_size,
                hidden,
                bits,
                group,
            );
        }
        quantized_linear(
            &mut profiles,
            &format!("{layer}.mlp.down_proj"),
            hidden,
            config.intermediate_size,
            bits,
            group,
        );
    }
    profiles.push(QwenImage21TensorProfile::bf16(
        "language_model.model.norm.weight",
        vec![hidden],
    ));

    // Vision tower. `patch_embed.proj` is a full-precision Conv3d kernel
    // [out, in, t, h, w]; `pos_embed` is a quantized embedding of the position count.
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
    quantized_linear(
        &mut profiles,
        "vision_tower.pos_embed",
        config.vision_num_position_embeddings,
        vision_hidden,
        bits,
        group,
    );

    for block_index in 0..config.vision_depth {
        let block = format!("vision_tower.blocks.{block_index}");
        // Quantized linears in the tower keep their original linear bias (`.bias`) in addition
        // to the per-group dequantization bias (`.biases`).
        quantized_linear(
            &mut profiles,
            &format!("{block}.attn.qkv"),
            vision_qkv,
            vision_hidden,
            bits,
            group,
        );
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{block}.attn.qkv.bias"),
            vec![vision_qkv],
        ));
        quantized_linear(
            &mut profiles,
            &format!("{block}.attn.proj"),
            vision_hidden,
            vision_hidden,
            bits,
            group,
        );
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{block}.attn.proj.bias"),
            vec![vision_hidden],
        ));
        quantized_linear(
            &mut profiles,
            &format!("{block}.mlp.linear_fc1"),
            config.vision_intermediate_size,
            vision_hidden,
            bits,
            group,
        );
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{block}.mlp.linear_fc1.bias"),
            vec![config.vision_intermediate_size],
        ));
        // fc2 stays full-precision in the reviewed artifact.
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{block}.mlp.linear_fc2.weight"),
            vec![vision_hidden, config.vision_intermediate_size],
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

    // Deepstack mergers (three) and the single spatial merger: quantized 4:1 spatial-merge
    // projections from the vision hidden dim to the text hidden dim.
    for merger_prefix in [
        "vision_tower.deepstack_merger_list.0",
        "vision_tower.deepstack_merger_list.1",
        "vision_tower.deepstack_merger_list.2",
        "vision_tower.merger",
    ] {
        quantized_linear(
            &mut profiles,
            &format!("{merger_prefix}.linear_fc1"),
            merger_input,
            merger_input,
            bits,
            group,
        );
        profiles.push(QwenImage21TensorProfile::bf16(
            &format!("{merger_prefix}.linear_fc1.bias"),
            vec![merger_input],
        ));
        quantized_linear(
            &mut profiles,
            &format!("{merger_prefix}.linear_fc2"),
            hidden,
            merger_input,
            bits,
            group,
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

    profiles
}
