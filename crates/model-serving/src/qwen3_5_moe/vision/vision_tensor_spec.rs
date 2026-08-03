//! Expected checkpoint tensor shapes for the Qwen3.5 vision graph.
//!
//! Linear weights follow `[output_features, input_features]`; model execution
//! transposes them before MLX-C `mlx_addmm`. Conv3d weights already follow MLX's
//! `[output_channels, kernel_depth, kernel_height, kernel_width, input_channels]`
//! layout. These profiles therefore document the boundary between stored model
//! tensors and the MLX operation contracts used by `vision_model.rs`.

use crate::{TensorDtype, TensorProfile};

use super::Qwen3_5MoEVisionConfig;

/// Generates the vision tower tensor metadata for the pinned Qwen3.5-MoE artifact.
///
/// All 333 tensors are BF16 and grouped into:
/// - 27 transformer blocks (12 tensors each = 324)
/// - patch embedding (weight + bias = 2)
/// - positional embedding (weight = 1)
/// - merger (norm + linear_fc1 + linear_fc2 = 6)
#[must_use]
pub fn qwen3_5_moe_vision_tensor_profiles(
    vision_config: &Qwen3_5MoEVisionConfig,
) -> Vec<TensorProfile> {
    let hidden_size = vision_config.hidden_size() as usize;
    let intermediate_size = vision_config.intermediate_size() as usize;
    let depth = vision_config.depth() as usize;
    let patch_size = vision_config.patch_size() as usize;
    let temporal_patch_size = vision_config.temporal_patch_size() as usize;
    let in_channels = vision_config.in_channels() as usize;
    let spatial_merge_size = vision_config.spatial_merge_size() as usize;
    let position_embedding_count = vision_config.position_embedding_count() as usize;
    let out_hidden_size = vision_config.out_hidden_size() as usize;

    let qkv_dimension = hidden_size * 3;
    let merger_input_dimension = hidden_size * spatial_merge_size * spatial_merge_size;

    let mut tensor_profiles = Vec::new();

    // Patch embedding: [hidden_size, temporal_patch_size, patch_size, patch_size, in_channels]
    tensor_profiles.push(tensor_profile(
        "vision_tower.patch_embed.proj.weight".to_owned(),
        TensorDtype::BFloat16,
        vec![
            hidden_size,
            temporal_patch_size,
            patch_size,
            patch_size,
            in_channels,
        ],
    ));
    tensor_profiles.push(tensor_profile(
        "vision_tower.patch_embed.proj.bias".to_owned(),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));

    // Positional embedding: [position_embedding_count, hidden_size]
    tensor_profiles.push(tensor_profile(
        "vision_tower.pos_embed.weight".to_owned(),
        TensorDtype::BFloat16,
        vec![position_embedding_count, hidden_size],
    ));

    // Transformer blocks
    for block_index in 0..depth {
        let block_prefix = format!("vision_tower.blocks.{block_index}");

        // Attention QKV: [3*hidden_size, hidden_size] + bias [3*hidden_size]
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.attn.qkv.weight"),
            TensorDtype::BFloat16,
            vec![qkv_dimension, hidden_size],
        ));
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.attn.qkv.bias"),
            TensorDtype::BFloat16,
            vec![qkv_dimension],
        ));

        // Attention projection: [hidden_size, hidden_size] + bias [hidden_size]
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.attn.proj.weight"),
            TensorDtype::BFloat16,
            vec![hidden_size, hidden_size],
        ));
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.attn.proj.bias"),
            TensorDtype::BFloat16,
            vec![hidden_size],
        ));

        // LayerNorm 1: [hidden_size] x2 (weight + bias)
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.norm1.weight"),
            TensorDtype::BFloat16,
            vec![hidden_size],
        ));
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.norm1.bias"),
            TensorDtype::BFloat16,
            vec![hidden_size],
        ));

        // MLP fc1: [intermediate_size, hidden_size] + bias [intermediate_size]
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.mlp.linear_fc1.weight"),
            TensorDtype::BFloat16,
            vec![intermediate_size, hidden_size],
        ));
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.mlp.linear_fc1.bias"),
            TensorDtype::BFloat16,
            vec![intermediate_size],
        ));

        // MLP fc2: [hidden_size, intermediate_size] + bias [hidden_size]
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.mlp.linear_fc2.weight"),
            TensorDtype::BFloat16,
            vec![hidden_size, intermediate_size],
        ));
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.mlp.linear_fc2.bias"),
            TensorDtype::BFloat16,
            vec![hidden_size],
        ));

        // LayerNorm 2: [hidden_size] x2 (weight + bias)
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.norm2.weight"),
            TensorDtype::BFloat16,
            vec![hidden_size],
        ));
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.norm2.bias"),
            TensorDtype::BFloat16,
            vec![hidden_size],
        ));
    }

    // The 2x2 spatial merge concatenates four hidden rows, so both fc1 axes are
    // merger_input_dimension=4*hidden_size. fc2 projects that vector to the text
    // model's out_hidden_size.
    tensor_profiles.push(tensor_profile(
        "vision_tower.merger.norm.weight".to_owned(),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    tensor_profiles.push(tensor_profile(
        "vision_tower.merger.norm.bias".to_owned(),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    tensor_profiles.push(tensor_profile(
        "vision_tower.merger.linear_fc1.weight".to_owned(),
        TensorDtype::BFloat16,
        vec![merger_input_dimension, merger_input_dimension],
    ));
    tensor_profiles.push(tensor_profile(
        "vision_tower.merger.linear_fc1.bias".to_owned(),
        TensorDtype::BFloat16,
        vec![merger_input_dimension],
    ));
    tensor_profiles.push(tensor_profile(
        "vision_tower.merger.linear_fc2.weight".to_owned(),
        TensorDtype::BFloat16,
        vec![out_hidden_size, merger_input_dimension],
    ));
    tensor_profiles.push(tensor_profile(
        "vision_tower.merger.linear_fc2.bias".to_owned(),
        TensorDtype::BFloat16,
        vec![out_hidden_size],
    ));

    tensor_profiles
}

fn tensor_profile(
    tensor_name: String,
    tensor_dtype: TensorDtype,
    tensor_shape: Vec<usize>,
) -> TensorProfile {
    TensorProfile {
        name: tensor_name,
        dtype: tensor_dtype,
        shape: tensor_shape,
    }
}
