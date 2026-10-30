//! Expected checkpoint tensor shapes for the Qwen3.5 vision graph.
//!
//! Linear weights follow `[output_features, input_features]`; model execution
//! transposes them before MLX-C `mlx_addmm`. Conv3d weights already follow MLX's
//! `[output_channels, kernel_depth, kernel_height, kernel_width, input_channels]`
//! layout. These profiles therefore document the boundary between stored model
//! tensors and the MLX operation contracts used by `vision_model.rs`.

use std::collections::HashMap;

#[cfg(test)]
mod tests;

use crate::qwen3_5_core::artifacts::Qwen3_5ArtifactError;
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::quantizations::optiq::{ModuleWeightCodec, OptiQQuantizationProfile};
use crate::{TensorDtype, TensorProfile};

use super::Qwen3_5VisionConfig;

/// Generates config-derived Qwen3.5 vision tower tensor metadata.
///
/// Stored floating tensors retain any MLX-supported model float dtype and are grouped into:
/// - 27 transformer blocks (12 tensors each = 324)
/// - patch embedding (weight + bias = 2)
/// - positional embedding (weight = 1)
/// - merger (norm + linear_fc1 + linear_fc2 = 6)
#[must_use]
pub fn qwen3_5_vision_tensor_profiles(vision_config: &Qwen3_5VisionConfig) -> Vec<TensorProfile> {
    qwen3_5_vision_tensor_profiles_with_quantization(vision_config, None)
        .expect("native vision profiles use the validated vision geometry")
}

/// Generates the vision tensor inventory using per-module storage profiles
/// resolved from the complete Qwen3.5 configuration.
pub fn qwen3_5_vision_tensor_profiles_for_model(
    vision_config: &Qwen3_5VisionConfig,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<Vec<TensorProfile>, Qwen3_5ArtifactError> {
    qwen3_5_vision_tensor_profiles_with_quantization(vision_config, Some(qwen3_5_config))
}

fn qwen3_5_vision_tensor_profiles_with_quantization(
    vision_config: &Qwen3_5VisionConfig,
    qwen3_5_config: Option<&Qwen3_5Config>,
) -> Result<Vec<TensorProfile>, Qwen3_5ArtifactError> {
    let vision_quantization_profiles = qwen3_5_config
        .map(|config| qwen3_5_vision_linear_quantization_profiles(vision_config, config));
    let geometry = VisionGeometry::from_config(vision_config);
    let mut tensor_profiles = Vec::new();
    append_patch_embedding_tensor_profiles(&mut tensor_profiles, &geometry);
    append_positional_embedding_tensor_profile(&mut tensor_profiles, &geometry);
    append_transformer_block_tensor_profiles(
        &mut tensor_profiles,
        vision_quantization_profiles.as_ref(),
        &geometry,
    )?;
    append_merger_tensor_profiles(
        &mut tensor_profiles,
        vision_quantization_profiles.as_ref(),
        &geometry,
    )?;
    Ok(tensor_profiles)
}

/// The validated vision geometry every tensor shape in the inventory derives
/// from, plus the two derived widths (qkv and merger input) shared by several
/// profile sections.
struct VisionGeometry {
    hidden_size: usize,
    intermediate_size: usize,
    depth: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    position_embedding_count: usize,
    out_hidden_size: usize,
    qkv_dimension: usize,
    merger_input_dimension: usize,
}

impl VisionGeometry {
    fn from_config(vision_config: &Qwen3_5VisionConfig) -> Self {
        let hidden_size = vision_config.hidden_size() as usize;
        let spatial_merge_size = vision_config.spatial_merge_size() as usize;
        Self {
            hidden_size,
            intermediate_size: vision_config.intermediate_size() as usize,
            depth: vision_config.depth() as usize,
            patch_size: vision_config.patch_size() as usize,
            temporal_patch_size: vision_config.temporal_patch_size() as usize,
            in_channels: vision_config.in_channels() as usize,
            position_embedding_count: vision_config.position_embedding_count() as usize,
            out_hidden_size: vision_config.out_hidden_size() as usize,
            qkv_dimension: hidden_size * 3,
            merger_input_dimension: hidden_size * spatial_merge_size * spatial_merge_size,
        }
    }
}

// Patch embedding: [hidden_size, temporal_patch_size, patch_size, patch_size, in_channels]
//
// MLX Conv3d consumes ODHWI weights, and the converted artifacts store that
// order. Some published checkpoints keep the upstream PyTorch Conv3d order
// [hidden_size, in_channels, temporal_patch_size, patch_size, patch_size]
// instead — a pure axis permutation of the same values — so the profile
// accepts it and the vision weights loader normalizes it at load.
fn append_patch_embedding_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    geometry: &VisionGeometry,
) {
    tensor_profiles.push(TensorProfile {
        name: "vision_tower.patch_embed.proj.weight".to_owned(),
        dtype: TensorDtype::ModelFloat,
        shape: vec![
            geometry.hidden_size,
            geometry.temporal_patch_size,
            geometry.patch_size,
            geometry.patch_size,
            geometry.in_channels,
        ],
        equivalent_published_shapes: vec![vec![
            geometry.hidden_size,
            geometry.in_channels,
            geometry.temporal_patch_size,
            geometry.patch_size,
            geometry.patch_size,
        ]],
    });
    tensor_profiles.push(tensor_profile(
        "vision_tower.patch_embed.proj.bias".to_owned(),
        TensorDtype::ModelFloat,
        vec![geometry.hidden_size],
    ));
}

// Positional embedding: [position_embedding_count, hidden_size]
fn append_positional_embedding_tensor_profile(
    tensor_profiles: &mut Vec<TensorProfile>,
    geometry: &VisionGeometry,
) {
    tensor_profiles.push(tensor_profile(
        "vision_tower.pos_embed.weight".to_owned(),
        TensorDtype::ModelFloat,
        vec![geometry.position_embedding_count, geometry.hidden_size],
    ));
}

// Each transformer block contributes 12 tensors in a fixed order: attention
// qkv, attention projection, norm1 pair, mlp fc1, mlp fc2, norm2 pair. The
// fixed ordering is what lets artifact loaders and tests index profiles by
// position.
fn append_transformer_block_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    vision_quantization_profiles: Option<&HashMap<String, OptiQQuantizationProfile>>,
    geometry: &VisionGeometry,
) -> Result<(), Qwen3_5ArtifactError> {
    for block_index in 0..geometry.depth {
        let block_prefix = format!("vision_tower.blocks.{block_index}");

        // Attention QKV: [3*hidden_size, hidden_size] + bias [3*hidden_size]
        append_linear_module_tensor_profiles(
            tensor_profiles,
            &format!("{block_prefix}.attn.qkv"),
            geometry.qkv_dimension,
            geometry.hidden_size,
            vision_module_quantization_profile(
                vision_quantization_profiles,
                &format!("{block_prefix}.attn.qkv"),
            ),
        )?;

        // Attention projection: [hidden_size, hidden_size] + bias [hidden_size]
        append_linear_module_tensor_profiles(
            tensor_profiles,
            &format!("{block_prefix}.attn.proj"),
            geometry.hidden_size,
            geometry.hidden_size,
            vision_module_quantization_profile(
                vision_quantization_profiles,
                &format!("{block_prefix}.attn.proj"),
            ),
        )?;

        // LayerNorm 1: [hidden_size] x2 (weight + bias)
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.norm1.weight"),
            TensorDtype::ModelFloat,
            vec![geometry.hidden_size],
        ));
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.norm1.bias"),
            TensorDtype::ModelFloat,
            vec![geometry.hidden_size],
        ));

        // MLP fc1: [intermediate_size, hidden_size] + bias [intermediate_size]
        append_linear_module_tensor_profiles(
            tensor_profiles,
            &format!("{block_prefix}.mlp.linear_fc1"),
            geometry.intermediate_size,
            geometry.hidden_size,
            vision_module_quantization_profile(
                vision_quantization_profiles,
                &format!("{block_prefix}.mlp.linear_fc1"),
            ),
        )?;

        // MLP fc2: [hidden_size, intermediate_size] + bias [hidden_size]
        append_linear_module_tensor_profiles(
            tensor_profiles,
            &format!("{block_prefix}.mlp.linear_fc2"),
            geometry.hidden_size,
            geometry.intermediate_size,
            vision_module_quantization_profile(
                vision_quantization_profiles,
                &format!("{block_prefix}.mlp.linear_fc2"),
            ),
        )?;

        // LayerNorm 2: [hidden_size] x2 (weight + bias)
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.norm2.weight"),
            TensorDtype::ModelFloat,
            vec![geometry.hidden_size],
        ));
        tensor_profiles.push(tensor_profile(
            format!("{block_prefix}.norm2.bias"),
            TensorDtype::ModelFloat,
            vec![geometry.hidden_size],
        ));
    }
    Ok(())
}

// The 2x2 spatial merge concatenates four hidden rows, so both fc1 axes are
// merger_input_dimension=4*hidden_size. fc2 projects that vector to the text
// model's out_hidden_size.
fn append_merger_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    vision_quantization_profiles: Option<&HashMap<String, OptiQQuantizationProfile>>,
    geometry: &VisionGeometry,
) -> Result<(), Qwen3_5ArtifactError> {
    tensor_profiles.push(tensor_profile(
        "vision_tower.merger.norm.weight".to_owned(),
        TensorDtype::ModelFloat,
        vec![geometry.hidden_size],
    ));
    tensor_profiles.push(tensor_profile(
        "vision_tower.merger.norm.bias".to_owned(),
        TensorDtype::ModelFloat,
        vec![geometry.hidden_size],
    ));
    append_linear_module_tensor_profiles(
        tensor_profiles,
        "vision_tower.merger.linear_fc1",
        geometry.merger_input_dimension,
        geometry.merger_input_dimension,
        vision_module_quantization_profile(
            vision_quantization_profiles,
            "vision_tower.merger.linear_fc1",
        ),
    )?;
    append_linear_module_tensor_profiles(
        tensor_profiles,
        "vision_tower.merger.linear_fc2",
        geometry.out_hidden_size,
        geometry.merger_input_dimension,
        vision_module_quantization_profile(
            vision_quantization_profiles,
            "vision_tower.merger.linear_fc2",
        ),
    )?;
    Ok(())
}

fn append_linear_module_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    module_name: &str,
    output_width: usize,
    input_width: usize,
    quantization_profile: Option<OptiQQuantizationProfile>,
) -> Result<(), Qwen3_5ArtifactError> {
    if let Some(module_quantization_profile) = quantization_profile {
        if let ModuleWeightCodec::Jangtq2CodebookQuantized { .. } =
            module_quantization_profile.weight_codec
        {
            return Err(Qwen3_5ArtifactError::InvalidVisionQuantizationProfile {
                module_name: module_name.to_owned(),
                description: "vision execution does not implement the expert codebook codec"
                    .to_owned(),
            });
        }
        if !module_quantization_profile.is_unquantized() {
            append_affine_module_tensor_profiles(
                tensor_profiles,
                module_name,
                output_width,
                input_width,
                module_quantization_profile.bits,
                module_quantization_profile.group_size,
            )?;
            tensor_profiles.push(tensor_profile(
                format!("{module_name}.bias"),
                TensorDtype::ModelFloat,
                vec![output_width],
            ));
            return Ok(());
        }
    }

    tensor_profiles.push(tensor_profile(
        format!("{module_name}.weight"),
        TensorDtype::ModelFloat,
        vec![output_width, input_width],
    ));
    tensor_profiles.push(tensor_profile(
        format!("{module_name}.bias"),
        TensorDtype::ModelFloat,
        vec![output_width],
    ));
    Ok(())
}

pub(crate) fn qwen3_5_vision_linear_quantization_profiles(
    vision_config: &Qwen3_5VisionConfig,
    qwen3_5_config: &Qwen3_5Config,
) -> HashMap<String, OptiQQuantizationProfile> {
    let mut module_quantization_profiles =
        HashMap::with_capacity(vision_config.depth() as usize * 4 + 2);
    for block_index in 0..vision_config.depth() {
        let block_prefix = format!("vision_tower.blocks.{block_index}");
        for module_suffix in ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"] {
            let module_name = format!("{block_prefix}.{module_suffix}");
            module_quantization_profiles.insert(
                module_name.clone(),
                qwen3_5_config.quantization_profile_for_module(&module_name),
            );
        }
    }
    for module_name in [
        "vision_tower.merger.linear_fc1",
        "vision_tower.merger.linear_fc2",
    ] {
        module_quantization_profiles.insert(
            module_name.to_owned(),
            qwen3_5_config.quantization_profile_for_module(module_name),
        );
    }
    module_quantization_profiles
}

fn vision_module_quantization_profile(
    module_quantization_profiles: Option<&HashMap<String, OptiQQuantizationProfile>>,
    module_name: &str,
) -> Option<OptiQQuantizationProfile> {
    module_quantization_profiles
        .and_then(|profile_by_module_name| profile_by_module_name.get(module_name))
        .copied()
}

fn append_affine_module_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    module_name: &str,
    output_width: usize,
    input_width: usize,
    quantization_bits: u32,
    quantization_group_size: u32,
) -> Result<(), Qwen3_5ArtifactError> {
    let invalid_profile =
        |description: String| Qwen3_5ArtifactError::InvalidVisionQuantizationProfile {
            module_name: module_name.to_owned(),
            description,
        };
    let quantization_bits = usize::try_from(quantization_bits)
        .map_err(|_| invalid_profile("affine bit width exceeds the host shape range".to_owned()))?;
    let quantization_group_size = usize::try_from(quantization_group_size).map_err(|_| {
        invalid_profile("affine group size exceeds the host shape range".to_owned())
    })?;
    if quantization_bits == 0 || quantization_group_size == 0 {
        return Err(invalid_profile(
            "affine bit width and group size must be positive".to_owned(),
        ));
    }
    if !input_width.is_multiple_of(quantization_group_size) {
        return Err(invalid_profile(format!(
            "input width {input_width} is not divisible by group size {quantization_group_size}"
        )));
    }
    let packed_width = input_width
        .checked_mul(quantization_bits)
        .filter(|packed_bit_count| packed_bit_count % 32 == 0)
        .map(|packed_bit_count| packed_bit_count / 32)
        .ok_or_else(|| {
            invalid_profile(format!(
                "input width {input_width} and bit width {quantization_bits} do not form packed uint32 words"
            ))
        })?;
    let companion_width = input_width / quantization_group_size;
    tensor_profiles.push(tensor_profile(
        format!("{module_name}.weight"),
        TensorDtype::UInt32,
        vec![output_width, packed_width],
    ));
    for companion_name in ["scales", "biases"] {
        tensor_profiles.push(tensor_profile(
            format!("{module_name}.{companion_name}"),
            TensorDtype::AffineQuantizationFloat,
            vec![output_width, companion_width],
        ));
    }
    Ok(())
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
        equivalent_published_shapes: Vec::new(),
    }
}
