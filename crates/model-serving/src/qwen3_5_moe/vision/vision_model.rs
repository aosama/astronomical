//! Direct Qwen3.5 vision execution for the pinned Qwen3.5-MoE artifact.
//!
//! Source lineage: Rust translation of MLX-VLM's Qwen3-VL vision implementation
//! (MIT License). MLX-VLM reuses that vision implementation for Qwen3.5. Keep
//! the operation graph, tensor layouts, dtypes, and addition order aligned with
//! that source: BF16 makes algebraically equivalent rewrites observably different.
//! See third-party license notices for complete attribution.
//!
//! This module does not implement convolution, normalization, matrix
//! multiplication, trigonometry, or attention kernels. Those operations end at
//! MLX-C entry points in `mlx-c/mlx/c/ops.h` or `mlx-c/mlx/c/fast.h`; this code
//! owns only model-specific graph assembly and image-boundary segmentation.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};
use std::cell::Cell;

use super::vision_attention::qwen3_5_moe_vision_self_attention;
use super::vision_rotary_embedding::Qwen3_5MoEVisionRotaryEmbedding;
use super::{
    Qwen3_5MoEExecutionError, Qwen3_5MoEProcessedImage, Qwen3_5MoEVisionConfig,
    Qwen3_5MoEVisionInputPlan, Qwen3_5MoEVisionWeights, ValidatedQwen3_5MoEArtifact,
};

const LAYER_NORM_EPSILON: f32 = 1e-6;

/// Resident Qwen3.5-MoE vision configuration and sidecar weights.
#[derive(Debug)]
pub struct Qwen3_5MoEVisionModel {
    config: Qwen3_5MoEVisionConfig,
    weights: Qwen3_5MoEVisionWeights,
    weights_have_been_used: Cell<bool>,
}

impl Qwen3_5MoEVisionModel {
    #[must_use]
    pub(in crate::qwen3_5_moe) fn resident_payload_bytes(&self) -> u64 {
        if self.weights_have_been_used.get() {
            self.weights.total_payload_bytes()
        } else {
            0
        }
    }
    /// Loads vision weights from a separate sidecar file (oQ4 model).
    /// Returns None when visual weights are embedded or absent.
    pub fn load_from_sidecar(
        runtime: &MlxRuntime,
        validated_artifact: &mut ValidatedQwen3_5MoEArtifact,
    ) -> Result<Option<Self>, Qwen3_5MoEExecutionError> {
        let config = validated_artifact.vision_config().cloned().ok_or(
            Qwen3_5MoEExecutionError::InvalidInput {
                description: "validated visual sidecar has no vision configuration",
            },
        )?;
        let Some(weights) =
            Qwen3_5MoEVisionWeights::load_from_sidecar(runtime, validated_artifact)?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            config,
            weights,
            weights_have_been_used: Cell::new(false),
        }))
    }

    /// Loads vision weights from model shards with embedded vision tensors.
    pub fn load_from_model_shards(
        vision_config: &Qwen3_5MoEVisionConfig,
        model_shards: &[astronomical_runtime_integration::MlxSafetensors],
        vision_tensor_name_to_shard_index: &std::collections::HashMap<String, usize>,
    ) -> Result<Self, Qwen3_5MoEExecutionError> {
        let weights = Qwen3_5MoEVisionWeights::load_from_model_shards(
            vision_config,
            model_shards,
            vision_tensor_name_to_shard_index,
        )?;
        Ok(Self {
            config: vision_config.clone(),
            weights,
            weights_have_been_used: Cell::new(false),
        })
    }

    /// Projects preprocessed images into text-model-width visual embeddings.
    ///
    /// Shape flow for the pinned model:
    /// `[patches, 1536] -> [patches, 1152] -> 27 transformer blocks ->
    /// [merged_patches, 2048]`. Four neighboring `1152`-wide patch rows become
    /// one merger row because `spatial_merge_size == 2`.
    pub fn forward(
        &self,
        runtime: &MlxRuntime,
        processed_images: &[Qwen3_5MoEProcessedImage],
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let image_grids = processed_images
            .iter()
            .map(|processed_image| processed_image.image_grid)
            .collect::<Vec<_>>();
        let vision_input_plan = Qwen3_5MoEVisionInputPlan::new(&image_grids, &self.config)
            .map_err(
                |_vision_input_plan_error| Qwen3_5MoEExecutionError::InvalidInput {
                    description: "processed image grids are invalid for Qwen3.5-MoE vision execution",
                },
            )?;
        let pixel_values =
            self.upload_pixel_values(runtime, processed_images, &vision_input_plan)?;
        // Patch rows already arrive in spatial-merge block order. Every later
        // reshape depends on preserving that order; sorting patches into ordinary
        // row-major image order would silently combine unrelated patches.
        let mut hidden_states =
            self.patch_embed(runtime, &pixel_values, vision_input_plan.patch_count())?;
        let interpolated_position_embeddings =
            self.interpolate_position_embeddings(runtime, &vision_input_plan)?;
        hidden_states = runtime.add(&hidden_states, &interpolated_position_embeddings)?;
        let (rotary_cosines, rotary_sines) =
            Qwen3_5MoEVisionRotaryEmbedding::build(runtime, &vision_input_plan, &self.config)?;

        for vision_block_index in 0..self.config.depth() {
            hidden_states = self.forward_vision_block(
                runtime,
                &hidden_states,
                vision_block_index,
                vision_input_plan.attention_sequence_boundaries(),
                &rotary_cosines,
                &rotary_sines,
            )?;
        }
        let visual_embeddings = self.merge_patches(
            runtime,
            &hidden_states,
            vision_input_plan.merged_patch_count(),
        )?;
        self.weights_have_been_used.set(true);
        Ok(visual_embeddings)
    }
    fn upload_pixel_values(
        &self,
        runtime: &MlxRuntime,
        processed_images: &[Qwen3_5MoEProcessedImage],
        vision_input_plan: &Qwen3_5MoEVisionInputPlan,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let pixel_values_column_count = self.pixel_values_column_count()?;
        let mut image_pixel_value_arrays = Vec::with_capacity(processed_images.len());
        for processed_image in processed_images {
            if processed_image.pixel_values_column_count != pixel_values_column_count
                || processed_image.pixel_values.len()
                    != processed_image.pixel_values_row_count * pixel_values_column_count
            {
                return Err(Qwen3_5MoEExecutionError::InvalidInput {
                    description: "processed image pixel values have an invalid shape",
                });
            }
            image_pixel_value_arrays.push(runtime.array_from_f32(
                &processed_image.pixel_values,
                &[
                    usize_to_i32(processed_image.pixel_values_row_count)?,
                    usize_to_i32(pixel_values_column_count)?,
                ],
            )?);
        }
        let uploaded_patch_count = image_pixel_value_arrays
            .iter()
            .try_fold(0_usize, |uploaded_patch_count, pixel_value_array| {
                uploaded_patch_count.checked_add(pixel_value_array.shape()[0] as usize)
            })
            .ok_or(Qwen3_5MoEExecutionError::InvalidInput {
                description: "processed image patch count overflowed",
            })?;
        if uploaded_patch_count != vision_input_plan.patch_count() {
            return Err(Qwen3_5MoEExecutionError::InvalidInput {
                description: "processed image pixel rows do not match the image grids",
            });
        }
        let image_pixel_value_references = image_pixel_value_arrays.iter().collect::<Vec<_>>();
        Ok(runtime.concatenate_axis(&image_pixel_value_references, 0)?)
    }
    fn patch_embed(
        &self,
        runtime: &MlxRuntime,
        pixel_values: &MlxArray,
        patch_count: usize,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let patch_count = usize_to_i32(patch_count)?;
        let in_channels = u32_to_i32(self.config.in_channels())?;
        let temporal_patch_size = u32_to_i32(self.config.temporal_patch_size())?;
        let patch_size = u32_to_i32(self.config.patch_size())?;
        let hidden_size = u32_to_i32(self.config.hidden_size())?;
        let patch_weight = self
            .weights
            .tensor("vision_tower.patch_embed.proj.weight")?;
        let typed_pixel_values = runtime.astype(pixel_values, patch_weight.dtype())?;
        // CPU preprocessing flattens each patch as C,T,H,W. MLX Conv3d requires
        // NDHWC input and ODHWI weights, hence this reshape followed by moving C
        // to the final axis. See MLX `conv3d` docs and `mlx-c/mlx/c/ops.h::mlx_conv3d`.
        let channels_first_patches = runtime.reshape(
            &typed_pixel_values,
            &[
                patch_count,
                in_channels,
                temporal_patch_size,
                patch_size,
                patch_size,
            ],
        )?;
        let channels_last_patches =
            runtime.transpose_axes(&channels_first_patches, &[0, 2, 3, 4, 1])?;
        // Kernel and stride equal the complete patch volume. Therefore every
        // input patch produces exactly one output cell; this Conv3d is the
        // checkpoint's learned patch projection, not a sliding image convolution.
        let projected_patches = runtime.conv3d(
            &channels_last_patches,
            patch_weight,
            [temporal_patch_size, patch_size, patch_size],
            [0, 0, 0],
            [1, 1, 1],
            1,
        )?;
        let patch_bias = self.weights.tensor("vision_tower.patch_embed.proj.bias")?;
        let biased_projected_patches = runtime.add(&projected_patches, patch_bias)?;
        Ok(runtime.reshape(&biased_projected_patches, &[patch_count, hidden_size])?)
    }
    fn interpolate_position_embeddings(
        &self,
        runtime: &MlxRuntime,
        vision_input_plan: &Qwen3_5MoEVisionInputPlan,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let position_embedding_weight = self.weights.tensor("vision_tower.pos_embed.weight")?;
        let patch_count = usize_to_i32(vision_input_plan.patch_count())?;
        // The learned position table is square, while runtime images are usually
        // rectangular. The CPU plan supplies four table indices and bilinear
        // weights per patch, equivalent to the translated Qwen3-VL interpolation
        // path. `take_axis` reaches `mlx_take_axis`; multiplication
        // and addition reach `mlx_multiply` and `mlx_add` in MLX-C.
        let mut weighted_corner_embeddings = Vec::with_capacity(4);
        for corner_index in 0..4 {
            let corner_indices = runtime.array_from_u32(
                &vision_input_plan.bilinear_corner_indices()[corner_index],
                &[patch_count],
            )?;
            let corner_embeddings =
                runtime.take_axis(position_embedding_weight, &corner_indices, 0)?;
            let corner_weights_f32 = runtime.array_from_f32(
                &vision_input_plan.bilinear_corner_weights()[corner_index],
                &[patch_count, 1],
            )?;
            let corner_weights =
                runtime.astype(&corner_weights_f32, position_embedding_weight.dtype())?;
            weighted_corner_embeddings.push(runtime.multiply(&corner_embeddings, &corner_weights)?);
        }
        // Deliberately preserve upstream left-associative evaluation:
        // `((corner0 + corner1) + corner2) + corner3`. Pairwise reduction changes
        // BF16 rounding and broke one-step parity even though the real-number
        // expression is identical.
        let first_two_corner_interpolation = runtime.add(
            &weighted_corner_embeddings[0],
            &weighted_corner_embeddings[1],
        )?;
        let first_three_corner_interpolation = runtime.add(
            &first_two_corner_interpolation,
            &weighted_corner_embeddings[2],
        )?;
        Ok(runtime.add(
            &first_three_corner_interpolation,
            &weighted_corner_embeddings[3],
        )?)
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_vision_block(
        &self,
        runtime: &MlxRuntime,
        hidden_states: &MlxArray,
        vision_block_index: u32,
        attention_sequence_boundaries: &[u32],
        rotary_cosines: &MlxArray,
        rotary_sines: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let vision_block_prefix = format!("vision_tower.blocks.{vision_block_index}");
        let norm1_output =
            self.layer_norm(runtime, hidden_states, &vision_block_prefix, "norm1")?;
        let attention_output = qwen3_5_moe_vision_self_attention(
            runtime,
            &self.config,
            &self.weights,
            &norm1_output,
            &vision_block_prefix,
            attention_sequence_boundaries,
            rotary_cosines,
            rotary_sines,
        )?;
        // Pre-normalization transformer block:
        // x = x + Attention(LayerNorm(x)); x = x + MLP(LayerNorm(x)).
        let post_attention_hidden_states = runtime.add(hidden_states, &attention_output)?;
        let norm2_output = self.layer_norm(
            runtime,
            &post_attention_hidden_states,
            &vision_block_prefix,
            "norm2",
        )?;
        let first_mlp_projection = self.linear(
            runtime,
            &norm2_output,
            &format!("{vision_block_prefix}.mlp.linear_fc1"),
        )?;
        // Vision blocks use PyTorch's tanh GELU approximation, unlike the final
        // patch merger which uses exact erf-based GELU. This distinction is part
        // of the Qwen3-VL checkpoint contract and affects output parity.
        let activated_mlp_projection = runtime.gelu_tanh(&first_mlp_projection)?;
        let second_mlp_projection = self.linear(
            runtime,
            &activated_mlp_projection,
            &format!("{vision_block_prefix}.mlp.linear_fc2"),
        )?;
        Ok(runtime.add(&post_attention_hidden_states, &second_mlp_projection)?)
    }
    fn layer_norm(
        &self,
        runtime: &MlxRuntime,
        hidden_states: &MlxArray,
        vision_block_prefix: &str,
        normalization_name: &str,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let normalization_prefix = format!("{vision_block_prefix}.{normalization_name}");
        let normalization_weight = self
            .weights
            .tensor(&format!("{normalization_prefix}.weight"))?;
        let normalization_bias = self
            .weights
            .tensor(&format!("{normalization_prefix}.bias"))?;
        Ok(runtime.layer_norm(
            hidden_states,
            normalization_weight,
            normalization_bias,
            LAYER_NORM_EPSILON,
        )?)
    }
    fn merge_patches(
        &self,
        runtime: &MlxRuntime,
        hidden_states: &MlxArray,
        merged_patch_count: usize,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let merger_normalization_weight = self.weights.tensor("vision_tower.merger.norm.weight")?;
        let merger_normalization_bias = self.weights.tensor("vision_tower.merger.norm.bias")?;
        let normalized_hidden_states = runtime.layer_norm(
            hidden_states,
            merger_normalization_weight,
            merger_normalization_bias,
            LAYER_NORM_EPSILON,
        )?;
        // Patch rows were laid out as [merged row, merged column, intra row,
        // intra column]. A reshape can therefore concatenate each 2x2 block into
        // one 4*hidden_size vector without a gather or transpose.
        let spatial_merge_area = self.config.spatial_merge_size().pow(2);
        let merger_input_dimension = self
            .config
            .hidden_size()
            .checked_mul(spatial_merge_area)
            .ok_or(Qwen3_5MoEExecutionError::InvalidInput {
                description: "vision merger input dimension overflowed",
            })?;
        let merged_hidden_states = runtime.reshape(
            &normalized_hidden_states,
            &[
                usize_to_i32(merged_patch_count)?,
                u32_to_i32(merger_input_dimension)?,
            ],
        )?;
        let first_merger_projection = self.linear(
            runtime,
            &merged_hidden_states,
            "vision_tower.merger.linear_fc1",
        )?;
        // The Qwen3-VL patch merger uses exact GELU, not the transformer's tanh
        // approximation.
        let activated_merger_projection = runtime.gelu(&first_merger_projection)?;
        self.linear(
            runtime,
            &activated_merger_projection,
            "vision_tower.merger.linear_fc2",
        )
    }

    fn linear(
        &self,
        runtime: &MlxRuntime,
        input_states: &MlxArray,
        linear_prefix: &str,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let linear_weight = self.weights.tensor(&format!("{linear_prefix}.weight"))?;
        let transposed_linear_weight = runtime.transpose_axes(linear_weight, &[1, 0])?;
        let linear_bias = self.weights.tensor(&format!("{linear_prefix}.bias"))?;
        // Use MLX's fused addmm (`mlx_addmm`) rather than spelling this as
        // matmul+add. The fused accumulation/rounding path is part of numerical
        // parity for BF16 checkpoint weights.
        Ok(runtime.addmm(
            linear_bias,
            input_states,
            &transposed_linear_weight,
            1.0,
            1.0,
        )?)
    }

    fn pixel_values_column_count(&self) -> Result<usize, Qwen3_5MoEExecutionError> {
        [
            self.config.in_channels(),
            self.config.temporal_patch_size(),
            self.config.patch_size(),
            self.config.patch_size(),
        ]
        .into_iter()
        .try_fold(1_usize, |element_count, dimension| {
            element_count.checked_mul(dimension as usize)
        })
        .ok_or(Qwen3_5MoEExecutionError::InvalidInput {
            description: "vision patch element count overflowed",
        })
    }
}

fn usize_to_i32(dimension_size: usize) -> Result<i32, Qwen3_5MoEExecutionError> {
    i32::try_from(dimension_size).map_err(|_conversion_error| {
        Qwen3_5MoEExecutionError::InvalidInput {
            description: "vision dimension exceeds the MLX integer range",
        }
    })
}

fn u32_to_i32(dimension_size: u32) -> Result<i32, Qwen3_5MoEExecutionError> {
    i32::try_from(dimension_size).map_err(|_conversion_error| {
        Qwen3_5MoEExecutionError::InvalidInput {
            description: "vision dimension exceeds the MLX integer range",
        }
    })
}
