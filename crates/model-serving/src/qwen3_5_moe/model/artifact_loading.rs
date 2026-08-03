//! Artifact binding and startup-only MLX resource construction for Qwen3.5-MoE.

use std::collections::HashMap;
use std::path::Path;

use astronomical_runtime_integration::{
    MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxDtype, MlxRuntime, MlxSafetensors,
};

use crate::qwen3_5_moe::Qwen3_5MoEMtpArtifactCapability;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::expert_paging::expert_pager::ExpertPager;
use super::model::Qwen3_5MoEModel;
use super::mtp::Qwen3_5MoEMtpWeights;
use super::{
    ExpertWeightMemoryCache, Qwen3_5MoEConfig, Qwen3_5MoEExecutionError, Qwen3_5MoEShardIndex,
    Qwen3_5MoEVisionModel, Qwen3_5MoEWeights, ValidatedQwen3_5MoEArtifact,
};

impl Qwen3_5MoEModel {
    pub(in crate::qwen3_5_moe) fn prewarm_complete_expert_layers_with_performance_attribution(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5MoEExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(());
        };
        expert_pager.prewarm_complete_layers_with_performance_attribution(
            &self.runtime,
            &self.expert_weight_memory_cache,
            performance_attribution,
        )?;
        Ok(())
    }

    /// Loads a model without diagnostic performance attribution.
    pub fn load(
        runtime: MlxRuntime,
        validated_artifact: ValidatedQwen3_5MoEArtifact,
        model_directory: &Path,
        bind_mtp_weights: bool,
    ) -> Result<Self, Qwen3_5MoEExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        Self::load_with_performance_attribution(
            runtime,
            validated_artifact,
            model_directory,
            bind_mtp_weights,
            &mut disabled_performance_attribution,
        )
    }

    /// Binds the complete validated language artifact and optional vision tower.
    ///
    /// For models with a separate vision sidecar (oQ4), vision weights are loaded
    /// from the sidecar file. For models with embedded vision (oQ6e), vision
    /// weights are extracted from the indexed model shards with the language model.
    ///
    /// Every sparse model constructs an `ExpertPager` at startup for prefill and
    /// decode. The `model_directory` must point to the directory containing the
    /// safetensors shard files so the pager can build bounded byte-range plans
    /// without loading expert payloads.
    pub fn load_with_performance_attribution(
        runtime: MlxRuntime,
        mut validated_artifact: ValidatedQwen3_5MoEArtifact,
        model_directory: &Path,
        bind_mtp_weights: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Qwen3_5MoEExecutionError> {
        let config = validated_artifact.config().clone();
        let decoder_cache_layout = crate::qwen3_5_moe::qwen3_5_moe_decoder_cache_layout(&config)
            .map_err(|decoder_cache_layout_error| {
                Qwen3_5MoEExecutionError::InvalidDecoderCacheLayout {
                    description: decoder_cache_layout_error.to_string(),
                }
            })?;
        let is_dense_model = config.is_dense_model();
        let model_id = validated_artifact.model_id().to_owned();
        let model_revision = validated_artifact.revision().to_owned();
        let vision_config = validated_artifact.vision_config().cloned();
        let has_separate_vision_sidecar = validated_artifact.has_separate_vision_sidecar();
        let has_embedded_vision_tower =
            validated_artifact.supports_image_input() && !has_separate_vision_sidecar;
        let mtp_artifact_capability = validated_artifact.mtp_artifact_capability().clone();
        let shard_index = validated_artifact.shard_index().clone();
        let sidecar_vision_model = if has_separate_vision_sidecar {
            performance_attribution.measure_operation(
                PerformanceOperation::ModelSafetensorsMapping,
                |_performance_attribution| -> Result<_, Qwen3_5MoEExecutionError> {
                    Qwen3_5MoEVisionModel::load_from_sidecar(&runtime, &mut validated_artifact)
                },
            )?
        } else {
            None
        };
        let model_shards = performance_attribution.measure_operation(
            PerformanceOperation::ModelSafetensorsMapping,
            |_performance_attribution| -> Result<_, Qwen3_5MoEExecutionError> {
                let model_shard_files = validated_artifact.into_shard_files()?;
                let mut model_shards = Vec::with_capacity(model_shard_files.len());
                for model_shard_file in model_shard_files {
                    model_shards.push(runtime.load_safetensors(model_shard_file.into_file())?);
                }
                Ok(model_shards)
            },
        )?;
        let (weights, vision_model, mtp_weights) = performance_attribution.measure_operation(
            PerformanceOperation::ModelTensorBinding,
            |_performance_attribution| -> Result<_, Qwen3_5MoEExecutionError> {
                let vision_model = if has_separate_vision_sidecar {
                    sidecar_vision_model
                } else if has_embedded_vision_tower {
                    let vision_config = vision_config.as_ref().ok_or(
                        Qwen3_5MoEExecutionError::InvalidInput {
                            description: "validated visual tensors have no vision configuration",
                        },
                    )?;
                    let vision_tensor_name_to_shard_index =
                        build_vision_tensor_shard_map(&shard_index);
                    Some(Qwen3_5MoEVisionModel::load_from_model_shards(
                        &vision_config,
                        &model_shards,
                        &vision_tensor_name_to_shard_index,
                    )?)
                } else {
                    None
                };
                let weights =
                    Qwen3_5MoEWeights::bind_from_model_shards(&config, &shard_index, model_shards)?;
                let mtp_weights = bind_optional_mtp_weights(
                    bind_mtp_weights,
                    &mtp_artifact_capability,
                    &config,
                    &shard_index,
                    weights.model_shards(),
                    &weights,
                    &runtime,
                );
                Ok((weights, vision_model, mtp_weights))
            },
        )?;
        let include_mtp_sparse_expert_layer = !is_dense_model && mtp_weights.is_some();
        let expert_pager = if !is_dense_model {
            let tensor_name_to_shard_file_name: HashMap<String, String> = shard_index
                .language_tensor_name_to_shard_file_name()
                .iter()
                .chain(shard_index.mtp_tensor_name_to_shard_file_name())
                .map(|(tensor_name, shard_file_name)| {
                    (tensor_name.clone(), shard_file_name.clone())
                })
                .collect();
            Some(performance_attribution.measure_operation(
                PerformanceOperation::ExpertPagerPlanConstruction,
                |_performance_attribution| {
                    ExpertPager::new(
                        model_directory.to_path_buf(),
                        &model_id,
                        &model_revision,
                        &tensor_name_to_shard_file_name,
                        &config,
                        runtime.configured_wired_memory_limit_bytes(),
                        include_mtp_sparse_expert_layer,
                    )
                },
            )?)
        } else {
            None
        };
        let gated_delta_kernel = super::gated_delta_sequence::qwen3_5_moe_gated_delta_kernel()?;
        let sorted_expert_weighted_sum_kernel =
            super::moe::qwen3_5_moe_sorted_expert_weighted_sum_kernel()?;
        let compiled_swiglu = MlxCompiledSwiGlu::new()?;
        let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()?;
        // Qwen3.5-MoE config validation accepts BF16 activations only. Construct
        // the two invariant scales once instead of rebuilding their scalar and
        // cast graphs for every linear-attention forward pass.
        let linear_head_dimension = config.linear_key_head_dimension() as f32;
        let inverse_linear_head_dimension_scale = runtime
            .array_from_f32(&[linear_head_dimension.recip()], &[])
            .and_then(|float32_scale| runtime.astype(&float32_scale, MlxDtype::BFloat16))?;
        let inverse_square_root_linear_head_dimension_scale = runtime
            .array_from_f32(&[linear_head_dimension.sqrt().recip()], &[])
            .and_then(|float32_scale| runtime.astype(&float32_scale, MlxDtype::BFloat16))?;
        let decoder_layer_count = config.layer_count() as usize;
        let minimum_decode_route_payload_byte_count_by_layer = match expert_pager.as_ref() {
            Some(expert_pager) => expert_pager
                .minimum_decode_route_payload_byte_count_by_layer(config.experts_per_token())?,
            None => vec![0; decoder_layer_count],
        };
        let expert_layer_count = minimum_decode_route_payload_byte_count_by_layer.len();
        Ok(Self {
            runtime,
            config,
            decoder_cache_layout,
            weights,
            mtp_weights,
            vision_model,
            expert_pager,
            expert_weight_memory_cache: std::cell::RefCell::new(ExpertWeightMemoryCache::new(
                expert_layer_count,
                minimum_decode_route_payload_byte_count_by_layer,
            )),
            gated_delta_kernel,
            sorted_expert_weighted_sum_kernel,
            compiled_swiglu,
            compiled_elementwise_graphs,
            inverse_linear_head_dimension_scale,
            inverse_square_root_linear_head_dimension_scale,
        })
    }
}

fn bind_optional_mtp_weights(
    bind_mtp_weights: bool,
    mtp_artifact_capability: &Qwen3_5MoEMtpArtifactCapability,
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
    shard_index: &Qwen3_5MoEShardIndex,
    model_shards: &[MlxSafetensors],
    target_weights: &Qwen3_5MoEWeights,
    runtime: &MlxRuntime,
) -> Option<Qwen3_5MoEMtpWeights> {
    if !bind_mtp_weights || !mtp_artifact_capability.is_mtp_capable() {
        return None;
    }
    let mut mtp_weights =
        match Qwen3_5MoEMtpWeights::bind(qwen3_5_moe_config, shard_index, model_shards) {
            Ok(mtp_weights) => mtp_weights,
            Err(mtp_weight_binding_error) => {
                tracing::warn!(
                    error = %mtp_weight_binding_error,
                    "optional MTP weight binding failed; serving target-only"
                );
                None
            }
        };
    if let Some(bound_mtp_weights) = mtp_weights.as_mut()
        && let Err(mtp_normalization_repair_error) =
            bound_mtp_weights.repair_raw_normalization_weights(runtime, target_weights)
    {
        tracing::warn!(
            error = %mtp_normalization_repair_error,
            "optional MTP normalization repair failed; serving target-only"
        );
        mtp_weights = None;
    }
    if mtp_weights.is_none()
        && let Err(mlx_allocator_cleanup_error) = runtime.clear_allocator_cache()
    {
        tracing::warn!(
            error = %mlx_allocator_cleanup_error,
            "failed to reclaim allocator memory after optional MTP initialization failure"
        );
    }
    mtp_weights
}

/// Maps embedded vision tensor names to their loaded model-shard positions.
fn build_vision_tensor_shard_map(
    shard_index: &super::Qwen3_5MoEShardIndex,
) -> HashMap<String, usize> {
    let mut vision_tensor_name_to_shard_index = HashMap::new();
    for (vision_tensor_name, shard_file_name) in shard_index.vision_tensor_name_to_shard_file_name()
    {
        if let Some(shard_position) = shard_index
            .model_shard_file_names()
            .iter()
            .position(|certified_file_name| *certified_file_name == *shard_file_name)
        {
            vision_tensor_name_to_shard_index.insert(vision_tensor_name.clone(), shard_position);
        }
    }
    vision_tensor_name_to_shard_index
}
