//! Artifact binding and startup-only MLX resource construction for Qwen3.5.

use std::collections::HashMap;
use std::path::Path;

use astronomical_runtime_integration::{
    MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxDtype, MlxRuntime, MlxSafetensors,
};

use crate::expert_paging::ExpertWeightMemoryCache;
use crate::qwen3_5::Qwen3_5MtpArtifactCapability;
use crate::qwen3_5_moe::{
    Qwen3_5ExpertPager, Qwen3_5PagedExpertWeights, qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::model::Qwen3_5Model;
use super::mtp::Qwen3_5MtpWeights;
use super::{
    Qwen3_5Config, Qwen3_5ExecutionError, Qwen3_5FeedForwardArchitecture, Qwen3_5ShardIndex,
    Qwen3_5VisionModel, Qwen3_5Weights, ValidatedQwen3_5Artifact,
};

impl Qwen3_5Model {
    pub(crate) fn prewarm_complete_expert_layers_with_performance_attribution(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(());
        };
        let expert_weight_memory_cache = self.sparse_expert_weight_memory_cache()?;
        expert_pager.prewarm_complete_layers_with_performance_attribution(
            &self.runtime,
            expert_weight_memory_cache,
            performance_attribution,
        )?;
        Ok(())
    }

    /// Loads a model without diagnostic performance attribution.
    pub fn load(
        runtime: MlxRuntime,
        validated_artifact: ValidatedQwen3_5Artifact,
        model_directory: &Path,
        bind_mtp_weights: bool,
    ) -> Result<Self, Qwen3_5ExecutionError> {
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
    /// Separate indexed vision files are loaded through the visual owner. Embedded
    /// vision weights are extracted from indexed language model shards.
    ///
    /// Every sparse model constructs a Qwen3_5ExpertPager at startup for prefill and
    /// decode. The `model_directory` must point to the directory containing the
    /// safetensors shard files so the pager can build bounded byte-range plans
    /// without loading expert payloads.
    pub fn load_with_performance_attribution(
        runtime: MlxRuntime,
        mut validated_artifact: ValidatedQwen3_5Artifact,
        model_directory: &Path,
        bind_mtp_weights: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let config = validated_artifact.config().clone();
        let decoder_cache_layout = crate::qwen3_5::qwen3_5_decoder_cache_layout(&config).map_err(
            |decoder_cache_layout_error| Qwen3_5ExecutionError::InvalidDecoderCacheLayout {
                description: decoder_cache_layout_error.to_string(),
            },
        )?;
        let vision_config = validated_artifact.vision_config().cloned();
        let has_separate_vision_sidecar = validated_artifact.has_separate_vision_sidecar();
        let has_embedded_vision_tower =
            validated_artifact.supports_image_input() && !has_separate_vision_sidecar;
        let mtp_artifact_capability = validated_artifact.mtp_artifact_capability().clone();
        let shard_index = validated_artifact.shard_index().clone();
        let sidecar_vision_model = if has_separate_vision_sidecar {
            performance_attribution.measure_operation(
                PerformanceOperation::ModelSafetensorsMapping,
                |_performance_attribution| -> Result<_, Qwen3_5ExecutionError> {
                    Qwen3_5VisionModel::load_from_sidecar(&runtime, &mut validated_artifact)
                },
            )?
        } else {
            None
        };
        let should_load_mtp_only_shards =
            bind_mtp_weights && mtp_artifact_capability.is_mtp_capable();
        let mtp_only_shard_files = if should_load_mtp_only_shards {
            validated_artifact.take_mtp_only_shard_files()?
        } else {
            Vec::new()
        };
        let (model_shards, mtp_only_shards) = performance_attribution.measure_operation(
            PerformanceOperation::ModelSafetensorsMapping,
            |performance_attribution| -> Result<_, Qwen3_5ExecutionError> {
                let model_shard_files = validated_artifact.into_shard_files()?;
                let mut model_shards = Vec::with_capacity(model_shard_files.len());
                let mut mtp_only_shards = Vec::with_capacity(mtp_only_shard_files.len());
                let positional_file_read_metrics =
                    performance_attribution.positional_file_read_metrics();
                for model_shard_file in model_shard_files {
                    model_shards.push(
                        runtime.load_safetensors(
                            model_shard_file.into_file(),
                            positional_file_read_metrics
                                .as_ref()
                                .map(std::sync::Arc::clone),
                        )?,
                    );
                }
                for mtp_only_shard_file in mtp_only_shard_files {
                    mtp_only_shards.push(
                        runtime.load_safetensors(
                            mtp_only_shard_file.into_file(),
                            positional_file_read_metrics
                                .as_ref()
                                .map(std::sync::Arc::clone),
                        )?,
                    );
                }
                Ok((model_shards, mtp_only_shards))
            },
        )?;
        let (weights, vision_model, mtp_weights) = performance_attribution.measure_operation(
            PerformanceOperation::ModelTensorBinding,
            |_performance_attribution| -> Result<_, Qwen3_5ExecutionError> {
                let vision_model = if has_separate_vision_sidecar {
                    sidecar_vision_model
                } else if has_embedded_vision_tower {
                    let vision_config = vision_config.as_ref().ok_or(
                        Qwen3_5ExecutionError::InvalidInput {
                            description: "validated visual tensors have no vision configuration",
                        },
                    )?;
                    let vision_tensor_name_to_shard_index =
                        build_vision_tensor_shard_map(&shard_index);
                    Some(Qwen3_5VisionModel::load_from_model_shards(
                        &vision_config,
                        &model_shards,
                        &vision_tensor_name_to_shard_index,
                    )?)
                } else {
                    None
                };
                let weights =
                    Qwen3_5Weights::bind_from_model_shards(&config, &shard_index, model_shards)?;
                let mtp_weights = bind_optional_mtp_weights(
                    bind_mtp_weights,
                    &mtp_artifact_capability,
                    &config,
                    &shard_index,
                    weights.model_shards(),
                    mtp_only_shards,
                    &weights,
                    &runtime,
                );
                Ok((weights, vision_model, mtp_weights))
            },
        )?;
        let (expert_pager, expert_weight_memory_cache, sorted_expert_weighted_sum_kernel) =
            match config.feed_forward_architecture() {
                Qwen3_5FeedForwardArchitecture::Dense => (None, None, None),
                Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
                    let tensor_name_to_shard_file_name: HashMap<String, String> = shard_index
                        .language_tensor_name_to_shard_file_name()
                        .iter()
                        .chain(shard_index.mtp_tensor_name_to_shard_file_name())
                        .map(|(tensor_name, shard_file_name)| {
                            (tensor_name.clone(), shard_file_name.clone())
                        })
                        .collect();
                    let expert_pager = performance_attribution.measure_operation(
                        PerformanceOperation::ExpertPagerPlanConstruction,
                        |_performance_attribution| {
                            Qwen3_5ExpertPager::new(
                                model_directory.to_path_buf(),
                                &tensor_name_to_shard_file_name,
                                &config,
                                runtime.configured_wired_memory_limit_bytes(),
                                mtp_weights.is_some(),
                            )
                        },
                    )?;
                    let minimum_decode_route_payload_byte_count_by_layer = expert_pager
                        .minimum_decode_route_payload_byte_count_by_layer(
                            config.experts_per_token(),
                        )?;
                    let expert_layer_count = minimum_decode_route_payload_byte_count_by_layer.len();
                    let expert_weight_memory_cache = std::cell::RefCell::new(
                        ExpertWeightMemoryCache::<Qwen3_5PagedExpertWeights>::new(
                            expert_layer_count,
                            minimum_decode_route_payload_byte_count_by_layer,
                        ),
                    );
                    let sorted_expert_weighted_sum_kernel =
                        qwen3_5_moe_sorted_expert_weighted_sum_kernel()?;
                    (
                        Some(expert_pager),
                        Some(expert_weight_memory_cache),
                        Some(sorted_expert_weighted_sum_kernel),
                    )
                }
            };
        let gated_delta_kernel = super::gated_delta_sequence::qwen3_5_gated_delta_kernel()?;
        let gated_delta_checkpoint_kernel =
            super::gated_delta_boundary_checkpoints::qwen3_5_gated_delta_checkpoint_kernel()?;
        let compiled_swiglu = MlxCompiledSwiGlu::new()?;
        let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()?;
        // Qwen3.5 config validation accepts BF16 activations only. Construct
        // the two invariant scales once instead of rebuilding their scalar and
        // cast graphs for every linear-attention forward pass.
        let linear_head_dimension = config.linear_key_head_dimension() as f32;
        let inverse_linear_head_dimension_scale = runtime
            .array_from_f32(&[linear_head_dimension.recip()], &[])
            .and_then(|float32_scale| runtime.astype(&float32_scale, MlxDtype::BFloat16))?;
        let inverse_square_root_linear_head_dimension_scale = runtime
            .array_from_f32(&[linear_head_dimension.sqrt().recip()], &[])
            .and_then(|float32_scale| runtime.astype(&float32_scale, MlxDtype::BFloat16))?;
        Ok(Self {
            runtime,
            config,
            decoder_cache_layout,
            weights,
            mtp_weights,
            vision_model,
            expert_pager,
            expert_weight_memory_cache,
            gated_delta_kernel,
            gated_delta_checkpoint_kernel,
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
    mtp_artifact_capability: &Qwen3_5MtpArtifactCapability,
    qwen3_5_config: &Qwen3_5Config,
    shard_index: &Qwen3_5ShardIndex,
    model_shards: &[MlxSafetensors],
    mtp_only_shards: Vec<MlxSafetensors>,
    target_weights: &Qwen3_5Weights,
    runtime: &MlxRuntime,
) -> Option<Qwen3_5MtpWeights> {
    if !bind_mtp_weights || !mtp_artifact_capability.is_mtp_capable() {
        return None;
    }
    let mut mtp_weights =
        match Qwen3_5MtpWeights::bind(qwen3_5_config, shard_index, model_shards, mtp_only_shards) {
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
fn build_vision_tensor_shard_map(shard_index: &super::Qwen3_5ShardIndex) -> HashMap<String, usize> {
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
