//! Artifact binding and startup-only MLX resource construction for Qwen3.5.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use astronomical_runtime_integration::{
    MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxDtype, MlxRuntime,
};

use crate::expert_paging::RetainedExpertLayerCache;
use crate::qwen3_5_moe::{
    Qwen3_5ExpertPager, Qwen3_5RetainedExpertLayer, qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
use crate::{
    MlxRamBudget, MlxRamBudgetModelGeometry, PerformanceAttribution, PerformanceOperation,
};

use super::model::Qwen3_5Model;
use super::{
    Qwen3_5ExecutionError, Qwen3_5FeedForwardArchitecture, Qwen3_5VisionModel, Qwen3_5Weights,
    ValidatedQwen3_5Artifact,
};
use crate::qwen3_5::multi_token_prediction::bind_optional_weights;

impl Qwen3_5Model {
    /// Loads a model without diagnostic performance attribution.
    pub fn load(
        runtime: MlxRuntime,
        validated_artifact: ValidatedQwen3_5Artifact,
        model_directory: &Path,
        bind_mtp_weights: bool,
        chunking: super::Qwen3_5ModelChunkingConfiguration,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        Self::load_with_performance_attribution(
            runtime,
            validated_artifact,
            model_directory,
            bind_mtp_weights,
            true,
            chunking,
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
        should_bind_vision_weights: bool,
        chunking: super::Qwen3_5ModelChunkingConfiguration,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let config = validated_artifact.config().clone();
        let vision_config = validated_artifact.vision_config().cloned();
        let has_separate_vision_sidecar =
            should_bind_vision_weights && validated_artifact.has_separate_vision_sidecar();
        let has_embedded_vision_tower = should_bind_vision_weights
            && validated_artifact.supports_image_input()
            && !has_separate_vision_sidecar;
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
                let mtp_weights = bind_optional_weights(
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
        let (expert_pager, retained_expert_layers, sorted_expert_weighted_sum_kernel) =
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
                                &runtime,
                                model_directory.to_path_buf(),
                                &tensor_name_to_shard_file_name,
                                &config,
                                // The live configurable MLX active-memory ceiling is the only
                                // paging budget. Reading the runtime policy here lets reloads
                                // apply on every machine without copying a stale wired limit.
                                runtime.memory_limits().active_memory_limit_bytes(),
                                mtp_weights.is_some(),
                            )
                        },
                    )?;
                    let sorted_expert_weighted_sum_kernel =
                        qwen3_5_moe_sorted_expert_weighted_sum_kernel()?;
                    let retained_expert_layers =
                        RefCell::new(RetainedExpertLayerCache::<Qwen3_5RetainedExpertLayer>::new(
                            expert_pager.layer_count(),
                        ));
                    (
                        Some(expert_pager),
                        Some(retained_expert_layers),
                        Some(sorted_expert_weighted_sum_kernel),
                    )
                }
            };
        // Cache geometry cannot be finalized before weights and sparse plans are
        // bound: their source dtypes determine MLX result-type promotion. Derive
        // the persistence contract at that boundary without evaluating tensors.
        let decoder_cache_layout = performance_attribution.measure_operation(
            PerformanceOperation::ModelTensorBinding,
            |_performance_attribution| {
                let decoder_layer_cache_dtypes =
                    super::decoder_cache_dtype_flow::derive_decoder_layer_cache_dtypes(
                        &weights,
                        expert_pager.as_ref(),
                    )?;
                crate::qwen3_5::qwen3_5_decoder_cache_layout(
                    &config,
                    usize::try_from(chunking.full_attention_key_value_growth_tokens).map_err(
                        |_| Qwen3_5ExecutionError::InvalidInput {
                            description:
                                "full-attention key/value growth tokens exceed the usize range",
                        },
                    )?,
                    &decoder_layer_cache_dtypes,
                )
                .map_err(|decoder_cache_layout_error| {
                    Qwen3_5ExecutionError::InvalidDecoderCacheLayout {
                        description: decoder_cache_layout_error.to_string(),
                    }
                })
            },
        )?;
        let gated_delta_kernel = super::gated_delta_sequence::qwen3_5_gated_delta_kernel()?;
        let gated_delta_checkpoint_kernel =
            super::gated_delta_boundary_checkpoints::qwen3_5_gated_delta_checkpoint_kernel()?;
        let target_verification_quantized_linear_kernel =
            super::target_verification_quantized_linear::target_verification_quantized_linear_kernel()?;
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
        let model_core_payload_bytes = weights
            .total_payload_bytes()
            .saturating_add(
                mtp_weights
                    .as_ref()
                    .map_or(0, |mtp_weights| mtp_weights.payload_byte_count()),
            )
            .saturating_add(
                vision_model
                    .as_ref()
                    .map_or(0, Qwen3_5VisionModel::resident_payload_bytes),
            );
        let complete_expert_payload_bytes = match expert_pager.as_ref() {
            Some(expert_pager) => expert_pager.complete_expert_payload_byte_count().map_err(
                |_| Qwen3_5ExecutionError::InvalidInput {
                    description: "complete expert payload byte count overflowed during model load",
                },
            )?,
            None => 0,
        };
        let largest_complete_expert_layer_bytes = expert_pager
            .as_ref()
            .map_or(0, Qwen3_5ExpertPager::maximum_expert_page_bytes);
        let largest_routed_expert_page_bytes =
            expert_pager.as_ref().map_or(Ok(0), |expert_pager| {
                expert_pager.maximum_routed_expert_page_bytes(
                    usize::try_from(config.experts_per_token()).unwrap_or(usize::MAX),
                )
            })?;
        let mlx_ram_budget = MlxRamBudget::new(
            u64::try_from(runtime.memory_limits().active_memory_limit_bytes()).map_err(|_| {
                Qwen3_5ExecutionError::InvalidInput {
                    description: "MLX active memory ceiling exceeds the u64 range",
                }
            })?,
            MlxRamBudgetModelGeometry {
                model_core_payload_bytes,
                complete_expert_payload_bytes,
                largest_complete_expert_layer_bytes,
                largest_routed_expert_page_bytes,
            },
        )
        .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
            description: "MLX RAM budget requires a positive active-memory ceiling",
        })?;
        Ok(Self {
            runtime,
            config,
            decoder_cache_layout,
            weights,
            mtp_weights,
            vision_model,
            expert_pager,
            // Publication occurs only after core materialization and a fresh idle
            // memory sample in the engine loading path.
            resident_expert_weights: None,
            retained_expert_layers,
            mlx_ram_budget: RefCell::new(mlx_ram_budget),
            active_expert_residency_plan: RefCell::new(None),

            gated_delta_kernel,
            gated_delta_checkpoint_kernel,
            sorted_expert_weighted_sum_kernel,
            target_verification_quantized_linear_kernel,
            compiled_swiglu,
            compiled_elementwise_graphs,
            chunking,
            inverse_linear_head_dimension_scale,
            inverse_square_root_linear_head_dimension_scale,
            paged_forward_missing_route_collector:
                crate::qwen3_5_moe::PagedForwardMissingRouteCollector::default(),
        })
    }
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
