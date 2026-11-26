//! Artifact binding and startup-only MLX resource construction for Qwen3.5.

use std::path::Path;

use runtime_integration::MlxRuntime;

use crate::qwen3_5_core::artifacts::QuantizedExpertLayerPlan;
use crate::qwen3_5_core::model::loading_pipeline::{
    BoundLanguageArtifact, InvariantLinearScales, ModelKernelVerdicts, bind_language_artifact_for,
    load_invariant_linear_scales, load_model_kernel_verdicts,
};
use crate::{ModelMemoryFacts, PerformanceAttribution, PerformanceOperation};

use crate::qwen3_5_core::model::loading_geometry::{
    ExpertPayloadGeometry, ExpertTensorNameMaps, expert_tensor_name_maps,
    measured_astro_decoder_state_cache_layout, model_memory_facts,
};

use super::Qwen3_5ResidentModel;
use super::{
    Qwen3_5FeedForwardArchitecture, Qwen3_5VisionModel, Qwen3_5Weights, TextExecutionError,
    ValidatedQwen3_5Artifact,
};
use mlx_bridge::{MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMetalKernel};

/// The resident engine holds every tensor as bound, so it binds them in the
/// dtype the compute device runs the model accurately in (#2441).
const LANGUAGE_WEIGHT_BINDING_DTYPE: crate::qwen3_5_core::model_math::weights::WeightBindingDtype =
    crate::qwen3_5_core::model_math::weights::WeightBindingDtype::ForComputeDevice;
impl Qwen3_5ResidentModel {
    /// Loads a model without diagnostic performance attribution.
    pub fn load(
        runtime: MlxRuntime,
        validated_artifact: ValidatedQwen3_5Artifact,
        model_directory: &Path,
        chunking: super::TextModelChunkingConfiguration,
    ) -> Result<Self, TextExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        Self::load_with_performance_attribution(
            runtime,
            validated_artifact,
            model_directory,
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
    /// Sparse artifacts build shared expert-layer plans from their indexed
    /// SafeTensors metadata; complete expert tensors are materialized later.
    pub fn load_with_performance_attribution(
        runtime: MlxRuntime,
        mut validated_artifact: ValidatedQwen3_5Artifact,
        model_directory: &Path,
        should_bind_vision_weights: bool,
        chunking: super::TextModelChunkingConfiguration,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, TextExecutionError> {
        // Taken before binding, which consumes parts of the artifact.
        let expert_atlas = validated_artifact.expert_atlas().cloned();
        let BoundLanguageArtifact {
            config,
            shard_index,
            tensor_inventory,
            weights,
            vision_model,
            ..
        } = bind_language_artifact_for(
            &runtime,
            &mut validated_artifact,
            should_bind_vision_weights,
            LANGUAGE_WEIGHT_BINDING_DTYPE,
            performance_attribution,
        )?;
        let ResidentMixtureOfExpertsParts {
            expert_layer_plans,
            sorted_expert_weighted_sum_kernel,
            sorted_expert_shared_combination_kernel,
        } = match config.feed_forward_architecture() {
            Qwen3_5FeedForwardArchitecture::Dense => ResidentMixtureOfExpertsParts::default(),
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts => resident_mixture_of_experts_parts(
                &runtime,
                model_directory,
                &config,
                &shard_index,
                &tensor_inventory,
                expert_atlas.as_ref(),
                performance_attribution,
            )?,
        };
        let astro_decoder_state_cache_layout = measured_astro_decoder_state_cache_layout(
            &config,
            &chunking,
            &weights,
            expert_layer_plans.as_slice(),
            performance_attribution,
        )?;
        let ModelKernelVerdicts {
            gated_delta_kernel,
            gated_delta_checkpoint_kernel,
            gdn_decode_prework_kernel,
        } = load_model_kernel_verdicts(&runtime, &config, performance_attribution)?;
        let compiled_swiglu = MlxCompiledSwiGlu::new()?;
        let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()?;
        let InvariantLinearScales {
            inverse_linear_head_dimension_scale,
            inverse_square_root_linear_head_dimension_scale,
            query_normalization_scale_weight,
            key_normalization_scale_weight,
        } = load_invariant_linear_scales(&runtime, &config, LANGUAGE_WEIGHT_BINDING_DTYPE)?;
        let memory_facts = build_resident_model_memory_facts(
            &runtime,
            &config,
            &weights,
            &vision_model,
            &expert_layer_plans,
            &astro_decoder_state_cache_layout,
        )?;
        Ok(Self {
            base: super::Qwen3_5ModelBase {
                disk_embedding_rows: None,
                ngram_table: None,
                runtime,
                config,
                astro_decoder_state_cache_layout,
                weights,
                vision_model,
                memory_facts: std::cell::RefCell::new(memory_facts),
                gated_delta_kernel,
                gated_delta_checkpoint_kernel,
                gdn_decode_prework_kernel,
                optional_accelerators:
                    crate::qwen3_5_core::model::base::Qwen3_5OptionalAccelerators {
                        sorted_expert_shared_combination_kernel,
                        ..Default::default()
                    },
                sorted_expert_weighted_sum_kernel,
                compiled_swiglu,
                compiled_elementwise_graphs,
                chunking,
                inverse_linear_head_dimension_scale,
                inverse_square_root_linear_head_dimension_scale,
                query_normalization_scale_weight,
                key_normalization_scale_weight,
                layer_additions: std::cell::RefCell::new(None),
            },
            expert_layer_plans,
            // Publication occurs only after core materialization and a fresh idle
            // memory sample in the engine loading path.
            resident_expert_weights: None,
            // Programs load after the weights are materialized, if at all.
            neural_engine_mlp_prefill: None,
            memory_governor: std::cell::RefCell::new(None),
        })
    }
}

/// The resident engine's MoE-only pieces: the quantized expert layer plans
/// built from the validated inventory and the custom MoE kernels this
/// machine's capability probes kept. A dense model has none of them.
#[derive(Default)]
struct ResidentMixtureOfExpertsParts {
    expert_layer_plans: Vec<QuantizedExpertLayerPlan>,
    sorted_expert_weighted_sum_kernel: Option<MlxMetalKernel>,
    sorted_expert_shared_combination_kernel: Option<MlxMetalKernel>,
}

/// Builds the resident expert layer plans for a sparse model, under the
/// plan-construction attribution span, and keeps the custom MoE kernels this
/// machine's capability verdicts allow.
#[allow(clippy::too_many_arguments)]
fn load_resident_mixture_of_experts_parts(
    runtime: &MlxRuntime,
    model_directory: &Path,
    config: &crate::Qwen3_5Config,
    shard_index: &super::Qwen3_5ShardIndex,
    tensor_inventory: &crate::TensorInventory,
    expert_atlas: Option<&expert_atlas::AtlasManifest>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<ResidentMixtureOfExpertsParts, TextExecutionError> {
    let ExpertTensorNameMaps {
        tensor_name_to_shard_file_name,
        stored_tensor_name_by_canonical_name,
    } = expert_tensor_name_maps(shard_index, tensor_inventory);
    let expert_layer_plans = performance_attribution.measure_operation(
        PerformanceOperation::ExpertLayerPlanConstruction,
        |_performance_attribution| {
            crate::qwen3_5_core::artifacts::atlas_expert_layer_plans::build_expert_layer_plans(
                model_directory,
                expert_atlas,
                &tensor_name_to_shard_file_name,
                &stored_tensor_name_by_canonical_name,
                config,
                config.layer_count() as usize,
            )
            .map_err(|layer_plan_error| TextExecutionError::ExpertLayerPlan {
                description: layer_plan_error.to_string(),
            })
        },
    )?;
    let worker_kernel_capabilities = crate::kernel_capability::worker_process_kernel_capabilities(
        runtime,
        performance_attribution,
    );
    let sorted_expert_weighted_sum_kernel =
        retained_sorted_expert_weighted_sum_kernel(worker_kernel_capabilities)?;
    let sorted_expert_shared_combination_kernel = crate::qwen3_5_core::model::sorted_expert_shared_combination_switch::retained_sorted_expert_shared_combination_kernel(
        worker_kernel_capabilities,
        sorted_expert_weighted_sum_kernel.is_some(),
    )?;
    Ok(ResidentMixtureOfExpertsParts {
        expert_layer_plans,
        sorted_expert_weighted_sum_kernel,
        sorted_expert_shared_combination_kernel,
    })
}

/// The GPU capability verdict decides whether this worker process retains the
/// measured sorted-reduction kernel. A demoted kernel leaves `None` and every
/// MoE forward takes the unsorted MLX route for the worker's lifetime.
fn retained_sorted_expert_weighted_sum_kernel(
    worker_kernel_capabilities: &crate::kernel_capability::WorkerKernelCapabilities,
) -> Result<Option<MlxMetalKernel>, TextExecutionError> {
    let sorted_reduction_family = crate::kernel_capability::SORTED_EXPERT_WEIGHTED_SUM;
    if !worker_kernel_capabilities.is_custom_kernel_supported(sorted_reduction_family) {
        tracing::info!(
            "sorted expert weighted-sum kernel demoted to the MLX fallback for this worker process"
        );
        return Ok(None);
    }
    Ok(Some(
        crate::qwen3_5_core::model_math::routing::qwen3_5_moe_sorted_expert_weighted_sum_kernel()?,
    ))
}

/// Assembles the limit and geometry this engine runs under from the
/// model-core payload and the expert layer plans.
#[allow(clippy::too_many_arguments)]
fn build_resident_model_memory_facts(
    runtime: &MlxRuntime,
    config: &crate::Qwen3_5Config,
    weights: &Qwen3_5Weights,
    vision_model: &Option<Qwen3_5VisionModel>,
    expert_layer_plans: &[QuantizedExpertLayerPlan],
    astro_decoder_state_cache_layout: &crate::AstroDecoderStateCacheLayout,
) -> Result<ModelMemoryFacts, TextExecutionError> {
    let model_core_payload_bytes =
        crate::qwen3_5_core::model::loading_pipeline::model_core_payload_bytes(
            weights,
            crate::qwen3_5_core::model_math::weights::EmbeddingResidency::Resident,
            vision_model,
        );
    let complete_expert_payload_bytes =
        expert_layer_plans
            .iter()
            .try_fold(0_u64, |total_payload_bytes, layer_plan| {
                let layer_payload_bytes =
                    layer_plan
                        .complete_expert_payload_byte_count()
                        .map_err(|plan_error| TextExecutionError::ExpertLayerPlan {
                            description: plan_error.to_string(),
                        })?;
                total_payload_bytes
                .checked_add(layer_payload_bytes)
                .ok_or(TextExecutionError::InvalidInput {
                    description: "complete expert payload byte count overflowed during model load",
                })
            })?;
    let largest_complete_expert_layer_bytes = expert_layer_plans
        .iter()
        .map(|layer_plan| {
            layer_plan
                .complete_expert_payload_byte_count()
                .map_err(|plan_error| TextExecutionError::ExpertLayerPlan {
                    description: plan_error.to_string(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let largest_routed_expert_page_bytes = expert_layer_plans.iter().try_fold(
        0_u64,
        |largest_page_bytes, layer_plan| -> Result<u64, TextExecutionError> {
            let complete_layer_payload_bytes = layer_plan
                .complete_expert_payload_byte_count()
                .map_err(|plan_error| TextExecutionError::ExpertLayerPlan {
                    description: plan_error.to_string(),
                })?;
            let routed_expert_count = usize::try_from(config.experts_per_token())
                .unwrap_or(usize::MAX)
                .min(layer_plan.expert_capacity);
            let routed_page_bytes = u128::from(complete_layer_payload_bytes)
                .saturating_mul(routed_expert_count as u128)
                / (layer_plan.expert_capacity.max(1) as u128);
            Ok(largest_page_bytes.max(u64::try_from(routed_page_bytes).unwrap_or(u64::MAX)))
        },
    )?;
    model_memory_facts(
        runtime,
        model_core_payload_bytes,
        ExpertPayloadGeometry {
            complete_expert_payload_bytes,
            largest_complete_expert_layer_bytes,
            largest_routed_expert_page_bytes,
        },
        astro_decoder_state_cache_layout,
    )
}

/// Flash-Next is served by the streaming engine only (issue #2533): its
/// routed experts alone are 70 GB, and its lanes, indexer and n-gram table
/// run only on the streaming layer path.
#[allow(clippy::too_many_arguments)]
fn resident_mixture_of_experts_parts(
    runtime: &MlxRuntime,
    model_directory: &Path,
    config: &crate::Qwen3_5Config,
    shard_index: &super::Qwen3_5ShardIndex,
    tensor_inventory: &crate::TensorInventory,
    expert_atlas: Option<&expert_atlas::AtlasManifest>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<ResidentMixtureOfExpertsParts, TextExecutionError> {
    if config.qwen4_exp_architecture().is_some() {
        return Err(TextExecutionError::InvalidInput {
            description: "Qwen3.8-Flash-Next runs on the streaming engine only",
        });
    }
    load_resident_mixture_of_experts_parts(
        runtime,
        model_directory,
        config,
        shard_index,
        tensor_inventory,
        expert_atlas,
        performance_attribution,
    )
}
