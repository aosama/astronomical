//! Artifact binding and startup-only MLX resource construction for Qwen3.5.

use std::path::Path;

use astronomical_runtime_integration::MlxRuntime;

use crate::qwen3_5_core::artifacts::QuantizedExpertLayerPlan;
use crate::qwen3_5_core::model::loading_pipeline::{
    BoundLanguageArtifact, InvariantLinearScales, ModelKernelVerdicts, bind_language_artifact,
    load_invariant_linear_scales, load_model_kernel_verdicts,
};
use crate::{
    MlxRamBudget, MlxRamBudgetModelGeometry, PerformanceAttribution, PerformanceOperation,
};

use super::Qwen3_5ModelBase;
use super::Qwen3_5ResidentModel;
use super::{
    Qwen3_5ExecutionError, Qwen3_5FeedForwardArchitecture, Qwen3_5VisionModel, Qwen3_5Weights,
    ValidatedQwen3_5Artifact,
};
use mlx_c_rust::{MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMetalKernel};

impl Qwen3_5ResidentModel {
    /// Loads a model without diagnostic performance attribution.
    pub fn load(
        runtime: MlxRuntime,
        validated_artifact: ValidatedQwen3_5Artifact,
        model_directory: &Path,
        chunking: super::Qwen3_5ModelChunkingConfiguration,
    ) -> Result<Self, Qwen3_5ExecutionError> {
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
        chunking: super::Qwen3_5ModelChunkingConfiguration,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let BoundLanguageArtifact {
            config,
            shard_index,
            tensor_inventory,
            weights,
            vision_model,
            ..
        } = bind_language_artifact(
            &runtime,
            &mut validated_artifact,
            should_bind_vision_weights,
            performance_attribution,
        )?;
        let (expert_layer_plans, sorted_expert_weighted_sum_kernel) =
            match config.feed_forward_architecture() {
                Qwen3_5FeedForwardArchitecture::Dense => (Vec::new(), None),
                Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
                    let resident_expert_layer_plans = load_resident_expert_layer_plans(
                        &runtime,
                        model_directory,
                        &config,
                        &shard_index,
                        &tensor_inventory,
                        performance_attribution,
                    )?;
                    (
                        resident_expert_layer_plans.expert_layer_plans,
                        resident_expert_layer_plans.sorted_expert_weighted_sum_kernel,
                    )
                }
            };
        // Cache geometry cannot be finalized before weights and sparse plans are
        // bound: their source dtypes determine MLX result-type promotion. Derive
        // the persistence contract at that boundary without evaluating tensors.
        let decoder_cache_layout = performance_attribution.measure_operation(
            PerformanceOperation::ModelTensorBinding,
            |_performance_attribution| {
                decoder_cache_layout_at_tensor_binding(
                    &config,
                    &chunking,
                    &weights,
                    expert_layer_plans.as_slice(),
                )
            },
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
        } = load_invariant_linear_scales(&runtime, &config)?;
        let mlx_ram_budget = build_resident_mlx_ram_budget(
            &runtime,
            &config,
            &weights,
            &vision_model,
            &expert_layer_plans,
            &decoder_cache_layout,
        )?;
        Ok(Self {
            base: super::Qwen3_5ModelBase {
                runtime,
                config,
                decoder_cache_layout,
                weights,
                vision_model,
                mlx_ram_budget: std::cell::RefCell::new(mlx_ram_budget),
                gated_delta_kernel,
                gated_delta_checkpoint_kernel,
                gdn_decode_prework_kernel,
                sorted_expert_weighted_sum_kernel,
                compiled_swiglu,
                compiled_elementwise_graphs,
                chunking,
                inverse_linear_head_dimension_scale,
                inverse_square_root_linear_head_dimension_scale,
                query_normalization_scale_weight,
                key_normalization_scale_weight,
            },
            expert_layer_plans,
            // Publication occurs only after core materialization and a fresh idle
            // memory sample in the engine loading path.
            resident_expert_weights: None,
        })
    }
}

/// The resident engine's MoE-only pieces: the quantized expert layer plans
/// built from the validated inventory and the sorted-reduction kernel verdict.
struct ResidentExpertLayerPlans {
    expert_layer_plans: Vec<QuantizedExpertLayerPlan>,
    sorted_expert_weighted_sum_kernel: Option<MlxMetalKernel>,
}

/// Builds the resident expert layer plans for a sparse model. Mirrors the MoE
/// branch of the inlined loader, including the plan-construction attribution
/// span and the sorted-reduction verdict.
#[allow(clippy::too_many_arguments)]
fn load_resident_expert_layer_plans(
    runtime: &MlxRuntime,
    model_directory: &Path,
    config: &crate::Qwen3_5Config,
    shard_index: &super::Qwen3_5ShardIndex,
    tensor_inventory: &crate::TensorInventory,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<ResidentExpertLayerPlans, Qwen3_5ExecutionError> {
    let tensor_name_to_shard_file_name: std::collections::HashMap<String, String> = shard_index
        .language_tensor_name_to_shard_file_name()
        .iter()
        .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
        .collect();
    let stored_tensor_name_by_canonical_name = tensor_inventory
        .locations()
        .map(|location| {
            (
                location.canonical_name().to_owned(),
                location.stored_name().to_owned(),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let expert_layer_plans = performance_attribution.measure_operation(
        PerformanceOperation::ExpertLayerPlanConstruction,
        |_performance_attribution| {
            crate::qwen3_5_core::artifacts::build_quantized_expert_layer_plans(
                model_directory,
                &tensor_name_to_shard_file_name,
                &stored_tensor_name_by_canonical_name,
                config,
                config.layer_count() as usize,
            )
            .map_err(|layer_plan_error| Qwen3_5ExecutionError::ExpertLayerPlan {
                description: layer_plan_error.to_string(),
            })
        },
    )?;
    // The GPU capability verdict decides whether this worker
    // process retains the measured sorted-reduction kernel. A
    // demoted kernel leaves None and every MoE forward takes the
    // unsorted MLX route for the worker's lifetime.
    let sorted_expert_weighted_sum_kernel =
        if crate::kernel_capability::worker_process_kernel_capabilities(
            runtime,
            performance_attribution,
        )
        .is_custom_kernel_supported(
            crate::kernel_capability::CustomMetalKernelFamily::SORTED_EXPERT_WEIGHTED_SUM,
        ) {
            Some(
                crate::qwen3_5_core::model_math::routing::qwen3_5_moe_sorted_expert_weighted_sum_kernel(
                )?,
            )
        } else {
            tracing::info!(
                "sorted expert weighted-sum kernel demoted to the MLX fallback for this worker process"
            );
            None
        };
    Ok(ResidentExpertLayerPlans {
        expert_layer_plans,
        sorted_expert_weighted_sum_kernel,
    })
}

/// Assembles the resident engine's RAM budget from the model-core payload and
/// the expert layer plans' geometry. Mirrors the payload folds and budget
/// construction of the inlined loader.
#[allow(clippy::too_many_arguments)]
fn build_resident_mlx_ram_budget(
    runtime: &MlxRuntime,
    config: &crate::Qwen3_5Config,
    weights: &Qwen3_5Weights,
    vision_model: &Option<Qwen3_5VisionModel>,
    expert_layer_plans: &[QuantizedExpertLayerPlan],
    decoder_cache_layout: &crate::DecoderCacheLayout,
) -> Result<MlxRamBudget, Qwen3_5ExecutionError> {
    let model_core_payload_bytes =
        crate::qwen3_5_core::model::loading_pipeline::model_core_payload_bytes(
            weights,
            vision_model,
        );
    let complete_expert_payload_bytes =
        expert_layer_plans
            .iter()
            .try_fold(0_u64, |total_payload_bytes, layer_plan| {
                let layer_payload_bytes =
                    layer_plan
                        .complete_expert_payload_byte_count()
                        .map_err(|plan_error| Qwen3_5ExecutionError::ExpertLayerPlan {
                            description: plan_error.to_string(),
                        })?;
                total_payload_bytes
                .checked_add(layer_payload_bytes)
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "complete expert payload byte count overflowed during model load",
                })
            })?;
    let largest_complete_expert_layer_bytes = expert_layer_plans
        .iter()
        .map(|layer_plan| {
            layer_plan
                .complete_expert_payload_byte_count()
                .map_err(|plan_error| Qwen3_5ExecutionError::ExpertLayerPlan {
                    description: plan_error.to_string(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let largest_routed_expert_page_bytes = expert_layer_plans.iter().try_fold(
        0_u64,
        |largest_page_bytes, layer_plan| -> Result<u64, Qwen3_5ExecutionError> {
            let complete_layer_payload_bytes = layer_plan
                .complete_expert_payload_byte_count()
                .map_err(|plan_error| Qwen3_5ExecutionError::ExpertLayerPlan {
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
    MlxRamBudget::new(
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
            sequence_state_bytes_per_token: u64::try_from(
                decoder_cache_layout
                    .sequence_state_payload_byte_count_per_token()
                    .unwrap_or(0),
            )
            .unwrap_or(0),
        },
    )
    .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
        description: "MLX RAM budget requires a positive active-memory ceiling",
    })
}

/// Derives the decoder cache layout at the tensor-binding boundary.
///
/// Cache geometry cannot be finalized before weights and sparse plans are
/// bound: their source dtypes determine MLX result-type promotion, so the
/// persistence contract is derived here without evaluating tensors. The
/// resident path derives dtypes from the quantized expert layer plans.
fn decoder_cache_layout_at_tensor_binding(
    config: &crate::Qwen3_5Config,
    chunking: &super::Qwen3_5ModelChunkingConfiguration,
    weights: &Qwen3_5Weights,
    expert_layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<crate::DecoderCacheLayout, Qwen3_5ExecutionError> {
    let decoder_layer_cache_dtypes =
        super::decoder_cache_dtype_flow::derive_decoder_layer_cache_dtypes(
            weights,
            expert_layer_plans,
        )?;
    crate::qwen3_5_core::decoder::qwen3_5_decoder_cache_layout(
        config,
        usize::try_from(chunking.full_attention_key_value_growth_tokens).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "full-attention key/value growth tokens exceed the usize range",
            }
        })?,
        &decoder_layer_cache_dtypes,
    )
    .map_err(
        |decoder_cache_layout_error| Qwen3_5ExecutionError::InvalidDecoderCacheLayout {
            description: decoder_cache_layout_error.to_string(),
        },
    )
}
