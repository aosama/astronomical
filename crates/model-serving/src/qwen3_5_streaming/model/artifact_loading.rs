//! Artifact binding and startup-only MLX resource construction for Qwen3.5.

use std::cell::RefCell;
use std::path::Path;

use astronomical_ipc_protocol::WorkerDecodeExpertCachePolicy;
use astronomical_runtime_integration::MlxRuntime;

use crate::qwen3_5_core::model::loading_pipeline::{
    BoundLanguageArtifact, InvariantLinearScales, ModelKernelVerdicts, bind_language_artifact,
    load_invariant_linear_scales, load_model_kernel_verdicts,
};
use crate::qwen3_5_streaming;
use crate::qwen3_5_streaming::Qwen3_5ExpertPager;
use crate::{
    MlxRamBudget, MlxRamBudgetModelGeometry, PerformanceAttribution, PerformanceOperation,
};

use super::Qwen3_5Model;
use super::Qwen3_5ModelBase;
use super::{
    Qwen3_5ExecutionError, Qwen3_5FeedForwardArchitecture, Qwen3_5VisionModel, Qwen3_5Weights,
    ValidatedQwen3_5Artifact,
};
use mlx_c_rust::{MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMetalKernel};

impl Qwen3_5Model {
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
    /// Every sparse model constructs a Qwen3_5ExpertPager at startup for prefill and
    /// decode. The `model_directory` must point to the directory containing the
    /// safetensors shard files so the pager can build bounded byte-range plans
    /// from the same validated inventory the weights bound.
    pub fn load_with_performance_attribution(
        runtime: MlxRuntime,
        validated_artifact: ValidatedQwen3_5Artifact,
        model_directory: &Path,
        should_bind_vision_weights: bool,
        chunking: super::Qwen3_5ModelChunkingConfiguration,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        Self::load_with_performance_attribution_and_decode_expert_cache_policy(
            runtime,
            validated_artifact,
            model_directory,
            should_bind_vision_weights,
            chunking,
            WorkerDecodeExpertCachePolicy::Baseline,
            performance_attribution,
        )
    }

    /// The decode-cache-policy-aware loader: binds the shared language
    /// artifact, builds the streaming expert runtime (pager, warm slot table,
    /// sorted-reduction kernel), derives the decoder cache layout, collects
    /// the capability-verdict kernels and invariant scales, and assembles the
    /// RAM budget with the fused-kernel verdict.
    pub(crate) fn load_with_performance_attribution_and_decode_expert_cache_policy(
        runtime: MlxRuntime,
        mut validated_artifact: ValidatedQwen3_5Artifact,
        model_directory: &Path,
        should_bind_vision_weights: bool,
        chunking: super::Qwen3_5ModelChunkingConfiguration,
        decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
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
        let (expert_pager, retained_experts, sorted_expert_weighted_sum_kernel) =
            match config.feed_forward_architecture() {
                Qwen3_5FeedForwardArchitecture::Dense => (None, None, None),
                Qwen3_5FeedForwardArchitecture::MixtureOfExperts => load_streaming_expert_runtime(
                    &runtime,
                    model_directory,
                    &config,
                    &shard_index,
                    &tensor_inventory,
                    decode_expert_cache_policy,
                    performance_attribution,
                )?,
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
                    expert_pager.as_ref(),
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
        let StreamingExpertBudgetOutcome {
            mlx_ram_budget,
            fused_jangtq2_gathered_projection_kernel,
        } = build_streaming_expert_budget(
            &runtime,
            &config,
            &weights,
            &vision_model,
            expert_pager.as_ref(),
            &decoder_cache_layout,
            performance_attribution,
        )?;
        Ok(Self {
            base: Qwen3_5ModelBase {
                runtime,
                config,
                decoder_cache_layout,
                weights,
                vision_model,
                mlx_ram_budget: RefCell::new(mlx_ram_budget),
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
            expert_pager,
            retained_experts,
            route_observation: RefCell::new(
                crate::qwen3_5_streaming::model::route_observation::RouteObservationCollector::new(
                ),
            ),
            deferred_expert_demand: RefCell::default(),
            prefill_complete_layer_prefetch: RefCell::default(),
            active_expert_residency_plan: RefCell::new(None),
            request_expert_residency: RefCell::new(None),
            paged_forward_missing_route_collector:
                crate::qwen3_5_streaming::PagedForwardMissingRouteCollector,
            hot_expert_warm_slot_count: std::cell::Cell::new(0),
            request_expert_storage_read_tally:
                super::request_expert_storage_read_tally::RequestExpertStorageReadTally::default(),
            fused_jangtq2_gathered_projection_kernel,
        })
    }
}

/// The streaming engine's MoE-only runtime trio: pager, warm slot table, and
/// the sorted-reduction kernel verdict.
type StreamingExpertRuntimeTrio = (
    Option<Qwen3_5ExpertPager>,
    Option<RefCell<qwen3_5_streaming::WarmExpertSlotTableCache>>,
    Option<MlxMetalKernel>,
);

/// Builds the streaming expert runtime for a sparse model. Mirrors the MoE
/// branch of the inlined loader, including the warm-table seeding from the
/// persisted `.coli_usage`-analog history.
#[allow(clippy::too_many_arguments)]
fn load_streaming_expert_runtime(
    runtime: &MlxRuntime,
    model_directory: &Path,
    config: &crate::Qwen3_5Config,
    shard_index: &super::Qwen3_5ShardIndex,
    tensor_inventory: &crate::TensorInventory,
    decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<StreamingExpertRuntimeTrio, Qwen3_5ExecutionError> {
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
    let expert_pager = performance_attribution.measure_operation(
        crate::PerformanceOperation::ExpertPagerPlanConstruction,
        |_performance_attribution| {
            Qwen3_5ExpertPager::new(
                runtime,
                model_directory.to_path_buf(),
                &tensor_name_to_shard_file_name,
                &stored_tensor_name_by_canonical_name,
                config,
                // The live configurable MLX active-memory ceiling is the only
                // paging budget. Reading the runtime policy here lets reloads
                // apply on every machine without copying a stale wired limit.
                runtime.memory_limits().active_memory_limit_bytes(),
            )
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
            Some(qwen3_5_streaming::qwen3_5_moe_sorted_expert_weighted_sum_kernel()?)
        } else {
            tracing::info!(
                "sorted expert weighted-sum kernel demoted to the MLX fallback for this worker process"
            );
            None
        };
    let retained_experts = RefCell::new(
        qwen3_5_streaming::WarmExpertSlotTableCache::new_with_decode_cache_policy(
            expert_pager.layer_count(),
            decode_expert_cache_policy,
        ),
    );
    // Seed demand ranking from the persisted usage history beside
    // the artifact (colibri's `.coli_usage` analog). Only the decode record
    // seeds hotness; a missing or stale file simply leaves the session cold.
    let usage_file_path =
        qwen3_5_streaming::expert_streaming::expert_usage_file_path(model_directory);
    match qwen3_5_streaming::expert_streaming::read_expert_usage(
        &usage_file_path,
        expert_pager.layer_count(),
    ) {
        Some(persisted_usage) => {
            retained_experts
                .borrow_mut()
                .seed_expert_demand_history(persisted_usage);
        }
        None => tracing::info!(
            usage_file_path = usage_file_path.display().to_string(),
            "no reusable expert usage history; warm tables start cold"
        ),
    }
    Ok((
        Some(expert_pager),
        Some(retained_experts),
        sorted_expert_weighted_sum_kernel,
    ))
}

/// The streaming RAM budget plus the fused jangtq2 verdict that the model
/// struct also carries.
struct StreamingExpertBudgetOutcome {
    mlx_ram_budget: MlxRamBudget,
    fused_jangtq2_gathered_projection_kernel: Option<
        astronomical_jangtq2_qwen35::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel,
    >,
}

/// Assembles the streaming engine's RAM budget. Mirrors the payload math,
/// fused-kernel verdict, routed-page sizing, and the prefill read-ahead slot
/// of the inlined loader, in the same order.
#[allow(clippy::too_many_arguments)]
fn build_streaming_expert_budget(
    runtime: &MlxRuntime,
    config: &crate::Qwen3_5Config,
    weights: &Qwen3_5Weights,
    vision_model: &Option<Qwen3_5VisionModel>,
    expert_pager: Option<&Qwen3_5ExpertPager>,
    decoder_cache_layout: &crate::DecoderCacheLayout,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<StreamingExpertBudgetOutcome, Qwen3_5ExecutionError> {
    let model_core_payload_bytes =
        crate::qwen3_5_core::model::loading_pipeline::model_core_payload_bytes(
            weights,
            vision_model,
        );
    let complete_expert_payload_bytes = match expert_pager {
        Some(expert_pager) => expert_pager
            .complete_expert_payload_byte_count()
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "complete expert payload byte count overflowed during model load",
            })?,
        None => 0,
    };
    // The RAM owner's three-layer prefill floor is based on packed source
    // payload geometry. The separately composed expert-page reservation
    // carries the decoded bfloat16 output a forward materializes.
    let largest_complete_expert_layer_bytes = match expert_pager {
        Some(expert_pager) => expert_pager
            .maximum_complete_expert_source_payload_bytes()
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "complete expert source layer byte count overflowed during model load",
            })?,
        None => 0,
    };
    let fused_jangtq2_gathered_projection_kernel = expert_pager.and_then(|expert_pager| {
        fused_jangtq2_gathered_projection_kernel_for_pager(
            runtime,
            expert_pager,
            performance_attribution,
        )
    });
    if let Some(expert_pager) = expert_pager {
        expert_pager.set_decode_tokens_multiply_packed_pages(
            fused_jangtq2_gathered_projection_kernel.is_some(),
        );
    }
    let largest_routed_expert_page_bytes = expert_pager.map_or(Ok(0), |expert_pager| {
        // The load-time budget is taken before the fused kernel
        // verdict, so it keeps the conservative decode peak.
        expert_pager.maximum_routed_expert_page_bytes(
            usize::try_from(config.experts_per_token()).unwrap_or(usize::MAX),
            true,
        )
    })?;
    let mut mlx_ram_budget = MlxRamBudget::new(
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
    })?;
    if let Some(expert_pager) = expert_pager {
        // A multi-token forward decodes the complete-layer page it
        // multiplies, so the prefill plan must keep that page's peak
        // (packed source plus decoded output) out of the expert budget;
        // the packed-source slot alone seated experts into the decode's
        // space and every chunk had to reclaim a layer before it could run.
        // The second slot funds the one complete layer a chunk reads ahead
        // of its route while earlier layers compute
        // (`prefill_complete_layer_prefetch`). That prefetch is only issued
        // for pages multiplied without a transient decode, so it holds the
        // packed source alone, and only of layers it can pick: codebook
        // layers are never eligible in prefill, so an all-codebook bundle
        // reserves no read-ahead bytes at all. Charging it the full
        // decoded peak instead doubled a 5.74 GB JANGTQ2 slot to 11.48 GB
        // and cut 122B prefill throughput nearly in two (measured
        // 2026-10-24: 134 versus 232–273 tokens per second).
        mlx_ram_budget.set_prefill_complete_layer_stream_slot_bytes(
            super::prefill_complete_layer_prefetch::prefill_complete_layer_stream_slot_bytes(
                expert_pager.maximum_expert_page_bytes(),
                super::prefill_complete_layer_prefetch::largest_prefill_prefetchable_complete_source_payload_bytes(
                    expert_pager.layer_plans(),
                )
                .map_err(qwen3_5_streaming::expert_streaming::ExpertPagingError::from)?,
            ),
        );
    }
    Ok(StreamingExpertBudgetOutcome {
        mlx_ram_budget,
        fused_jangtq2_gathered_projection_kernel,
    })
}

/// Builds the fused jangtq2 decode-token projection kernel when the pager has
/// codebook experts and the worker's capability verdict allows it.
/// A demoted kernel leaves `None`, and
/// decode tokens then decode a transient bfloat16 page exactly like prefill.
fn fused_jangtq2_gathered_projection_kernel_for_pager(
    runtime: &MlxRuntime,
    expert_pager: &Qwen3_5ExpertPager,
    performance_attribution: &mut PerformanceAttribution,
) -> Option<
    astronomical_jangtq2_qwen35::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel,
> {
    let has_codebook_experts = expert_pager.layer_plans().iter().any(|layer_plan| {
        ["gate_proj", "up_proj", "down_proj"]
            .iter()
            .any(|projection_name| {
                layer_plan.quantization_mode_for_projection(projection_name)
                    == crate::expert_paging::QuantizationMode::Jangtq2Codebook
            })
    });
    if !has_codebook_experts {
        return None;
    }
    if !crate::kernel_capability::worker_process_kernel_capabilities(
        runtime,
        performance_attribution,
    )
    .is_custom_kernel_supported(
        crate::kernel_capability::CustomMetalKernelFamily::FUSED_JANGTQ2_GATHERED_PROJECTION,
    ) {
        tracing::info!(
            "fused jangtq2 gathered projection kernel demoted to the materializing decoder for this worker process"
        );
        return None;
    }
    match astronomical_jangtq2_qwen35::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel::new() {
        Ok(kernel) => Some(kernel),
        Err(error) => {
            tracing::warn!(
                %error,
                "fused jangtq2 gathered projection kernel failed to compile after a passing probe; using the materializing decoder"
            );
            None
        }
    }
}

/// Derives the decoder cache layout at the tensor-binding boundary.
///
/// Cache geometry cannot be finalized before weights and sparse plans are
/// bound: their source dtypes determine MLX result-type promotion, so the
/// persistence contract is derived here without evaluating tensors.
fn decoder_cache_layout_at_tensor_binding(
    config: &crate::Qwen3_5Config,
    chunking: &super::Qwen3_5ModelChunkingConfiguration,
    weights: &Qwen3_5Weights,
    expert_pager: Option<&Qwen3_5ExpertPager>,
) -> Result<crate::DecoderCacheLayout, Qwen3_5ExecutionError> {
    let decoder_layer_cache_dtypes =
        super::decoder_cache_dtype_flow::derive_decoder_layer_cache_dtypes(weights, expert_pager)?;
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
