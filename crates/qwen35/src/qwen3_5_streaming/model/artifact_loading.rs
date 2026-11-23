//! Artifact binding and startup-only MLX resource construction for Qwen3.5.

use crate::qwen3_5_streaming::model::route_observation::RouteObservationCollector;
use std::cell::RefCell;
use std::path::Path;

use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;
use runtime_integration::MlxRuntime;

use crate::qwen3_5_core::model::disk_embedding_rows::Qwen3_5DiskEmbeddingRows;
use crate::qwen3_5_core::model::loading_pipeline::{
    BoundLanguageArtifact, InvariantLinearScales, bind_language_artifact,
    load_invariant_linear_scales, load_model_kernel_verdicts,
};
use crate::qwen3_5_streaming;
use crate::qwen3_5_streaming::Qwen3_5ExpertPager;
use crate::qwen3_5_streaming::model::dense_layer_streamer::Qwen3_5DenseLayerStreamer;
use crate::{ModelMemoryFacts, ModelMemoryGeometry, PerformanceAttribution, PerformanceOperation};

use super::Qwen3_5Model;
use super::Qwen3_5ModelBase;
use super::direct_expert_fetch::DirectExpertFetching;
use super::{
    Qwen3_5ExecutionError, Qwen3_5FeedForwardArchitecture, Qwen3_5VisionModel, Qwen3_5Weights,
    ValidatedQwen3_5Artifact,
};
use mlx_bridge::{MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMetalKernel};

/// The streaming engine plans memory from the bytes on disk, so it binds
/// tensors exactly as stored.
const LANGUAGE_WEIGHT_BINDING_DTYPE: crate::qwen3_5_core::model_math::weights::WeightBindingDtype =
    crate::qwen3_5_core::model_math::weights::WeightBindingDtype::AsStored;
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
        Self::load_with_performance_attribution_and_decode_astro_expert_cache_policy(
            runtime,
            validated_artifact,
            model_directory,
            should_bind_vision_weights,
            chunking,
            WorkerDecodeAstroExpertCachePolicy::Baseline,
            performance_attribution,
        )
    }

    /// The decode-`astro_expert_cache`-policy-aware loader: binds the shared language
    /// artifact, builds the streaming expert runtime (pager, warm slot table,
    /// sorted-reduction kernel), derives the `astro_decoder_state_cache` layout, collects
    /// the capability-verdict kernels and invariant scales, and assembles the
    /// memory facts with the fused-kernel verdict.
    pub fn load_with_performance_attribution_and_decode_astro_expert_cache_policy(
        runtime: MlxRuntime,
        mut validated_artifact: ValidatedQwen3_5Artifact,
        model_directory: &Path,
        should_bind_vision_weights: bool,
        chunking: super::Qwen3_5ModelChunkingConfiguration,
        decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        // Both taken before binding, which consumes parts of the artifact.
        let expert_atlas = validated_artifact.expert_atlas().cloned();
        let (disk_embedding_rows, dense_layer_streamer) =
            dense_streaming_sources(&validated_artifact)?;
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
        let (expert_pager, astro_expert_cache, sorted_expert_weighted_sum_kernel) =
            match config.feed_forward_architecture() {
                Qwen3_5FeedForwardArchitecture::Dense => (None, None, None),
                Qwen3_5FeedForwardArchitecture::MixtureOfExperts => load_streaming_expert_runtime(
                    &runtime,
                    model_directory,
                    &config,
                    &shard_index,
                    &tensor_inventory,
                    expert_atlas.as_ref(),
                    decode_astro_expert_cache_policy,
                    performance_attribution,
                )?,
            };
        let astro_decoder_state_cache_layout = measured_astro_decoder_state_cache_layout(
            &config,
            &chunking,
            &weights,
            expert_pager.as_ref(),
            performance_attribution,
        )?;
        let kernel_verdicts =
            load_model_kernel_verdicts(&runtime, &config, performance_attribution)?;
        let (compiled_swiglu, compiled_elementwise_graphs) = compiled_model_graphs()?;
        let InvariantLinearScales {
            inverse_linear_head_dimension_scale,
            inverse_square_root_linear_head_dimension_scale,
            query_normalization_scale_weight,
            key_normalization_scale_weight,
        } = load_invariant_linear_scales(&runtime, &config, LANGUAGE_WEIGHT_BINDING_DTYPE)?;
        let streaming_expert_budget = build_streaming_expert_budget(
            &runtime,
            &config,
            &weights,
            embedding_residency_of(disk_embedding_rows.as_ref()),
            &vision_model,
            expert_pager.as_ref(),
            dense_layer_streamer.as_ref(),
            &astro_decoder_state_cache_layout,
            performance_attribution,
        )?;
        Ok(Self {
            base: Qwen3_5ModelBase {
                runtime,
                config,
                astro_decoder_state_cache_layout,
                weights,
                disk_embedding_rows,
                vision_model,
                memory_facts: RefCell::new(streaming_expert_budget.memory_facts),
                gated_delta_kernel: kernel_verdicts.gated_delta_kernel,
                gated_delta_checkpoint_kernel: kernel_verdicts.gated_delta_checkpoint_kernel,
                gdn_decode_prework_kernel: kernel_verdicts.gdn_decode_prework_kernel,
                optional_accelerators: Default::default(),
                sorted_expert_weighted_sum_kernel,
                compiled_swiglu,
                compiled_elementwise_graphs,
                chunking,
                inverse_linear_head_dimension_scale,
                inverse_square_root_linear_head_dimension_scale,
                query_normalization_scale_weight,
                key_normalization_scale_weight,
                layer_additions: RefCell::new(None),
            },
            direct_expert_fetching: DirectExpertFetching::open(expert_pager.as_ref()),
            expert_pager,
            dense_layer_streamer,
            astro_expert_cache,
            brings_every_route_to_the_host: std::cell::Cell::new(false),
            route_observation: RefCell::new(RouteObservationCollector::new()),
            deferred_expert_demand: RefCell::default(),
            prompt_expert_pages_in_hand: Default::default(),
            prefill_layer_completion_refused: RefCell::default(),
            active_expert_residency_plan: RefCell::new(None),
            paged_forward_missing_route_collector:
                crate::qwen3_5_streaming::PagedForwardMissingRouteCollector::default(),
            hot_expert_warm_slot_count: std::cell::Cell::new(0),
            optimistic_serving_history: Default::default(),
            request_expert_storage_read_tally: Default::default(),
            governed_memory: Default::default(),
            probed_prompt_layer_mlx_working_buffers: std::cell::Cell::new(None),
            fused_jangtq2_gathered_projection_kernel: streaming_expert_budget
                .fused_jangtq2_gathered_projection_kernel,
        })
    }
}

/// The compiled SwiGLU and elementwise graphs every forward reuses.
fn compiled_model_graphs()
-> Result<(MlxCompiledSwiGlu, MlxCompiledElementwiseGraphs), Qwen3_5ExecutionError> {
    Ok((
        MlxCompiledSwiGlu::new()?,
        MlxCompiledElementwiseGraphs::new()?,
    ))
}

/// Derives the `astro_decoder_state_cache` layout under attribution. It waits for bound
/// weights and sparse plans, whose source dtypes decide MLX result-type
/// promotion, and evaluates no tensor.
fn measured_astro_decoder_state_cache_layout(
    config: &crate::Qwen3_5Config,
    chunking: &super::Qwen3_5ModelChunkingConfiguration,
    weights: &Qwen3_5Weights,
    expert_pager: Option<&Qwen3_5ExpertPager>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<crate::AstroDecoderStateCacheLayout, Qwen3_5ExecutionError> {
    performance_attribution.measure_operation(
        PerformanceOperation::ModelTensorBinding,
        |_performance_attribution| {
            astro_decoder_state_cache_layout_at_tensor_binding(
                config,
                chunking,
                weights,
                expert_pager,
            )
        },
    )
}

/// A streaming dense model's core and streaming workspace.
struct DenseStreamingGeometry {
    model_core_payload_bytes: u64,
    dense_streaming_workspace_bytes: u64,
}

/// Leaves a dense model's decoder layers out of its fixed core: kept layers
/// are given back under request pressure like retained experts, so only the
/// tensors that never stream are fixed. The two layer slots a streamed
/// forward reads into become the per-operation page the floor reserves once.
/// They stay out of the complete-expert-layer field, whose prefill rule
/// would reserve three of it; prefill MLX working buffers are measured instead.
fn dense_streaming_geometry(
    bound_model_core_payload_bytes: u64,
    dense_layer_streamer: Option<&Qwen3_5DenseLayerStreamer>,
) -> DenseStreamingGeometry {
    let Some(dense_layer_streamer) = dense_layer_streamer else {
        return DenseStreamingGeometry {
            model_core_payload_bytes: bound_model_core_payload_bytes,
            dense_streaming_workspace_bytes: 0,
        };
    };
    let mut decoder_layer_payload_bytes = 0_u64;
    for layer_payload_bytes in dense_layer_streamer.layer_payload_bytes() {
        decoder_layer_payload_bytes =
            decoder_layer_payload_bytes.saturating_add(*layer_payload_bytes);
    }
    DenseStreamingGeometry {
        model_core_payload_bytes: bound_model_core_payload_bytes
            .saturating_sub(decoder_layer_payload_bytes),
        dense_streaming_workspace_bytes: dense_layer_streamer.streaming_slot_bytes(),
    }
}

/// A dense model's on-disk embedding rows and decoder layer streamer, both
/// planned from the validated headers before binding takes the sources.
fn dense_streaming_sources(
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<
    (
        Option<Qwen3_5DiskEmbeddingRows>,
        Option<Qwen3_5DenseLayerStreamer>,
    ),
    Qwen3_5ExecutionError,
> {
    Ok((
        dense_disk_embedding_rows(validated_artifact)?,
        dense_layer_streamer(validated_artifact)?,
    ))
}

/// The decoder layer streamer of a dense model, which the streaming engine
/// serves only when its text weights do not fit the limit.
fn dense_layer_streamer(
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<Option<Qwen3_5DenseLayerStreamer>, Qwen3_5ExecutionError> {
    match validated_artifact.config().feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => {
            Qwen3_5DenseLayerStreamer::from_validated_artifact(validated_artifact).map(Some)
        }
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => Ok(None),
    }
}

/// Whether the loaded model reads embedding rows from the SSD.
const fn embedding_residency_of(
    disk_embedding_rows: Option<&Qwen3_5DiskEmbeddingRows>,
) -> crate::qwen3_5_core::model_math::weights::EmbeddingResidency {
    match disk_embedding_rows {
        Some(_) => crate::qwen3_5_core::model_math::weights::EmbeddingResidency::RowsReadFromDisk,
        None => crate::qwen3_5_core::model_math::weights::EmbeddingResidency::Resident,
    }
}

/// Embedding rows read from the SSD for a dense model, which the streaming
/// engine serves only when its text weights do not fit the limit; a sparse
/// model keeps its resident table, which its expert budget already plans.
fn dense_disk_embedding_rows(
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<Option<Qwen3_5DiskEmbeddingRows>, Qwen3_5ExecutionError> {
    match validated_artifact.config().feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => {
            Qwen3_5DiskEmbeddingRows::from_validated_artifact(validated_artifact)
        }
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => Ok(None),
    }
}

/// The streaming engine's MoE-only runtime trio: pager, warm slot table, and
/// the sorted-reduction kernel verdict.
type StreamingExpertRuntimeTrio = (
    Option<Qwen3_5ExpertPager>,
    Option<RefCell<qwen3_5_streaming::AstroExpertCache>>,
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
    expert_atlas: Option<&expert_atlas::AtlasManifest>,
    decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
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
                expert_atlas,
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
    let astro_expert_cache = RefCell::new(
        qwen3_5_streaming::AstroExpertCache::new_with_decode_astro_expert_cache_policy(
            expert_pager.layer_count(),
            decode_astro_expert_cache_policy,
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
            astro_expert_cache
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
        Some(astro_expert_cache),
        sorted_expert_weighted_sum_kernel,
    ))
}

/// The streaming memory facts plus the fused jangtq2 verdict that the model
/// struct also carries.
struct StreamingExpertBudgetOutcome {
    memory_facts: ModelMemoryFacts,
    fused_jangtq2_gathered_projection_kernel:
        Option<jangtq2_qwen35::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel>,
}

/// Assembles the limit and geometry the streaming engine runs under. Mirrors the payload math,
/// fused-kernel verdict, routed-page sizing, and the prefill read-ahead slot
/// of the inlined loader, in the same order.
#[allow(clippy::too_many_arguments)]
fn build_streaming_expert_budget(
    runtime: &MlxRuntime,
    config: &crate::Qwen3_5Config,
    weights: &Qwen3_5Weights,
    embedding_residency: crate::qwen3_5_core::model_math::weights::EmbeddingResidency,
    vision_model: &Option<Qwen3_5VisionModel>,
    expert_pager: Option<&Qwen3_5ExpertPager>,
    dense_layer_streamer: Option<&Qwen3_5DenseLayerStreamer>,
    astro_decoder_state_cache_layout: &crate::AstroDecoderStateCacheLayout,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<StreamingExpertBudgetOutcome, Qwen3_5ExecutionError> {
    let DenseStreamingGeometry {
        model_core_payload_bytes,
        dense_streaming_workspace_bytes,
    } = dense_streaming_geometry(
        crate::qwen3_5_core::model::loading_pipeline::model_core_payload_bytes(
            weights,
            embedding_residency,
            vision_model,
        ),
        dense_layer_streamer,
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
    let largest_routed_expert_page_bytes =
        expert_pager.map_or(Ok(dense_streaming_workspace_bytes), |expert_pager| {
            // The load-time budget is taken before the fused kernel
            // verdict, so it keeps the conservative decode peak.
            expert_pager.maximum_routed_expert_page_bytes(
                usize::try_from(config.experts_per_token()).unwrap_or(usize::MAX),
                true,
            )
        })?;
    let memory_facts = ModelMemoryFacts::new(
        u64::try_from(runtime.memory_limits().active_memory_limit_bytes()).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "MLX active memory ceiling exceeds the u64 range",
            }
        })?,
        ModelMemoryGeometry {
            model_core_payload_bytes,
            complete_expert_payload_bytes,
            largest_complete_expert_layer_bytes,
            largest_routed_expert_page_bytes,
            sequence_state_bytes_per_token: u64::try_from(
                astro_decoder_state_cache_layout
                    .sequence_state_payload_byte_count_per_token()
                    .unwrap_or(0),
            )
            .unwrap_or(0),
        },
    )
    .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
        description: "the MLX memory limit must be positive",
    })?;
    Ok(StreamingExpertBudgetOutcome {
        memory_facts,
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
) -> Option<jangtq2_qwen35::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel> {
    let has_codebook_experts = expert_pager.layer_plans().iter().any(|layer_plan| {
        ["gate_proj", "up_proj", "down_proj"]
            .iter()
            .any(|projection_name| {
                layer_plan.quantization_mode_for_projection(projection_name)
                    == engine_core::expert_paging::QuantizationMode::Jangtq2Codebook
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
    match jangtq2_qwen35::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel::new() {
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

/// Derives the `astro_decoder_state_cache` layout at the tensor-binding boundary.
///
/// The `astro_decoder_state_cache` geometry cannot be finalized before weights and sparse plans are
/// bound: their source dtypes determine MLX result-type promotion, so the
/// persistence contract is derived here without evaluating tensors.
fn astro_decoder_state_cache_layout_at_tensor_binding(
    config: &crate::Qwen3_5Config,
    chunking: &super::Qwen3_5ModelChunkingConfiguration,
    weights: &Qwen3_5Weights,
    expert_pager: Option<&Qwen3_5ExpertPager>,
) -> Result<crate::AstroDecoderStateCacheLayout, Qwen3_5ExecutionError> {
    let astro_decoder_state_cache_layer_dtypes =
        crate::qwen3_5_core::model::bound_weight_dtype_flow::derive_astro_decoder_state_cache_layer_dtypes(
            weights,
            // A sparse layer without a pager finds no plan here, so the
            // derivation still fails, now with the shared "no layer plan" text.
            expert_pager.map_or(&[], Qwen3_5ExpertPager::layer_plans),
        )?;
    crate::qwen3_5_core::decoder::qwen3_5_astro_decoder_state_cache_layout(
        config,
        usize::try_from(chunking.full_attention_key_value_growth_tokens).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "full-attention key/value growth tokens exceed the usize range",
            }
        })?,
        &astro_decoder_state_cache_layer_dtypes,
    )
    .map_err(|astro_decoder_state_cache_layout_error| {
        Qwen3_5ExecutionError::InvalidAstroDecoderStateCacheLayout {
            description: astro_decoder_state_cache_layout_error.to_string(),
        }
    })
}
