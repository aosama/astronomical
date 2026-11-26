//! Artifact binding and startup-only MLX resource construction for Qwen3.5.

use crate::qwen3_5_text::model::route_observation::RouteObservationCollector;
use std::cell::RefCell;
use std::path::Path;

use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;
use runtime_integration::MlxRuntime;

use crate::qwen3_5_core::model::disk_embedding_rows::Qwen3_5DiskEmbeddingRows;
use crate::qwen3_5_core::model::loading_geometry::{
    ExpertPayloadGeometry, measured_astro_decoder_state_cache_layout, model_memory_facts,
};
use crate::qwen3_5_core::model::loading_pipeline::{
    BoundLanguageArtifact, InvariantLinearScales, ModelKernelVerdicts, bind_language_artifact,
    load_invariant_linear_scales, load_model_kernel_verdicts,
};
use crate::qwen3_5_text;
use crate::qwen3_5_text::Qwen3_5ExpertPager;
use crate::qwen3_5_text::model::dense_layer_streamer::Qwen3_5DenseLayerStreamer;
use crate::qwen4_exp::ngram_table::{
    NgramTable, NgramTableShards, ngram_table_from_bound_weights, open_ngram_table_shards,
};
use crate::{ModelMemoryFacts, PerformanceAttribution};

use super::Qwen3_5Model;
use super::Qwen3_5ModelBase;
use super::dense_streaming_loading::{
    DenseStreamingGeometry, dense_streaming_geometry, dense_streaming_sources,
    embedding_residency_of,
};
use super::direct_expert_fetch::DirectExpertFetching;
use super::streaming_expert_runtime_loading::{
    ExpertCombinationKernels, accelerators_with, streaming_expert_runtime,
};
use super::{Qwen3_5VisionModel, Qwen3_5Weights, TextExecutionError, ValidatedQwen3_5Artifact};
use mlx_bridge::{MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu};

/// Tensors bind in the dtype the compute device runs the model accurately
/// in (#2441): as stored on the Metal build, so memory plans sized from the
/// bytes on disk hold there, and float32 for half-precision tensors where MLX
/// computes on its CPU, whose BF16 arithmetic changes the model's answers.
const LANGUAGE_WEIGHT_BINDING_DTYPE: crate::qwen3_5_core::model_math::weights::WeightBindingDtype =
    crate::qwen3_5_core::model_math::weights::WeightBindingDtype::ForComputeDevice;
impl Qwen3_5Model {
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
    /// Every sparse model constructs a Qwen3_5ExpertPager at startup for prefill and
    /// decode. The `model_directory` must point to the directory containing the
    /// safetensors shard files so the pager can build bounded byte-range plans
    /// from the same validated inventory the weights bound.
    pub fn load_with_performance_attribution(
        runtime: MlxRuntime,
        validated_artifact: ValidatedQwen3_5Artifact,
        model_directory: &Path,
        should_bind_vision_weights: bool,
        chunking: super::TextModelChunkingConfiguration,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, TextExecutionError> {
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
        chunking: super::TextModelChunkingConfiguration,
        decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, TextExecutionError> {
        let (bound_sources, ngram_table_shards) = bind_model_sources(
            &runtime,
            &mut validated_artifact,
            model_directory,
            should_bind_vision_weights,
            decode_astro_expert_cache_policy,
            performance_attribution,
        )?;
        let astro_decoder_state_cache_layout = measured_astro_decoder_state_cache_layout(
            &bound_sources.config,
            &chunking,
            &bound_sources.weights,
            pager_layer_plans(bound_sources.expert_pager.as_ref()),
            performance_attribution,
        )?;
        let ngram_table = ngram_table_from_bound_weights(
            ngram_table_shards,
            &bound_sources.config,
            &bound_sources.weights,
        )?;
        let startup_kernels = load_startup_kernel_resources(
            &runtime,
            &bound_sources.config,
            performance_attribution,
        )?;
        let streaming_expert_budget = build_streaming_expert_budget(
            &runtime,
            &bound_sources.config,
            &bound_sources.weights,
            embedding_residency_of(bound_sources.disk_embedding_rows.as_ref()),
            &bound_sources.vision_model,
            bound_sources.expert_pager.as_ref(),
            bound_sources.dense_layer_streamer.as_ref(),
            &astro_decoder_state_cache_layout,
            performance_attribution,
        )?;
        Ok(Self::assemble_loaded_model(
            runtime,
            chunking,
            LoadedModelStages {
                bound_sources,
                astro_decoder_state_cache_layout,
                ngram_table,
                startup_kernels,
                streaming_expert_budget,
            },
        ))
    }

    /// Moves every loaded piece into the model. Each resource was built by an
    /// earlier stage of the load, so nothing here returns an error or reads
    /// the GPU. The one step that touches storage is
    /// `DirectExpertFetching::open`, which opens the expert readers and, when
    /// they do not open, logs a warning and leaves the experts on the copying
    /// path instead of failing the load.
    fn assemble_loaded_model(
        runtime: MlxRuntime,
        chunking: super::TextModelChunkingConfiguration,
        loaded_stages: LoadedModelStages,
    ) -> Self {
        let LoadedModelStages {
            bound_sources,
            astro_decoder_state_cache_layout,
            ngram_table,
            startup_kernels,
            streaming_expert_budget,
        } = loaded_stages;
        let BoundModelSources {
            config,
            weights,
            vision_model,
            disk_embedding_rows,
            dense_layer_streamer,
            expert_pager,
            astro_expert_cache,
            expert_kernels,
        } = bound_sources;
        let StartupKernelResources {
            invariant_linear_scales,
            kernel_verdicts,
            compiled_swiglu,
            compiled_elementwise_graphs,
        } = startup_kernels;
        Self {
            base: Qwen3_5ModelBase {
                runtime,
                config,
                astro_decoder_state_cache_layout,
                weights,
                disk_embedding_rows,
                ngram_table,
                vision_model,
                memory_facts: RefCell::new(streaming_expert_budget.memory_facts),
                gated_delta_kernel: kernel_verdicts.gated_delta_kernel,
                gated_delta_checkpoint_kernel: kernel_verdicts.gated_delta_checkpoint_kernel,
                gdn_decode_prework_kernel: kernel_verdicts.gdn_decode_prework_kernel,
                optional_accelerators: accelerators_with(expert_kernels.sorted_shared_combination),
                sorted_expert_weighted_sum_kernel: expert_kernels.sorted_weighted_sum,
                compiled_swiglu,
                compiled_elementwise_graphs,
                chunking,
                inverse_linear_head_dimension_scale: invariant_linear_scales
                    .inverse_linear_head_dimension_scale,
                inverse_square_root_linear_head_dimension_scale: invariant_linear_scales
                    .inverse_square_root_linear_head_dimension_scale,
                query_normalization_scale_weight: invariant_linear_scales
                    .query_normalization_scale_weight,
                key_normalization_scale_weight: invariant_linear_scales
                    .key_normalization_scale_weight,
                layer_additions: RefCell::new(None),
            },
            direct_expert_fetching: DirectExpertFetching::open(expert_pager.as_ref()),
            expert_pager,
            dense_layer_streamer,
            astro_expert_cache,
            reply_measurement: Default::default(),
            route_observation: RefCell::new(RouteObservationCollector::new()),
            deferred_expert_demand: RefCell::default(),
            prompt_expert_pages_in_hand: Default::default(),
            prefill_layer_completion_refused: RefCell::default(),
            active_expert_residency_plan: RefCell::new(None),
            paged_forward_missing_route_collector:
                crate::qwen3_5_text::PagedForwardMissingRouteCollector::default(),
            hot_expert_warm_slot_count: std::cell::Cell::new(0),
            optimistic_serving_history: Default::default(),
            request_expert_storage_read_tally: Default::default(),
            governed_memory: Default::default(),
            probed_prompt_layer_mlx_working_buffers: std::cell::Cell::new(None),
            fused_jangtq2_gathered_projection_kernel: streaming_expert_budget
                .fused_jangtq2_gathered_projection_kernel,
        }
    }
}

/// What binding the artifact and planning the streaming runtime produce,
/// before the memory facts and kernels that depend on them are derived.
struct BoundModelSources {
    config: crate::Qwen3_5Config,
    weights: Qwen3_5Weights,
    vision_model: Option<Qwen3_5VisionModel>,
    disk_embedding_rows: Option<Qwen3_5DiskEmbeddingRows>,
    dense_layer_streamer: Option<Qwen3_5DenseLayerStreamer>,
    expert_pager: Option<Qwen3_5ExpertPager>,
    astro_expert_cache: Option<RefCell<qwen3_5_text::AstroExpertCache>>,
    expert_kernels: ExpertCombinationKernels,
}

/// The kernels, compiled graphs and invariant scales every forward reuses,
/// none of which depend on the weights.
struct StartupKernelResources {
    invariant_linear_scales: InvariantLinearScales,
    kernel_verdicts: ModelKernelVerdicts,
    compiled_swiglu: MlxCompiledSwiGlu,
    compiled_elementwise_graphs: MlxCompiledElementwiseGraphs,
}

/// The stages of a load, gathered for the final assembly of the model.
struct LoadedModelStages {
    bound_sources: BoundModelSources,
    astro_decoder_state_cache_layout: crate::AstroDecoderStateCacheLayout,
    ngram_table: Option<NgramTable>,
    startup_kernels: StartupKernelResources,
    streaming_expert_budget: StreamingExpertBudgetOutcome,
}

/// Binds the language artifact and plans how its experts, embedding rows and
/// decoder layers stream. The n-gram table shards come back beside the bound
/// sources because the caller builds the table from them once the weights are
/// bound.
fn bind_model_sources(
    runtime: &MlxRuntime,
    validated_artifact: &mut ValidatedQwen3_5Artifact,
    model_directory: &Path,
    should_bind_vision_weights: bool,
    decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(BoundModelSources, Option<NgramTableShards>), TextExecutionError> {
    // Both taken before binding, which consumes parts of the artifact.
    let expert_atlas = validated_artifact.expert_atlas().cloned();
    let (disk_embedding_rows, dense_layer_streamer) = dense_streaming_sources(validated_artifact)?;
    let ngram_table_shards = open_ngram_table_shards(validated_artifact)?;
    let bound_language = bind_language_artifact(
        runtime,
        validated_artifact,
        should_bind_vision_weights,
        performance_attribution,
    )?;
    let (expert_pager, astro_expert_cache, expert_kernels) = streaming_expert_runtime(
        runtime,
        model_directory,
        &bound_language.config,
        &bound_language.shard_index,
        &bound_language.tensor_inventory,
        expert_atlas.as_ref(),
        decode_astro_expert_cache_policy,
        performance_attribution,
    )?;
    let BoundLanguageArtifact {
        config,
        weights,
        vision_model,
        ..
    } = bound_language;
    let bound_sources = BoundModelSources {
        config,
        weights,
        vision_model,
        disk_embedding_rows,
        dense_layer_streamer,
        expert_pager,
        astro_expert_cache,
        expert_kernels,
    };
    Ok((bound_sources, ngram_table_shards))
}

/// Loads the capability-verdict kernels, the compiled graphs and the
/// invariant scales, in that order.
fn load_startup_kernel_resources(
    runtime: &MlxRuntime,
    config: &crate::Qwen3_5Config,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<StartupKernelResources, TextExecutionError> {
    let kernel_verdicts = load_model_kernel_verdicts(runtime, config, performance_attribution)?;
    let (compiled_swiglu, compiled_elementwise_graphs) = compiled_model_graphs()?;
    let invariant_linear_scales =
        load_invariant_linear_scales(runtime, config, LANGUAGE_WEIGHT_BINDING_DTYPE)?;
    Ok(StartupKernelResources {
        invariant_linear_scales,
        kernel_verdicts,
        compiled_swiglu,
        compiled_elementwise_graphs,
    })
}

/// The compiled SwiGLU and elementwise graphs every forward reuses.
fn compiled_model_graphs()
-> Result<(MlxCompiledSwiGlu, MlxCompiledElementwiseGraphs), TextExecutionError> {
    Ok((
        MlxCompiledSwiGlu::new()?,
        MlxCompiledElementwiseGraphs::new()?,
    ))
}

/// The pager's expert layer plans, or none for a dense model. A sparse layer
/// without a pager finds no plan, so the layout derivation still fails, with
/// the shared "no layer plan" text.
fn pager_layer_plans(
    expert_pager: Option<&Qwen3_5ExpertPager>,
) -> &[crate::qwen3_5_core::artifacts::QuantizedExpertLayerPlan] {
    expert_pager.map_or(&[], Qwen3_5ExpertPager::layer_plans)
}

/// The streaming memory facts plus the fused jangtq2 verdict that the model
/// struct also carries.
struct StreamingExpertBudgetOutcome {
    memory_facts: ModelMemoryFacts,
    fused_jangtq2_gathered_projection_kernel:
        Option<jangtq2_qwen35::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel>,
}

/// Assembles the limit and geometry the engine runs under. Mirrors the payload math,
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
) -> Result<StreamingExpertBudgetOutcome, TextExecutionError> {
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
            .map_err(|_| TextExecutionError::InvalidInput {
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
            .map_err(|_| TextExecutionError::InvalidInput {
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
    let memory_facts = model_memory_facts(
        runtime,
        model_core_payload_bytes,
        ExpertPayloadGeometry {
            complete_expert_payload_bytes,
            largest_complete_expert_layer_bytes,
            largest_routed_expert_page_bytes,
        },
        astro_decoder_state_cache_layout,
    )?;
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
    .is_custom_kernel_supported(crate::kernel_capability::FUSED_JANGTQ2_GATHERED_PROJECTION)
    {
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
