//! The expert runtime a sparse model is loaded with: its pager, its
//! `astro_expert_cache` and the kernels that reduce a layer's routed experts.

use std::cell::RefCell;
use std::path::Path;

use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;
use runtime_integration::MlxRuntime;

use crate::PerformanceAttribution;
use crate::qwen3_5_core::model::loading_geometry::{ExpertTensorNameMaps, expert_tensor_name_maps};
use crate::qwen3_5_text;
use crate::qwen3_5_text::Qwen3_5ExpertPager;

use super::{Qwen3_5FeedForwardArchitecture, TextExecutionError};
use mlx_bridge::MlxMetalKernel;

/// The engine's MoE-only runtime trio: pager, warm slot table, and
/// the kernels that reduce a layer's routed experts.
type StreamingExpertRuntimeTrio = (
    Option<Qwen3_5ExpertPager>,
    Option<RefCell<qwen3_5_text::AstroExpertCache>>,
    ExpertCombinationKernels,
);

/// The expert runtime of a sparse model; a dense model streams its layers,
/// not experts, and has none.
#[allow(clippy::too_many_arguments)]
pub(super) fn streaming_expert_runtime(
    runtime: &MlxRuntime,
    model_directory: &Path,
    config: &crate::Qwen3_5Config,
    shard_index: &super::Qwen3_5ShardIndex,
    tensor_inventory: &crate::TensorInventory,
    expert_atlas: Option<&expert_atlas::AtlasManifest>,
    decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<StreamingExpertRuntimeTrio, TextExecutionError> {
    if config.feed_forward_architecture() == Qwen3_5FeedForwardArchitecture::Dense {
        return Ok((None, None, ExpertCombinationKernels::default()));
    }
    load_streaming_expert_runtime(
        runtime,
        model_directory,
        config,
        shard_index,
        tensor_inventory,
        expert_atlas,
        decode_astro_expert_cache_policy,
        performance_attribution,
    )
}

/// Builds the streaming expert runtime for a sparse model, including the
/// warm-table seeding from the persisted `.coli_usage`-analog history.
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
) -> Result<StreamingExpertRuntimeTrio, TextExecutionError> {
    let ExpertTensorNameMaps {
        tensor_name_to_shard_file_name,
        stored_tensor_name_by_canonical_name,
    } = expert_tensor_name_maps(shard_index, tensor_inventory);
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
    let expert_combination_kernels =
        kept_expert_combination_kernels(runtime, performance_attribution)?;
    let astro_expert_cache = RefCell::new(
        qwen3_5_text::AstroExpertCache::new_with_decode_astro_expert_cache_policy(
            expert_pager.layer_count(),
            decode_astro_expert_cache_policy,
        ),
    );
    // Seed demand ranking from the persisted usage history beside
    // the artifact (colibri's `.coli_usage` analog). Only the decode record
    // seeds hotness; a missing or stale file simply leaves the session cold.
    let usage_file_path = qwen3_5_text::expert_streaming::expert_usage_file_path(model_directory);
    match qwen3_5_text::expert_streaming::read_expert_usage(
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
        expert_combination_kernels,
    ))
}

/// The kernels a layer's routed experts run through beyond MLX's own, kept
/// by the GPU's capability verdicts.
#[derive(Default)]
pub(super) struct ExpertCombinationKernels {
    /// The sorted router-weighted sum. A demoted kernel leaves `None` and
    /// every MoE forward takes the unsorted MLX route for the worker's
    /// lifetime.
    pub(super) sorted_weighted_sum: Option<MlxMetalKernel>,
    /// The sorted sum joined with the gated shared expert in one pass, for
    /// prompt chunks. A chunk's routed rows are the same whether its experts
    /// were held or streamed, so this engine keeps it as the in-memory
    /// engine does (issue #2585).
    pub(super) sorted_shared_combination: Option<MlxMetalKernel>,
    /// The sorted-expert matmul for a prompt chunk's gate and up projections.
    /// It reads the sorted order, so it is kept only with the sorted sum.
    pub(super) sorted_expert_matmul: Option<crate::sorted_expert_matmul::SortedExpertMatmulKernels>,
    /// The tiled sorted-expert kernels for a prompt chunk's gate, up, their
    /// SwiGLU and down. They read the sorted order too, so they are kept only
    /// with the sorted sum.
    pub(super) sorted_expert_tiled: Option<crate::sorted_expert_matmul::TiledExpertKernels>,
}

/// The model's optional accelerators with only the expert kernels and the
/// signed rotation set.
pub(super) fn accelerators_with(
    sorted_expert_shared_combination_kernel: Option<MlxMetalKernel>,
    (sorted_expert_matmul_kernels, sorted_expert_tiled_kernels): (
        Option<crate::sorted_expert_matmul::SortedExpertMatmulKernels>,
        Option<crate::sorted_expert_matmul::TiledExpertKernels>,
    ),
    signed_hadamard_rotation_kernel: Option<MlxMetalKernel>,
) -> crate::qwen3_5_core::model::base::Qwen3_5OptionalAccelerators {
    crate::qwen3_5_core::model::base::Qwen3_5OptionalAccelerators {
        sorted_expert_shared_combination_kernel,
        sorted_expert_matmul_kernels,
        sorted_expert_tiled_kernels,
        signed_hadamard_rotation_kernel,
        ..Default::default()
    }
}

fn kept_expert_combination_kernels(
    runtime: &MlxRuntime,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<ExpertCombinationKernels, TextExecutionError> {
    let worker_kernel_capabilities = crate::kernel_capability::worker_process_kernel_capabilities(
        runtime,
        performance_attribution,
    );
    let sorted_weighted_sum = if worker_kernel_capabilities
        .is_custom_kernel_supported(crate::kernel_capability::SORTED_EXPERT_WEIGHTED_SUM)
    {
        Some(qwen3_5_text::qwen3_5_moe_sorted_expert_weighted_sum_kernel()?)
    } else {
        tracing::info!(
            "sorted expert weighted-sum kernel demoted to the MLX fallback for this worker process"
        );
        None
    };
    let sorted_shared_combination =
        crate::qwen3_5_core::model::sorted_expert_shared_combination_switch::retained_sorted_expert_shared_combination_kernel(
            worker_kernel_capabilities,
            sorted_weighted_sum.is_some(),
        )?;
    let sorted_expert_matmul =
        kept_sorted_expert_matmul(worker_kernel_capabilities, sorted_weighted_sum.is_some());
    let sorted_expert_tiled =
        kept_sorted_expert_tiled(worker_kernel_capabilities, sorted_weighted_sum.is_some());
    Ok(ExpertCombinationKernels {
        sorted_weighted_sum,
        sorted_shared_combination,
        sorted_expert_matmul,
        sorted_expert_tiled,
    })
}

#[cfg(test)]
#[path = "streaming_expert_runtime_loading_tests.rs"]
mod streaming_expert_runtime_loading_tests;

/// The tiled sorted-expert kernels of every weight width whose probe
/// reproduced MLX's bits on this GPU, when the sorted route they read is
/// kept; `None` when no width passed or that route is not kept, and prompt
/// chunks keep the routes before them. A stack of a width not kept is
/// refused chunk by chunk and keeps those routes too.
fn kept_sorted_expert_tiled(
    worker_kernel_capabilities: &crate::kernel_capability::WorkerKernelCapabilities,
    sorted_route_kept: bool,
) -> Option<crate::sorted_expert_matmul::TiledExpertKernels> {
    let mut proven_weight_bits = Vec::new();
    for (family, weight_bits) in crate::kernel_capability::SORTED_EXPERT_TILED_MATMUL_FAMILIES {
        if worker_kernel_capabilities.is_custom_kernel_supported(family) {
            proven_weight_bits.push(weight_bits);
        }
    }
    if !sorted_route_kept || proven_weight_bits.is_empty() {
        tracing::info!(
            "tiled sorted-expert kernels not kept: prompt chunks keep the sorted-expert matmul and MLX's down"
        );
        return None;
    }
    match crate::sorted_expert_matmul::TiledExpertKernels::declare_for_weight_bits(
        &proven_weight_bits,
    ) {
        Ok(kernels) => Some(kernels),
        Err(declaration_error) => {
            tracing::warn!(
                "tiled sorted-expert kernels could not be declared: {declaration_error}"
            );
            None
        }
    }
}

/// The sorted-expert matmul's kernels when this GPU reproduced MLX's bits
/// with them and the sorted route they read is kept; `None` otherwise, and
/// prompt chunks then keep MLX's gather_qmm for gate and up.
fn kept_sorted_expert_matmul(
    worker_kernel_capabilities: &crate::kernel_capability::WorkerKernelCapabilities,
    sorted_route_kept: bool,
) -> Option<crate::sorted_expert_matmul::SortedExpertMatmulKernels> {
    if !sorted_route_kept
        || !worker_kernel_capabilities
            .is_custom_kernel_supported(crate::kernel_capability::SORTED_EXPERT_MATMUL)
    {
        tracing::info!(
            "sorted-expert matmul not kept: prompt chunks use MLX's gather_qmm for gate and up"
        );
        return None;
    }
    match crate::sorted_expert_matmul::SortedExpertMatmulKernels::declare() {
        Ok(kernels) => Some(kernels),
        Err(declaration_error) => {
            tracing::warn!(
                "sorted-expert matmul kernels could not be declared: {declaration_error}"
            );
            None
        }
    }
}
