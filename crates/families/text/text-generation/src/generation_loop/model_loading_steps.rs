//! The model-loading steps every text engine take: starting the MLX
//! runtime, validating the chunking settings, resolving the model's identity
//! and visual contract, and opening the `astro_ssd_prompt_cache`. What each
//! engine loads around them (whole experts and the Neural Engine for
//! resident, the expert pager and residency facts for streaming) stays with
//! the engine.

use std::sync::Arc;

use ipc_protocol::WorkerChunkingConfiguration;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::engine_errors::{fatal_engine_error, fatal_runtime_error};
use crate::model::TextServingModel;
use crate::model::model_chunking_configuration::TextModelChunkingConfiguration;
use crate::{
    AstroDecoderStateCacheLayout, AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheDiskStoreConfig,
    AstroSsdPromptCacheModelContract, AstroSsdPromptCacheModelContractError, InferenceEngineError,
    PerformanceAttribution, PerformanceOperation, PersistentVisualEmbeddingModelContract,
};

use super::astro_ssd_prompt_cache_startup_logging;

/// Resolves the validated artifact identity the state fields report after
/// a successful load; losing either during loading is a fatal engine error.
pub fn resolve_validated_model_identity(
    model_id: &Option<String>,
    model_revision: &Option<String>,
) -> Result<(String, String), InferenceEngineError> {
    let resolved_model_id = model_id
        .clone()
        .ok_or_else(|| fatal_engine_error("model loading lost the validated model identifier"))?;
    let resolved_model_revision = model_revision
        .clone()
        .ok_or_else(|| fatal_engine_error("model loading lost the validated model revision"))?;
    Ok((resolved_model_id, resolved_model_revision))
}

/// Initializes the MLX runtime under attribution and immediately clears the
/// MLX buffer cache. Dropping the prior model moves its Metal buffers into
/// MLX's process-global allocator pool; releasing them before replacement
/// weights load keeps stale residency from competing during first-request
/// paging.
pub fn initialize_mlx_runtime_with_cleanup(
    memory_limits: MlxMemoryLimits,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxRuntime, InferenceEngineError> {
    let runtime = model_loading_performance_attribution
        .measure_operation(
            PerformanceOperation::MlxRuntimeInitialization,
            |_performance_attribution| MlxRuntime::initialize(memory_limits),
        )
        .map_err(fatal_runtime_error)?;
    model_loading_performance_attribution
        .measure_operation(
            PerformanceOperation::MlxBufferCacheCleanup,
            |_performance_attribution| runtime.clear_mlx_buffer_cache(),
        )
        .map_err(fatal_runtime_error)?;
    Ok(runtime)
}

/// Pure: validates the worker chunking settings into the model chunking
/// configuration both model loaders consume.
pub fn resolve_model_chunking_configuration(
    chunking: &WorkerChunkingConfiguration,
) -> Result<TextModelChunkingConfiguration, InferenceEngineError> {
    TextModelChunkingConfiguration::new(
        chunking.full_attention_key_value_growth_tokens,
        chunking.prefill_graph_submission_layer_interval,
        chunking.experimental_ssd_paging_generation_graph_submission_layer_interval,
    )
    .map_err(|configuration_error| {
        fatal_engine_error(format!(
            "failed to validate model chunking configuration: {configuration_error}"
        ))
    })
}

/// Sizes a model's persistent `astro_ssd_prompt_cache` contract from its
/// identity, `astro_decoder_state_cache` layout, position budget, active
/// memory limit, disk quota, configured block size and checkpoint stride, in
/// [`AstroSsdPromptCacheModelContract::resolve`]'s argument order.
pub type AstroSsdPromptCacheContractResolver =
    fn(
        String,
        String,
        AstroDecoderStateCacheLayout,
        usize,
        u64,
        u64,
        Option<usize>,
        u32,
    ) -> Result<AstroSsdPromptCacheModelContract, AstroSsdPromptCacheModelContractError>;

/// Resolves the persistent astro SSD prompt cache model-state storage contract from
/// the model's `astro_decoder_state_cache` layout, position budget, active memory limit,
/// and the request's chunking overrides.
fn resolve_astro_ssd_prompt_cache_model_contract(
    astro_ssd_prompt_cache_disk_store_config: &AstroSsdPromptCacheDiskStoreConfig,
    model: &impl TextServingModel,
    chunking: &WorkerChunkingConfiguration,
    resolved_model_id: &str,
    resolved_model_revision: &str,
    contract_resolver: AstroSsdPromptCacheContractResolver,
) -> Result<AstroSsdPromptCacheModelContract, InferenceEngineError> {
    let global_astro_ssd_prompt_cache_maximum_size_bytes =
        astro_ssd_prompt_cache_disk_store_config.global_astro_ssd_prompt_cache_maximum_size_bytes();
    contract_resolver(
        resolved_model_id.to_owned(),
        resolved_model_revision.to_owned(),
        model.astro_decoder_state_cache_layout().clone(),
        model.maximum_position_count() as usize,
        model.runtime().memory_limits().active_memory_limit_bytes() as u64,
        global_astro_ssd_prompt_cache_maximum_size_bytes,
        chunking
            .ssd_prompt_cache_block_tokens
            .map(|block_token_count| block_token_count as usize),
        chunking.ssd_prompt_cache_common_prefix_stride_blocks,
    )
    .map_err(|model_contract_error| {
        fatal_engine_error(format!(
            "could not resolve persistent model-state storage contract: {model_contract_error}"
        ))
    })
}

/// Opens the `astro_ssd_prompt_cache` when a disk store is configured; a
/// disabled `astro_ssd_prompt_cache` yields no contract and no store, so it cannot fail model
/// loading because of storage alignment, quota, stale files, or filesystem
/// availability (contract derivation is intentionally inside the same branch
/// as store ownership). The contract is sized by `contract_resolver`:
/// [`AstroSsdPromptCacheModelContract::resolve`] for most engines, and a
/// resolver that fits the contract to the disk quota for an engine whose
/// model needs that.
#[allow(clippy::too_many_arguments)]
pub fn open_astro_ssd_prompt_cache_with_contract_resolver(
    astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
    model: &impl TextServingModel,
    chunking: &WorkerChunkingConfiguration,
    resolved_model_id: &str,
    resolved_model_revision: &str,
    persistent_visual_embedding_model_contract: Option<&PersistentVisualEmbeddingModelContract>,
    contract_resolver: AstroSsdPromptCacheContractResolver,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<
    (
        Option<AstroSsdPromptCacheModelContract>,
        Option<Arc<AstroSsdPromptCacheDiskStore>>,
    ),
    InferenceEngineError,
> {
    let Some(astro_ssd_prompt_cache_disk_store_config) = astro_ssd_prompt_cache_disk_store_config
    else {
        return Ok((None, None));
    };
    let astro_ssd_prompt_cache_model_contract = resolve_astro_ssd_prompt_cache_model_contract(
        &astro_ssd_prompt_cache_disk_store_config,
        model,
        chunking,
        resolved_model_id,
        resolved_model_revision,
        contract_resolver,
    )?;
    let astro_ssd_prompt_cache = open_and_scan_astro_ssd_prompt_cache_store(
        astro_ssd_prompt_cache_disk_store_config,
        astro_ssd_prompt_cache_model_contract.clone(),
        persistent_visual_embedding_model_contract,
        model_loading_performance_attribution,
    )
    .map_err(fatal_engine_error)?;
    Ok((
        Some(astro_ssd_prompt_cache_model_contract),
        Some(astro_ssd_prompt_cache),
    ))
}

/// Opens the persistent astro SSD prompt cache disk store under attribution, scans the
/// required visual prompt-state entries, logs the startup cleanup and
/// `astro_ssd_prompt_cache` inventory, and hands back the shared store.
fn open_and_scan_astro_ssd_prompt_cache_store(
    astro_ssd_prompt_cache_disk_store_config: AstroSsdPromptCacheDiskStoreConfig,
    model_contract: AstroSsdPromptCacheModelContract,
    persistent_visual_embedding_model_contract: Option<&PersistentVisualEmbeddingModelContract>,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<Arc<AstroSsdPromptCacheDiskStore>, String> {
    let global_astro_ssd_prompt_cache_maximum_size_bytes =
        astro_ssd_prompt_cache_disk_store_config.global_astro_ssd_prompt_cache_maximum_size_bytes();
    let astro_ssd_prompt_cache = model_loading_performance_attribution
        .measure_operation(
            PerformanceOperation::AstroSsdPromptCacheOpenAndScan,
            |_performance_attribution| {
                AstroSsdPromptCacheDiskStore::open(
                    astro_ssd_prompt_cache_disk_store_config,
                    model_contract.clone(),
                )
            },
        )
        .map_err(|astro_ssd_prompt_cache_error| {
            format!("required target prompt-state storage initialization failed: {astro_ssd_prompt_cache_error}")
        })?;
    if let Some(persistent_visual_embedding_model_contract) =
        persistent_visual_embedding_model_contract
        && let Err(visual_embedding_scan_error) = astro_ssd_prompt_cache
            .scan_visual_embeddings(persistent_visual_embedding_model_contract)
    {
        return Err(format!(
            "required visual prompt-state storage scan failed: {visual_embedding_scan_error}"
        ));
    }
    astro_ssd_prompt_cache_startup_logging::log_astro_ssd_prompt_cache_startup_cleanup(
        "target",
        &astro_ssd_prompt_cache,
    );
    tracing::info!(
        sequence_state_block_count = astro_ssd_prompt_cache.sequence_state_block_count(),
        boundary_state_snapshot_count = astro_ssd_prompt_cache.boundary_state_snapshot_count(),
        total_size_bytes = astro_ssd_prompt_cache.total_size_bytes(),
        maximum_size_bytes = global_astro_ssd_prompt_cache_maximum_size_bytes,
        "opened astro-ssd-prompt-cache"
    );
    Ok(Arc::new(astro_ssd_prompt_cache))
}

#[cfg(test)]
#[path = "model_loading_steps_tests.rs"]
mod model_loading_steps_tests;
