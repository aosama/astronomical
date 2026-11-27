//! Loading a model into any text engine: the steps all of them take, in
//! the order all of them take them, with the engine's own work at three
//! points.
//!
//! An engine that already holds a model reports it as loaded. Otherwise the
//! validated artifact moves into a freshly loaded model, the
//! `astro_ssd_prompt_cache` opens beside it, the engine prepares what else the
//! model needs, and the model is installed. The loading attribution is
//! recorded whether the load succeeded or failed.

use crate::model::TextServingModel;
use std::sync::Arc;

use runtime_integration::{MlxMemorySnapshot, MlxRuntime};

use crate::engine_errors::fatal_engine_error;
use crate::inference_engine::load_result::EngineLoadResult;
use crate::model::model_chunking_configuration::TextModelChunkingConfiguration;
use crate::{
    AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheModelContract, InferenceEngineError,
    ModelLoadingPerformanceAttributionMetadata, PerformanceAttribution,
    PerformanceAttributionOutcome, PersistentVisualEmbeddingModelContract,
};

use super::generation_finalization_steps::TextFinalizationHooks;
use super::model_loading_steps::{
    AstroSsdPromptCacheContractResolver, initialize_mlx_runtime_with_cleanup,
    open_astro_ssd_prompt_cache_with_contract_resolver, resolve_model_chunking_configuration,
    resolve_validated_model_identity,
};
use super::serving_artifact::TextServingArtifact;
use crate::model::text_model::TextModel;
use engine_core::astro_ssd_prompt_cache::{
    AstroSsdPromptCachePublicationQueue, MAXIMUM_QUEUED_PUBLICATION_PAYLOAD_BYTES,
};

/// The points in a load where the engines differ. The Qwen3.5 text engine
/// opens its memory governor as the model loads; the folded-ternary engine
/// sizes its `astro_ssd_prompt_cache` contract to the disk quota.
pub trait TextModelLoadingHooks: TextFinalizationHooks {
    /// Sizes the model's `astro_ssd_prompt_cache` contract when a disk store
    /// is configured.
    const ASTRO_SSD_PROMPT_CACHE_CONTRACT_RESOLVER: AstroSsdPromptCacheContractResolver =
        AstroSsdPromptCacheModelContract::resolve;

    /// Loads the model from the artifact on `runtime` and materializes its
    /// weights.
    fn load_and_materialize_model(
        &mut self,
        runtime: MlxRuntime,
        validated_artifact: Self::Artifact,
        model_chunking: TextModelChunkingConfiguration,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self::Model, InferenceEngineError>;

    /// Prepares the rest of the model once its `astro_ssd_prompt_cache` is
    /// open.
    fn prepare_loaded_model(
        &mut self,
        model: &mut Self::Model,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError>;

    /// Takes in the model the engine is about to hold.
    fn adopt_loaded_model(&mut self, model: &Self::Model);

    /// What the worker learns from a finished load.
    fn engine_load_result(&self, minimum_mlx_memory_ceiling_bytes: u64) -> EngineLoadResult;
}

/// What the artifact said about itself before loading went further; a
/// failed load still records whatever was read.
#[derive(Clone, Debug, Default)]
struct ModelLoadingArtifactFacts {
    model_id: Option<String>,
    model_revision: Option<String>,
    total_artifact_payload_bytes: Option<u64>,
    model_shard_count: Option<usize>,
}

/// What a successful load hands to installation.
struct LoadedModelParts<Model> {
    model: Model,
    astro_ssd_prompt_cache_model_contract: Option<AstroSsdPromptCacheModelContract>,
    persistent_visual_embedding_model_contract: Option<PersistentVisualEmbeddingModelContract>,
    astro_ssd_prompt_cache: Option<Arc<AstroSsdPromptCacheDiskStore>>,
}

/// Loads the engine's model, or reports the one it already holds.
pub fn load_model<Engine: TextModelLoadingHooks>(
    engine: &mut Engine,
) -> Result<EngineLoadResult, InferenceEngineError> {
    if let Some(model) = engine.execution_core().model.as_ref() {
        return Ok(engine.engine_load_result(model.minimum_mlx_memory_ceiling_bytes()?));
    }
    let mut model_loading_performance_attribution = engine
        .execution_core_mut()
        .take_model_loading_performance_attribution();
    let mut artifact_facts = ModelLoadingArtifactFacts::default();
    match load_model_parts(
        engine,
        &mut artifact_facts,
        &mut model_loading_performance_attribution,
    ) {
        Ok(loaded_model_parts) => install_loaded_model(
            engine,
            loaded_model_parts,
            model_loading_performance_attribution,
            artifact_facts,
        ),
        Err(model_loading_error) => {
            record_model_loading_performance_attribution(
                engine,
                model_loading_performance_attribution,
                ModelLoadingRecord {
                    outcome: PerformanceAttributionOutcome::Failed,
                    artifact_facts,
                    resident_model_payload_bytes: None,
                    mlx_memory_snapshot: None,
                    failure_description: Some(model_loading_error.to_string()),
                },
            );
            Err(model_loading_error)
        }
    }
}

fn load_model_parts<Engine: TextModelLoadingHooks>(
    engine: &mut Engine,
    artifact_facts: &mut ModelLoadingArtifactFacts,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<LoadedModelParts<Engine::Model>, InferenceEngineError> {
    let validated_artifact = engine
        .execution_core_mut()
        .validated_artifact
        .take()
        .ok_or_else(|| fatal_engine_error("validated artifact is unavailable during MLX load"))?;
    let persistent_visual_embedding_model_contract =
        validated_artifact.persistent_visual_embedding_model_contract();
    *artifact_facts = ModelLoadingArtifactFacts {
        model_id: Some(validated_artifact.model_id().to_owned()),
        model_revision: Some(validated_artifact.revision().to_owned()),
        total_artifact_payload_bytes: Some(validated_artifact.total_payload_bytes()),
        model_shard_count: Some(validated_artifact.shard_count()),
    };
    let runtime = initialize_mlx_runtime_with_cleanup(
        engine.execution_core().memory_limits,
        model_loading_performance_attribution,
    )?;
    let model_chunking = resolve_model_chunking_configuration(&engine.execution_core().chunking)?;
    let mut model = engine.load_and_materialize_model(
        runtime,
        validated_artifact,
        model_chunking,
        model_loading_performance_attribution,
    )?;
    // Loading may bound the `mlx_buffer_cache` to what the model measured;
    // the core keeps the installed limits, so a reload installs the same.
    engine.execution_core_mut().memory_limits = model.runtime().memory_limits();
    let (resolved_model_id, resolved_model_revision) =
        resolve_validated_model_identity(&artifact_facts.model_id, &artifact_facts.model_revision)?;
    let execution_core = engine.execution_core();
    let (astro_ssd_prompt_cache_model_contract, astro_ssd_prompt_cache) =
        open_astro_ssd_prompt_cache_with_contract_resolver(
            execution_core
                .astro_ssd_prompt_cache_disk_store_config
                .clone(),
            &model,
            &execution_core.chunking,
            &resolved_model_id,
            &resolved_model_revision,
            persistent_visual_embedding_model_contract.as_ref(),
            Engine::ASTRO_SSD_PROMPT_CACHE_CONTRACT_RESOLVER,
            model_loading_performance_attribution,
        )?;
    engine.prepare_loaded_model(&mut model, model_loading_performance_attribution)?;
    Ok(LoadedModelParts {
        model,
        astro_ssd_prompt_cache_model_contract,
        persistent_visual_embedding_model_contract,
        astro_ssd_prompt_cache,
    })
}

/// Moves the loaded model and its `astro_ssd_prompt_cache` ownership into the
/// engine, records the successful loading attribution, and reports the load.
fn install_loaded_model<Engine: TextModelLoadingHooks>(
    engine: &mut Engine,
    loaded_model_parts: LoadedModelParts<Engine::Model>,
    model_loading_performance_attribution: PerformanceAttribution,
    artifact_facts: ModelLoadingArtifactFacts,
) -> Result<EngineLoadResult, InferenceEngineError> {
    let LoadedModelParts {
        model,
        astro_ssd_prompt_cache_model_contract,
        persistent_visual_embedding_model_contract,
        astro_ssd_prompt_cache,
    } = loaded_model_parts;
    let execution_core = engine.execution_core_mut();
    execution_core.model_id = artifact_facts.model_id.clone();
    execution_core.model_revision = artifact_facts.model_revision.clone();
    execution_core.astro_ssd_prompt_cache_model_contract = astro_ssd_prompt_cache_model_contract;
    execution_core.persistent_visual_embedding_model_contract =
        persistent_visual_embedding_model_contract;
    // The old writer, if any, is dropped first so its queued blocks reach the
    // old store before the new one takes over.
    execution_core.astro_ssd_prompt_cache_publication_queue = None;
    execution_core.astro_ssd_prompt_cache_publication_queue =
        started_publication_queue(astro_ssd_prompt_cache.as_ref());
    execution_core.astro_ssd_prompt_cache = astro_ssd_prompt_cache;
    let mlx_memory_snapshot = model.runtime().memory_snapshot().ok();
    let resident_model_payload_bytes = Some(model.resident_model_payload_byte_count());
    let minimum_mlx_memory_ceiling_bytes = model.minimum_mlx_memory_ceiling_bytes()?;
    engine.adopt_loaded_model(&model);
    engine.execution_core_mut().model = Some(model);
    record_model_loading_performance_attribution(
        engine,
        model_loading_performance_attribution,
        ModelLoadingRecord {
            outcome: PerformanceAttributionOutcome::Success,
            artifact_facts,
            resident_model_payload_bytes,
            mlx_memory_snapshot,
            failure_description: None,
        },
    );
    Ok(engine.engine_load_result(minimum_mlx_memory_ceiling_bytes))
}

/// How a load ended, for its attribution report.
struct ModelLoadingRecord {
    outcome: PerformanceAttributionOutcome,
    artifact_facts: ModelLoadingArtifactFacts,
    resident_model_payload_bytes: Option<u64>,
    mlx_memory_snapshot: Option<MlxMemorySnapshot>,
    failure_description: Option<String>,
}

/// The background writer for `astro_ssd_prompt_cache`, or `None` when no
/// store is open or the operating system refuses the thread; blocks are then
/// written on the engine thread, as before the writer existed.
fn started_publication_queue(
    astro_ssd_prompt_cache: Option<&Arc<AstroSsdPromptCacheDiskStore>>,
) -> Option<AstroSsdPromptCachePublicationQueue> {
    let astro_ssd_prompt_cache = astro_ssd_prompt_cache?;
    match AstroSsdPromptCachePublicationQueue::start(
        Arc::clone(astro_ssd_prompt_cache),
        MAXIMUM_QUEUED_PUBLICATION_PAYLOAD_BYTES,
    ) {
        Ok(publication_queue) => Some(publication_queue),
        Err(thread_start_error) => {
            tracing::warn!(
                %thread_start_error,
                "astro_ssd_prompt_cache writer did not start; blocks are written while the prompt is read"
            );
            None
        }
    }
}

/// Finishes the loading attribution and appends it to the log. A write that
/// fails is logged and never masks how the load ended.
fn record_model_loading_performance_attribution<Engine: TextModelLoadingHooks>(
    engine: &mut Engine,
    model_loading_performance_attribution: PerformanceAttribution,
    model_loading_record: ModelLoadingRecord,
) {
    let measured_prompt_mlx_working_buffer_bytes =
        engine.measured_prompt_mlx_working_buffer_bytes();
    let ModelLoadingRecord {
        outcome,
        artifact_facts,
        resident_model_payload_bytes,
        mlx_memory_snapshot,
        failure_description,
    } = model_loading_record;
    let Some(performance_attribution_report) = model_loading_performance_attribution
        .finish_model_loading(ModelLoadingPerformanceAttributionMetadata {
            outcome,
            model_id: artifact_facts.model_id,
            model_revision: artifact_facts.model_revision,
            prefill_transient_observation_completed: measured_prompt_mlx_working_buffer_bytes
                .is_some(),
            prefill_observed_transient_high_water_bytes: measured_prompt_mlx_working_buffer_bytes
                .unwrap_or(0),
            total_artifact_payload_bytes: artifact_facts.total_artifact_payload_bytes,
            resident_model_payload_bytes,
            model_shard_count: artifact_facts.model_shard_count,
            mlx_active_memory_bytes: mlx_memory_snapshot
                .as_ref()
                .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
            mlx_buffer_cache_bytes: mlx_memory_snapshot
                .as_ref()
                .and_then(|snapshot| u64::try_from(snapshot.mlx_buffer_cache_bytes()).ok()),
            mlx_peak_memory_bytes: mlx_memory_snapshot
                .as_ref()
                .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
            failure_description,
        })
    else {
        return;
    };
    if let Err(performance_attribution_write_error) = engine
        .execution_core_mut()
        .performance_attribution_log
        .record(&performance_attribution_report)
    {
        tracing::warn!(
            error = %performance_attribution_write_error,
            "failed to append model-loading performance attribution"
        );
    }
}
