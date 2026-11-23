//! The state both Qwen3.5 engines keep between engine steps.
//!
//! The resident and streaming engines held the same 23 fields under the same
//! names and built them the same way from a validated artifact. They now
//! share one [`Qwen3_5ExecutionCore`], and each engine adds only the fields
//! of its own features (Neural Engine and drafting for resident, residency
//! facts for streaming). Steps that read only this state live here once.

use std::path::PathBuf;
use std::sync::Arc;

use ipc_protocol::{WorkerChunkingConfiguration, WorkerEvent};
use runtime_integration::MlxMemoryLimits;

use crate::astro_ssd_prompt_cache::counters::AstroSsdPromptCacheCounters;
use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_core::engine_errors::{fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5_core::model::base::Qwen3_5ModelBase;
use crate::{
    AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheDiskStoreConfig,
    AstroSsdPromptCacheModelContract, InferenceEngineError, PerformanceAttribution,
    PerformanceAttributionLog, PersistentVisualEmbeddingModelContract,
    build_astro_ssd_prompt_cache_stats_event,
};

use super::generation_request_validation::Qwen3_5GenerationRequestLimits;
use super::prompt_processing_chunk_sizer::Qwen3_5PromptProcessingChunkSizer;

pub struct Qwen3_5ExecutionCore<Model, RequestExtension> {
    pub active_request: Option<Qwen3_5EngineRequest<RequestExtension>>,
    pub model: Option<Model>,
    pub validated_artifact: Option<ValidatedQwen3_5Artifact>,
    /// Directory containing the safetensors model shards.
    pub model_directory: PathBuf,
    pub model_id: Option<String>,
    pub model_revision: Option<String>,
    pub end_of_sequence_token_ids: Vec<u32>,
    /// Token ID that closes the thinking block. Used to enforce thinking_budget.
    pub think_end_token_id: u32,
    pub vocabulary_size: u32,
    /// The configured context limit; advisory, and advertised unchanged.
    pub maximum_position_count: usize,
    /// Model artifact's native position range: the only hard rejection
    /// boundary for request context.
    pub hard_maximum_position_count: usize,
    pub context_memory_reservation_bytes_per_token: usize,
    /// Capacity-growth granularity for full-attention KV slabs.
    pub full_attention_kv_state_growth_tokens: i32,
    pub memory_limits: MlxMemoryLimits,
    pub prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
    pub chunking: WorkerChunkingConfiguration,
    pub model_loading_performance_attribution: Option<PerformanceAttribution>,
    pub performance_attribution_log: PerformanceAttributionLog,
    pub astro_ssd_prompt_cache: Option<Arc<AstroSsdPromptCacheDiskStore>>,
    pub astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
    pub astro_ssd_prompt_cache_counters: AstroSsdPromptCacheCounters,
    pub astro_ssd_prompt_cache_model_contract: Option<AstroSsdPromptCacheModelContract>,
    pub persistent_visual_embedding_model_contract: Option<PersistentVisualEmbeddingModelContract>,
}

/// An engine that keeps a [`Qwen3_5ExecutionCore`], so code shared by both
/// engines can reach the core of either.
pub trait Qwen3_5ExecutionCoreOwner {
    type Model: std::ops::Deref<Target = Qwen3_5ModelBase>;
    type RequestExtension;

    fn execution_core(&self) -> &Qwen3_5ExecutionCore<Self::Model, Self::RequestExtension>;

    fn execution_core_mut(
        &mut self,
    ) -> &mut Qwen3_5ExecutionCore<Self::Model, Self::RequestExtension>;
}

/// What the worker resolves before an engine starts.
pub struct Qwen3_5ExecutionCoreSettings {
    pub validated_artifact: ValidatedQwen3_5Artifact,
    pub active_memory_limit_bytes: usize,
    pub mlx_buffer_cache_limit_bytes: usize,
    pub astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
    pub prompt_processing_chunk_sizer: Qwen3_5PromptProcessingChunkSizer,
    pub think_end_token_id: u32,
    pub model_directory: PathBuf,
    pub maximum_context_tokens: u32,
    pub chunking: WorkerChunkingConfiguration,
    pub model_loading_performance_attribution: PerformanceAttribution,
    pub performance_attribution_log: PerformanceAttributionLog,
}

impl Qwen3_5ExecutionCoreSettings {
    /// Derives the engine limits from the artifact, refusing a configured
    /// context that is zero or larger than the artifact allows. Runs before
    /// the MLX owner thread starts, so a bad setting fails the engine's
    /// construction instead of its first request.
    pub fn validate(self) -> Result<Qwen3_5ValidatedExecutionCoreSettings, InferenceEngineError> {
        let full_attention_kv_state_growth_tokens = checked_full_attention_growth_tokens(
            self.chunking.full_attention_key_value_growth_tokens,
        )?;
        let artifact_maximum_position_count =
            self.validated_artifact.config().maximum_position_count();
        check_configured_context(self.maximum_context_tokens, artifact_maximum_position_count)?;
        let context_memory_reservation_bytes_per_token = self
            .validated_artifact
            .config()
            .context_memory_reservation_bytes(1)
            .ok_or_else(|| fatal_engine_error("Qwen3.5 context memory reservation overflowed"))?;
        let memory_limits = MlxMemoryLimits::new(
            self.active_memory_limit_bytes,
            self.mlx_buffer_cache_limit_bytes,
        )
        .map_err(qwen3_5_runtime_error)?;
        Ok(Qwen3_5ValidatedExecutionCoreSettings {
            hard_maximum_position_count: artifact_maximum_position_count as usize,
            context_memory_reservation_bytes_per_token,
            full_attention_kv_state_growth_tokens,
            memory_limits,
            settings: self,
        })
    }
}

/// The full-attention KV growth step as MLX takes it, an `i32`.
pub fn checked_full_attention_growth_tokens(
    full_attention_key_value_growth_tokens: u32,
) -> Result<i32, InferenceEngineError> {
    i32::try_from(full_attention_key_value_growth_tokens)
        .map_err(|_| fatal_engine_error("full-attention growth tokens exceed Int32 range"))
}

/// The configured context must be positive and no larger than the artifact's
/// own position range, which is the only hard limit a request is held to.
pub fn check_configured_context(
    maximum_context_tokens: u32,
    artifact_maximum_position_count: u32,
) -> Result<(), InferenceEngineError> {
    if maximum_context_tokens == 0 || maximum_context_tokens > artifact_maximum_position_count {
        return Err(fatal_engine_error(
            "effective Qwen3.5 context must be positive and not exceed the artifact context",
        ));
    }
    Ok(())
}

/// Settings that passed [`Qwen3_5ExecutionCoreSettings::validate`], with the
/// limits derived from them. They hold no MLX handle, so they can move onto
/// the MLX owner thread, where the core is built.
pub struct Qwen3_5ValidatedExecutionCoreSettings {
    settings: Qwen3_5ExecutionCoreSettings,
    hard_maximum_position_count: usize,
    context_memory_reservation_bytes_per_token: usize,
    full_attention_kv_state_growth_tokens: i32,
    memory_limits: MlxMemoryLimits,
}

impl<Model, RequestExtension> Qwen3_5ExecutionCore<Model, RequestExtension> {
    /// Builds the core on the MLX owner thread, with no model loaded yet.
    pub fn new(validated_settings: Qwen3_5ValidatedExecutionCoreSettings) -> Self {
        let Qwen3_5ValidatedExecutionCoreSettings {
            settings,
            hard_maximum_position_count,
            context_memory_reservation_bytes_per_token,
            full_attention_kv_state_growth_tokens,
            memory_limits,
        } = validated_settings;
        let artifact_config = settings.validated_artifact.config();
        Self {
            active_request: None,
            model: None,
            end_of_sequence_token_ids: artifact_config.end_of_sequence_token_ids().to_vec(),
            think_end_token_id: settings.think_end_token_id,
            vocabulary_size: artifact_config.vocabulary_size(),
            maximum_position_count: settings.maximum_context_tokens as usize,
            hard_maximum_position_count,
            context_memory_reservation_bytes_per_token,
            full_attention_kv_state_growth_tokens,
            memory_limits,
            validated_artifact: Some(settings.validated_artifact),
            model_directory: settings.model_directory,
            model_id: None,
            model_revision: None,
            prompt_processing_chunk_sizer: settings.prompt_processing_chunk_sizer,
            chunking: settings.chunking,
            model_loading_performance_attribution: Some(
                settings.model_loading_performance_attribution,
            ),
            performance_attribution_log: settings.performance_attribution_log,
            astro_ssd_prompt_cache: None,
            astro_ssd_prompt_cache_disk_store_config: settings
                .astro_ssd_prompt_cache_disk_store_config,
            astro_ssd_prompt_cache_counters: AstroSsdPromptCacheCounters::default(),
            astro_ssd_prompt_cache_model_contract: None,
            persistent_visual_embedding_model_contract: None,
        }
    }

    /// The attribution recorded while the engine was built, taken once by the
    /// model load that finishes it; disabled when it was already taken.
    pub fn take_model_loading_performance_attribution(&mut self) -> PerformanceAttribution {
        self.model_loading_performance_attribution
            .take()
            .unwrap_or_else(PerformanceAttribution::disabled)
    }

    /// The loaded model's limits a new request is checked against.
    pub fn generation_request_limits(&self) -> Qwen3_5GenerationRequestLimits {
        Qwen3_5GenerationRequestLimits {
            is_model_loaded: self.model.is_some(),
            think_end_token_id: self.think_end_token_id,
            vocabulary_size: self.vocabulary_size,
            maximum_position_count: self.maximum_position_count,
            hard_maximum_position_count: self.hard_maximum_position_count,
        }
    }

    /// The `astro_ssd_prompt_cache` statistics event, when one is open.
    pub fn astro_ssd_prompt_cache_stats_event(&self) -> Option<WorkerEvent> {
        let astro_ssd_prompt_cache = self.astro_ssd_prompt_cache.as_ref()?;
        let disk_store_config = self.astro_ssd_prompt_cache_disk_store_config.as_ref()?;
        Some(build_astro_ssd_prompt_cache_stats_event(
            &self.astro_ssd_prompt_cache_counters,
            u64::try_from(
                astro_ssd_prompt_cache
                    .model_contract_ref()
                    .block_token_count(),
            )
            .unwrap_or(u64::MAX),
            u64::try_from(astro_ssd_prompt_cache.sequence_state_block_count()).unwrap_or(u64::MAX),
            u64::try_from(astro_ssd_prompt_cache.boundary_state_snapshot_count())
                .unwrap_or(u64::MAX),
            u64::try_from(astro_ssd_prompt_cache.visual_embedding_count()).unwrap_or(u64::MAX),
            astro_ssd_prompt_cache.total_size_bytes(),
            astro_ssd_prompt_cache.visual_embedding_total_size_bytes(),
            disk_store_config.global_astro_ssd_prompt_cache_maximum_size_bytes(),
        ))
    }

    /// Clears this model's blocks, or every model's, from the
    /// `astro_ssd_prompt_cache`; `None` when none is open.
    pub fn clear_astro_ssd_prompt_cache(
        &self,
        model_id: Option<&str>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        let Some(astro_ssd_prompt_cache) = self.astro_ssd_prompt_cache.as_ref() else {
            return Ok(None);
        };
        let clear_outcome = astro_ssd_prompt_cache
            .clear_astro_ssd_prompt_cache(model_id)
            .map_err(|clear_error| fatal_engine_error(clear_error.to_string()))?;
        Ok(Some(WorkerEvent::AstroSsdPromptCacheCleared {
            model_id: clear_outcome.model_id,
            blocks_removed: clear_outcome.blocks_removed,
            bytes_freed: clear_outcome.bytes_freed,
        }))
    }
}

#[cfg(test)]
#[path = "execution_core_tests.rs"]
mod execution_core_tests;
