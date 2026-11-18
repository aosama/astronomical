mod advance_generation;
mod astro_ssd_prompt_cache_capture_engine;
mod astro_ssd_prompt_cache_startup_logging;
mod astro_ssd_prompt_cache_visual_identity;
mod completed_forward_memory;
/// Prefill-to-decode no-I/O reconciliation: lift pressure and preserve topology.
mod decode_expert_memory_handoff;
/// How many layers a reply keeps whole, and giving up the rest.
mod decoder_state_reuse;
mod draft_runtime_hosting;
pub(crate) mod engine_request;
pub(in crate::qwen3_5_streaming) mod forward_memory_baseline;
pub(in crate::qwen3_5_streaming) mod generated_token_emission;
mod generation_finalization;
mod generation_request_validation;
mod inject_input_tokens;
mod layer_at_a_time_prefill;
pub(crate) mod layer_major_span_plan;
mod memory_limit;
mod model_loading;
mod model_loading_finalization;
mod prefill_advance;
mod prefill_chunk_planning;
mod prefill_memory_recovery;
mod prompt_prefill_errors;
mod prompt_processing_chunk_sizer;
mod request_memory_release;
mod speculative_decode_advance;
mod start_generation;
mod test_controls;
use std::path::PathBuf;
use std::sync::Arc;

use astronomical_ipc_protocol::{
    RequestId, WorkerChunkingConfiguration, WorkerDecodeAstroExpertCachePolicy, WorkerEvent,
};
use astronomical_runtime_integration::MlxMemoryLimits;

use crate::astro_ssd_prompt_cache::counters::AstroSsdPromptCacheCounters;
use crate::inference_engine::load_result::EngineLoadResult;
use crate::{
    AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheDiskStoreConfig,
    AstroSsdPromptCacheModelContract, GeneratedToken, GenerationFinalization, InferenceEngineError,
    MlxInferenceEngine, MlxInferenceExecution, PerformanceAttribution, PerformanceAttributionLog,
    PerformanceAttributionOutcome, PersistentVisualEmbeddingModelContract, Qwen3_5InferenceRequest,
    StreamingStartupResidencyFacts, build_astro_ssd_prompt_cache_stats_event,
};

use self::engine_request::Qwen3_5EngineRequest;
use super::ValidatedQwen3_5Artifact;
use super::model::Qwen3_5Model;

pub use memory_limit::safe_minimum_mlx_memory_ceiling_bytes;
pub use prompt_processing_chunk_sizer::{
    Qwen3_5StreamingPromptProcessingChunkSizer, Qwen3_5StreamingPromptProcessingChunkSizerError,
};

/// Qwen3.5 inference engine backed by the architecture-neutral MLX owner driver.
pub type Qwen3_5StreamingEngine = MlxInferenceEngine<Qwen3_5InferenceExecution>;

impl MlxInferenceEngine<Qwen3_5InferenceExecution> {
    /// Starts the owner thread with an explicit `prefill_chunk_tokens` sizer.
    ///
    /// The model directory remains required for sparse expert paging.
    // Construction dependencies remain explicit to avoid another configuration facade.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_prompt_processing_chunk_sizer(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5StreamingPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
    ) -> Result<Qwen3_5StreamingEngine, InferenceEngineError> {
        Self::new_with_runtime_chunking_and_performance_attribution(
            validated_artifact,
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            chunking,
            PerformanceAttribution::disabled(),
            PerformanceAttributionLog::disabled(),
        )
    }
    /// Starts the owner thread with every model-serving work boundary resolved.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_runtime_chunking_and_performance_attribution(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5StreamingPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Qwen3_5StreamingEngine, InferenceEngineError> {
        let maximum_context_tokens = validated_artifact.config().maximum_position_count();
        Self::new_with_effective_context_runtime_chunking_and_performance_attribution(
            validated_artifact,
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            maximum_context_tokens,
            chunking,
            model_loading_performance_attribution,
            performance_attribution_log,
        )
    }

    /// Starts the owner thread with a config-resolved context no larger than the artifact.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_effective_context_runtime_chunking_and_performance_attribution(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5StreamingPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        maximum_context_tokens: u32,
        chunking: WorkerChunkingConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Qwen3_5StreamingEngine, InferenceEngineError> {
        Self::new_with_effective_context_runtime_chunking_performance_attribution_and_decode_astro_expert_cache_policy(
            validated_artifact,
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            maximum_context_tokens,
            chunking,
            model_loading_performance_attribution,
            performance_attribution_log,
            WorkerDecodeAstroExpertCachePolicy::Baseline,
        )
    }

    /// Starts the owner thread with an internal decode `astro_expert_cache` benchmark policy.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_effective_context_runtime_chunking_performance_attribution_and_decode_astro_expert_cache_policy(
        validated_artifact: ValidatedQwen3_5Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: Qwen3_5StreamingPromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        maximum_context_tokens: u32,
        chunking: WorkerChunkingConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
        decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    ) -> Result<Qwen3_5StreamingEngine, InferenceEngineError> {
        let full_attention_kv_state_growth_tokens =
            i32::try_from(chunking.full_attention_key_value_growth_tokens).map_err(|_| {
                fatal_engine_error("full-attention growth tokens exceed Int32 range")
            })?;
        let end_of_sequence_token_ids = validated_artifact
            .config()
            .end_of_sequence_token_ids()
            .to_vec();
        let artifact_maximum_position_count = validated_artifact.config().maximum_position_count();
        if maximum_context_tokens == 0 || maximum_context_tokens > artifact_maximum_position_count {
            return Err(fatal_engine_error(
                "effective Qwen3.5 context must be positive and not exceed the artifact context",
            ));
        }
        let maximum_position_count = maximum_context_tokens as usize;
        let hard_maximum_position_count = artifact_maximum_position_count as usize;
        let vocabulary_size = validated_artifact.config().vocabulary_size();
        let context_memory_reservation_bytes_per_token = validated_artifact
            .config()
            .context_memory_reservation_bytes(1)
            .ok_or_else(|| fatal_engine_error("Qwen3.5 context memory reservation overflowed"))?;
        let memory_limits =
            MlxMemoryLimits::new(active_memory_limit_bytes, mlx_buffer_cache_limit_bytes)
                .map_err(qwen3_5_runtime_error)?;
        MlxInferenceEngine::new(move || Qwen3_5InferenceExecution {
            active_request: None,
            decode_astro_expert_cache_policy,
            astro_ssd_prompt_cache_disk_store_config,
            astro_ssd_prompt_cache_counters: AstroSsdPromptCacheCounters::default(),
            context_memory_reservation_bytes_per_token,
            end_of_sequence_token_ids,
            think_end_token_id,
            full_attention_kv_state_growth_tokens,
            memory_limits,
            model_directory,
            model_id: None,
            model_revision: None,
            model_loading_performance_attribution: Some(model_loading_performance_attribution),
            performance_attribution_log,
            maximum_position_count,
            hard_maximum_position_count,
            model: None,
            speculative_draft_runtime: None,
            prompt_span_to_fail_once_for_tests: None,
            astro_ssd_prompt_cache_model_contract: None,
            persistent_visual_embedding_model_contract: None,
            astro_ssd_prompt_cache: None,
            prompt_processing_chunk_sizer,
            chunking,
            validated_artifact: Some(validated_artifact),
            startup_residency_facts: None,
            vocabulary_size,
        })
    }
}

pub struct Qwen3_5InferenceExecution {
    pub(super) active_request: Option<Qwen3_5EngineRequest>,
    pub(super) decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
    pub(crate) astro_ssd_prompt_cache_counters: AstroSsdPromptCacheCounters,
    pub(super) context_memory_reservation_bytes_per_token: usize,
    end_of_sequence_token_ids: Vec<u32>,
    /// Token ID that closes the thinking block. Used to enforce thinking_budget.
    think_end_token_id: u32,
    /// Capacity-growth granularity for full-attention KV slabs.
    full_attention_kv_state_growth_tokens: i32,
    pub(super) memory_limits: MlxMemoryLimits,
    /// Directory containing the safetensors model shards. Needed for
    /// expert paging to read safetensors headers at startup.
    model_directory: PathBuf,
    model_id: Option<String>,
    model_revision: Option<String>,
    model_loading_performance_attribution: Option<PerformanceAttribution>,
    performance_attribution_log: PerformanceAttributionLog,
    maximum_position_count: usize,
    /// Model artifact's native position range: the only hard rejection
    /// boundary for request context. The configured `maximum_position_count`
    /// above is advisory and advertised unchanged.
    hard_maximum_position_count: usize,
    pub(crate) model: Option<Qwen3_5Model>,
    /// The DFlash2 drafter hosted beside the model; `None` without one.
    pub(super) speculative_draft_runtime: Option<
        crate::qwen3_5_core::draft_tree_decode::engine_draft_runtime::Qwen3_5SpeculativeDraftRuntime,
    >,
    /// A span, as (layer, span), that the next layer-at-a-time prompt fails
    /// once on purpose; set only by a test.
    pub(super) prompt_span_to_fail_once_for_tests: Option<(usize, usize)>,
    pub(crate) astro_ssd_prompt_cache_model_contract: Option<AstroSsdPromptCacheModelContract>,
    pub(crate) persistent_visual_embedding_model_contract:
        Option<PersistentVisualEmbeddingModelContract>,
    pub(crate) astro_ssd_prompt_cache: Option<Arc<AstroSsdPromptCacheDiskStore>>,
    prompt_processing_chunk_sizer: Qwen3_5StreamingPromptProcessingChunkSizer,
    chunking: WorkerChunkingConfiguration,
    validated_artifact: Option<ValidatedQwen3_5Artifact>,
    /// Startup engine-selection facts, memoized from the first load; a raised ceiling consults them
    /// to decide whether a replacement worker must load the resident engine.
    startup_residency_facts: Option<StreamingStartupResidencyFacts>,
    vocabulary_size: u32,
}

pub(crate) type Qwen3_5EngineState = Qwen3_5InferenceExecution;

impl MlxInferenceExecution for Qwen3_5InferenceExecution {
    type Request = Qwen3_5InferenceRequest;

    fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Qwen3_5EngineState::load(self)
    }

    fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        Qwen3_5EngineState::start_generation(self, inference_request)
    }

    fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        Qwen3_5EngineState::advance_generation(self, request_id)
    }

    fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        Qwen3_5EngineState::inject_input_tokens(self, request_id, input_token_ids)
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Qwen3_5EngineState::cancel_generation(self, request_id)
    }

    fn collect_astro_ssd_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        Ok(Qwen3_5EngineState::collect_astro_ssd_prompt_cache_stats(
            self,
        ))
    }

    fn clear_astro_ssd_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        let Some(astro_ssd_prompt_cache) = self.astro_ssd_prompt_cache.as_ref() else {
            return Ok(None);
        };
        let clear_outcome = astro_ssd_prompt_cache
            .clear_astro_ssd_prompt_cache(model_id.as_deref())
            .map_err(|clear_error| fatal_engine_error(clear_error.to_string()))?;
        Ok(Some(WorkerEvent::AstroSsdPromptCacheCleared {
            model_id: clear_outcome.model_id,
            blocks_removed: clear_outcome.blocks_removed,
            bytes_freed: clear_outcome.bytes_freed,
        }))
    }

    fn collect_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        Qwen3_5EngineState::collect_current_mlx_memory_telemetry(self)
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        Qwen3_5EngineState::update_mlx_memory_limit(self, requested_mlx_memory_ceiling_bytes)
    }
}

impl Qwen3_5EngineState {
    fn collect_astro_ssd_prompt_cache_stats(&self) -> Option<WorkerEvent> {
        let astro_ssd_prompt_cache = self.astro_ssd_prompt_cache.as_ref()?;
        let global_astro_ssd_prompt_cache_maximum_size_bytes = self
            .astro_ssd_prompt_cache_disk_store_config
            .as_ref()
            .map(|disk_store_config| {
                disk_store_config.global_astro_ssd_prompt_cache_maximum_size_bytes()
            })?;
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
            global_astro_ssd_prompt_cache_maximum_size_bytes,
        ))
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        let Some(active_request) = self.active_request.take() else {
            return Ok(GenerationFinalization::default());
        };
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            Ok(GenerationFinalization::default())
        } else {
            Ok(self.finalize_generation_request(
                active_request,
                PerformanceAttributionOutcome::Cancelled,
                None,
            ))
        }
    }
}

pub(crate) use crate::qwen3_5_core::engine_errors::{fatal_engine_error, qwen3_5_runtime_error};

/// The streaming engine keeps no decoder state between requests, so a
/// finalized measurement has no retained live prefix to charge to the
/// context owner.
pub(super) const NO_RETAINED_LIVE_PREFIX_BYTES: u64 = 0;

use crate::{EngineGenerationStart, MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
