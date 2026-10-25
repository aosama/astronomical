//! The resident Qwen3.5 engine state owner and its state-local methods.

use std::path::PathBuf;
use std::sync::Arc;

use astronomical_ipc_protocol::{RequestId, WorkerChunkingConfiguration, WorkerEvent};
use astronomical_runtime_integration::MlxMemoryLimits;

use crate::persistent_cache::counters::PersistentPromptCacheCounters;
use crate::{
    AdaptiveRamGrowthGuard, GenerationFinalization, InferenceEngineError, PerformanceAttribution,
    PerformanceAttributionLog, PerformanceAttributionOutcome, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig, PersistentPromptCacheModelContract,
    PersistentVisualEmbeddingModelContract, build_persistent_prompt_cache_stats_event,
};

use super::super::ValidatedQwen3_5Artifact;
use super::super::model::Qwen3_5ResidentModel;
use super::Qwen3_5ResidentPromptProcessingChunkSizer;
use super::engine_request::Qwen3_5EngineRequest;
use super::persistent_prompt_cache::retained_live_prefix::RetainedLiveDecoderPrefix;

pub struct Qwen3_5InferenceExecution {
    pub(in crate::qwen3_5_resident::inference_execution) active_request:
        Option<Qwen3_5EngineRequest>,
    pub(in crate::qwen3_5_resident::inference_execution) adaptive_ram_growth_guard:
        AdaptiveRamGrowthGuard,
    pub(in crate::qwen3_5_resident::inference_execution) adaptive_ram_growth_guard_enabled: bool,
    /// Diagnostic replacement for the static prefill activation floor applied to
    /// the loaded model's RAM budget; `None` keeps the production floor.
    pub(in crate::qwen3_5_resident::inference_execution) diagnostic_prefill_activation_floor_bytes:
        Option<u64>,
    pub(in crate::qwen3_5_resident::inference_execution) persistent_prompt_cache_disk_store_config:
        Option<PersistentPromptCacheDiskStoreConfig>,
    pub(crate) persistent_prompt_cache_counters: PersistentPromptCacheCounters,
    pub(in crate::qwen3_5_resident::inference_execution) context_memory_reservation_bytes_per_token:
        usize,
    pub(in crate::qwen3_5_resident::inference_execution) end_of_sequence_token_ids: Vec<u32>,
    /// Token ID that closes the thinking block. Used to enforce thinking_budget.
    pub(in crate::qwen3_5_resident::inference_execution) think_end_token_id: u32,
    /// Capacity-growth granularity for full-attention KV slabs.
    pub(in crate::qwen3_5_resident::inference_execution) full_attention_kv_state_growth_tokens: i32,
    pub(in crate::qwen3_5_resident::inference_execution) memory_limits: MlxMemoryLimits,
    /// Directory containing the safetensors shards used to load complete resident weights.
    pub(in crate::qwen3_5_resident::inference_execution) model_directory: PathBuf,
    pub(in crate::qwen3_5_resident::inference_execution) model_id: Option<String>,
    pub(in crate::qwen3_5_resident::inference_execution) model_revision: Option<String>,
    pub(in crate::qwen3_5_resident::inference_execution) model_loading_performance_attribution:
        Option<PerformanceAttribution>,
    pub(in crate::qwen3_5_resident::inference_execution) performance_attribution_log:
        PerformanceAttributionLog,
    pub(in crate::qwen3_5_resident::inference_execution) maximum_position_count: usize,
    /// Model artifact's native position range: the only hard rejection
    /// boundary for request context. The configured `maximum_position_count`
    /// above is advisory and advertised unchanged.
    pub(in crate::qwen3_5_resident::inference_execution) hard_maximum_position_count: usize,
    pub(crate) model: Option<Qwen3_5ResidentModel>,
    pub(crate) persistent_prompt_cache_model_contract: Option<PersistentPromptCacheModelContract>,
    pub(crate) persistent_visual_embedding_model_contract:
        Option<PersistentVisualEmbeddingModelContract>,
    pub(crate) persistent_prompt_cache: Option<Arc<PersistentPromptCacheDiskStore>>,
    pub(in crate::qwen3_5_resident::inference_execution) prompt_processing_chunk_sizer:
        Qwen3_5ResidentPromptProcessingChunkSizer,
    pub(in crate::qwen3_5_resident::inference_execution) chunking: WorkerChunkingConfiguration,
    /// The last completed generation's live decoder state, retained for
    /// in-memory prefix adoption (issue #1167). Dropped when the next request
    /// fails to adopt it or when the retention policy says so at completion.
    pub(in crate::qwen3_5_resident) retained_live_prefix: Option<RetainedLiveDecoderPrefix>,
    pub(in crate::qwen3_5_resident::inference_execution) validated_artifact:
        Option<ValidatedQwen3_5Artifact>,
    pub(in crate::qwen3_5_resident::inference_execution) vocabulary_size: u32,
}

pub(crate) type Qwen3_5EngineState = Qwen3_5InferenceExecution;

impl Qwen3_5EngineState {
    pub(super) fn collect_persistent_prompt_cache_stats(&self) -> Option<WorkerEvent> {
        let persistent_prompt_cache = self.persistent_prompt_cache.as_ref()?;
        let global_prompt_cache_maximum_size_bytes = self
            .persistent_prompt_cache_disk_store_config
            .as_ref()
            .map(|disk_store_config| disk_store_config.global_prompt_cache_maximum_size_bytes())?;
        Some(build_persistent_prompt_cache_stats_event(
            &self.persistent_prompt_cache_counters,
            u64::try_from(
                persistent_prompt_cache
                    .model_contract_ref()
                    .block_token_count(),
            )
            .unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.sequence_state_block_count()).unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.boundary_state_snapshot_count())
                .unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.visual_embedding_count()).unwrap_or(u64::MAX),
            persistent_prompt_cache.total_size_bytes(),
            persistent_prompt_cache.visual_embedding_total_size_bytes(),
            global_prompt_cache_maximum_size_bytes,
        ))
    }

    pub(super) fn cancel_generation(
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
