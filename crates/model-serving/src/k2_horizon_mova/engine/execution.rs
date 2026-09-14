//! Owner-thread K2 Horizon MoVA execution state vocabulary.
//!
//! The execution structs, active-generation state, and the pending-startup
//! constructor live here; the `MlxInferenceExecution` engine implementation
//! that drives them lives in `execution::engine_trait_impl`.

mod engine_trait_impl;

use std::time::Instant;

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxArray;

use crate::k2_horizon_mova::model::K2HorizonMoVAKvState;
use crate::k2_horizon_mova::model::K2HorizonMoVAModel;
use crate::k2_horizon_mova::{
    K2HorizonMoVAInferenceRequest, K2HorizonMoVAThinkingBudgetState, ValidatedK2HorizonMoVAArtifact,
};
use crate::{
    PerformanceAttribution, PerformanceAttributionLog, PersistentPromptCacheBlockKey,
    PersistentPromptCacheCounters, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig,
};

pub struct K2HorizonMoVAPendingStartup {
    pub validated_artifact: ValidatedK2HorizonMoVAArtifact,
    pub effective_mlx_memory_ceiling_bytes: usize,
    pub allocator_cache_memory_limit_bytes: usize,
    pub prompt_processing_chunk_tokens: u32,
    pub performance_attribution: PerformanceAttribution,
    pub performance_attribution_enabled: bool,
    pub performance_attribution_log: PerformanceAttributionLog,
    pub attribution_model_id: String,
    pub attribution_model_revision: String,
    pub prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
    pub configured_prompt_cache_block_token_count: Option<usize>,
    pub prompt_cache_common_prefix_stride_blocks: u32,
    pub full_attention_kv_state_growth_tokens: u32,
    pub decode_stage_attribution_enabled: bool,
    pub quantized_kv_cache_enabled: bool,
    pub fused_expert_decode_enabled: bool,
}

pub struct K2HorizonMoVAInferenceExecution {
    pending_startup: Option<K2HorizonMoVAPendingStartup>,
    pub(super) model: Option<K2HorizonMoVAModel>,
    pub(super) active: Option<K2HorizonMoVAActiveGeneration>,
    pub(super) ceiling_bytes: u64,
    pub(super) allocator_cache_bytes: u64,
    prompt_processing_chunk_tokens: u32,
    pub(super) performance_attribution_log: PerformanceAttributionLog,
    performance_attribution_enabled: bool,
    pub(super) attribution_model_id: Option<String>,
    pub(super) attribution_model_revision: Option<String>,
    pub(super) total_artifact_payload_bytes: Option<u64>,
    pub(super) model_shard_count: Option<usize>,
    pub(super) persistent_prompt_cache: Option<std::sync::Arc<PersistentPromptCacheDiskStore>>,
    pub(super) persistent_prompt_cache_disk_store_config:
        Option<PersistentPromptCacheDiskStoreConfig>,
    pub(super) persistent_prompt_cache_counters: PersistentPromptCacheCounters,
    configured_prompt_cache_block_token_count: Option<usize>,
    prompt_cache_common_prefix_stride_blocks: u32,
    full_attention_kv_state_growth_tokens: u32,
    decode_stage_attribution_enabled: bool,
    quantized_kv_cache_enabled: bool,
    fused_expert_decode_enabled: bool,
}

pub(super) struct K2HorizonMoVAActiveGeneration {
    pub(super) request_id: RequestId,
    pub(super) remaining_output_tokens: u32,
    pub(super) remaining_prompt_token_ids: Vec<u32>,
    pub(super) prompt_token_ids: Vec<u32>,
    pub(super) processed_prompt_token_count: u32,
    pub(super) caches: Vec<K2HorizonMoVAKvState>,
    pub(super) random_state: MlxArray,
    pub(super) next_input_token_ids: Vec<u32>,
    pub(super) generated_count: u32,
    pub(super) request: K2HorizonMoVAInferenceRequest,
    pub(super) prefill_started_at: Instant,
    pub(super) performance_attribution: PerformanceAttribution,
    pub(super) last_published_block_key: Option<PersistentPromptCacheBlockKey>,
    pub(super) cached_token_count: u32,
    pub(super) thinking_budget: K2HorizonMoVAThinkingBudgetState,
}

impl K2HorizonMoVAInferenceExecution {
    pub(crate) fn pending(pending_startup: K2HorizonMoVAPendingStartup) -> Self {
        let ceiling_bytes = pending_startup.effective_mlx_memory_ceiling_bytes as u64;
        let allocator_cache_bytes = pending_startup.allocator_cache_memory_limit_bytes as u64;
        let prompt_processing_chunk_tokens = pending_startup.prompt_processing_chunk_tokens.max(1);
        Self {
            pending_startup: Some(pending_startup),
            model: None,
            active: None,
            ceiling_bytes,
            allocator_cache_bytes,
            prompt_processing_chunk_tokens,
            performance_attribution_log: PerformanceAttributionLog::disabled(),
            performance_attribution_enabled: false,
            attribution_model_id: None,
            attribution_model_revision: None,
            total_artifact_payload_bytes: None,
            model_shard_count: None,
            persistent_prompt_cache: None,
            persistent_prompt_cache_disk_store_config: None,
            persistent_prompt_cache_counters: PersistentPromptCacheCounters::default(),
            configured_prompt_cache_block_token_count: None,
            prompt_cache_common_prefix_stride_blocks: 1,
            full_attention_kv_state_growth_tokens: 256,
            decode_stage_attribution_enabled: false,
            quantized_kv_cache_enabled: false,
            fused_expert_decode_enabled: false,
        }
    }
}
