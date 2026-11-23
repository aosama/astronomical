//! The streaming engine's execution state, owned by the MLX owner thread.

use std::path::PathBuf;
use std::sync::Arc;

use ipc_protocol::{WorkerChunkingConfiguration, WorkerDecodeAstroExpertCachePolicy};
use runtime_integration::MlxMemoryLimits;

use crate::astro_ssd_prompt_cache::counters::AstroSsdPromptCacheCounters;
use crate::{
    AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheDiskStoreConfig,
    AstroSsdPromptCacheModelContract, PerformanceAttribution, PerformanceAttributionLog,
    PersistentVisualEmbeddingModelContract, StreamingStartupResidencyFacts,
};

use super::super::ValidatedQwen3_5Artifact;
use super::super::model::Qwen3_5Model;
use super::Qwen3_5StreamingPromptProcessingChunkSizer;
use super::engine_request::Qwen3_5EngineRequest;

pub struct Qwen3_5InferenceExecution {
    pub(in crate::qwen3_5_streaming) active_request: Option<Qwen3_5EngineRequest>,
    pub(in crate::qwen3_5_streaming) decode_astro_expert_cache_policy:
        WorkerDecodeAstroExpertCachePolicy,
    pub(in crate::qwen3_5_streaming::inference_execution) astro_ssd_prompt_cache_disk_store_config:
        Option<AstroSsdPromptCacheDiskStoreConfig>,
    pub astro_ssd_prompt_cache_counters: AstroSsdPromptCacheCounters,
    pub(in crate::qwen3_5_streaming) context_memory_reservation_bytes_per_token: usize,
    pub(in crate::qwen3_5_streaming::inference_execution) end_of_sequence_token_ids: Vec<u32>,
    /// Token ID that closes the thinking block. Used to enforce thinking_budget.
    pub(in crate::qwen3_5_streaming::inference_execution) think_end_token_id: u32,
    /// Capacity-growth granularity for full-attention KV slabs.
    pub(in crate::qwen3_5_streaming::inference_execution) full_attention_kv_state_growth_tokens:
        i32,
    pub(in crate::qwen3_5_streaming) memory_limits: MlxMemoryLimits,
    /// Directory containing the safetensors model shards. Needed for
    /// expert paging to read safetensors headers at startup.
    pub(in crate::qwen3_5_streaming::inference_execution) model_directory: PathBuf,
    pub(in crate::qwen3_5_streaming::inference_execution) model_id: Option<String>,
    pub(in crate::qwen3_5_streaming::inference_execution) model_revision: Option<String>,
    pub(in crate::qwen3_5_streaming::inference_execution) model_loading_performance_attribution:
        Option<PerformanceAttribution>,
    pub(in crate::qwen3_5_streaming::inference_execution) performance_attribution_log:
        PerformanceAttributionLog,
    pub(in crate::qwen3_5_streaming::inference_execution) maximum_position_count: usize,
    /// Model artifact's native position range: the only hard rejection
    /// boundary for request context. The configured `maximum_position_count`
    /// above is advisory and advertised unchanged.
    pub(in crate::qwen3_5_streaming::inference_execution) hard_maximum_position_count: usize,
    pub model: Option<Qwen3_5Model>,
    /// A span, as (layer, span), that the next layer-at-a-time prompt fails
    /// once on purpose; set only by a test.
    pub(in crate::qwen3_5_streaming) prompt_span_to_fail_once_for_tests: Option<(usize, usize)>,
    pub astro_ssd_prompt_cache_model_contract: Option<AstroSsdPromptCacheModelContract>,
    pub persistent_visual_embedding_model_contract: Option<PersistentVisualEmbeddingModelContract>,
    pub astro_ssd_prompt_cache: Option<Arc<AstroSsdPromptCacheDiskStore>>,
    pub(in crate::qwen3_5_streaming::inference_execution) prompt_processing_chunk_sizer:
        Qwen3_5StreamingPromptProcessingChunkSizer,
    pub(in crate::qwen3_5_streaming::inference_execution) chunking: WorkerChunkingConfiguration,
    pub(in crate::qwen3_5_streaming::inference_execution) validated_artifact:
        Option<ValidatedQwen3_5Artifact>,
    /// Startup engine-selection facts, memoized from the first load; a raised ceiling consults them
    /// to decide whether a replacement worker must load the resident engine.
    pub(in crate::qwen3_5_streaming::inference_execution) startup_residency_facts:
        Option<StreamingStartupResidencyFacts>,
    pub(in crate::qwen3_5_streaming::inference_execution) vocabulary_size: u32,
}

pub type Qwen3_5EngineState = Qwen3_5InferenceExecution;
