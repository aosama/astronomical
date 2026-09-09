//! Explicit serving settings for one K2 Horizon MoVA engine load.
//!
//! The struct stays GPU-free so hermetic tests and non-MLX callers can pin
//! the served configuration without the direct-MLX feature.

use std::path::PathBuf;

use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::WorkerChunkingConfiguration;

#[derive(Debug)]
pub struct K2HorizonMoVAServingSettings {
    pub maximum_context_tokens: Option<u32>,
    pub maximum_output_tokens: Option<u32>,
    pub prompt_processing_chunk_tokens: u32,
    pub chunking: Option<WorkerChunkingConfiguration>,
    pub persistent_prompt_cache_enabled: bool,
    pub prompt_cache_config: Option<PromptCacheConfig>,
    pub performance_attribution_log_path: Option<PathBuf>,
    pub full_attention_kv_state_growth_tokens: u32,
    pub decode_stage_attribution_enabled: bool,
    pub quantized_kv_cache_enabled: bool,
    pub fused_expert_decode_enabled: bool,
}

impl K2HorizonMoVAServingSettings {
    #[must_use]
    pub fn default_fixed() -> Self {
        Self {
            maximum_context_tokens: None,
            maximum_output_tokens: None,
            prompt_processing_chunk_tokens: 2_048,
            chunking: None,
            persistent_prompt_cache_enabled: false,
            prompt_cache_config: None,
            performance_attribution_log_path: None,
            full_attention_kv_state_growth_tokens: 256,
            decode_stage_attribution_enabled: false,
            quantized_kv_cache_enabled: false,
            fused_expert_decode_enabled: false,
        }
    }
}
