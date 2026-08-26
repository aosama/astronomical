//! Supervisor-owned per-model catalog entries used for routing and worker swaps.

use std::path::PathBuf;

use astronomical_config::ResolvedModelConfig;
use astronomical_ipc_protocol::{WorkerChunkingConfiguration, WorkerModelConfiguration};

/// Request defaults resolved from one model's configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeModelGenerationDefaults {
    pub maximum_output_tokens: u16,
    pub configured_maximum_output_tokens: Option<u16>,
    pub temperature_thousandths: Option<u16>,
    pub top_p_thousandths: Option<u16>,
}

/// User-authored speculative-prefill relationship before auxiliary availability is applied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfiguredSpeculativePrefillPolicy {
    pub draft_model_id: String,
    pub keep_percentage: u32,
    pub minimum_prompt_tokens: u32,
}

/// Configured speculative-prefill intent and the bounded reason it cannot currently execute.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeModelAccelerationAvailability {
    pub configured_speculative_prefill: Option<ConfiguredSpeculativePrefillPolicy>,
    pub speculative_prefill_unavailable_reason: Option<String>,
    pub configured_mtp_enabled: Option<bool>,
}

/// One canonical requestable model's directory and fully resolved execution policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeModelPolicy {
    pub model_directory: PathBuf,
    pub generation_defaults: RuntimeModelGenerationDefaults,
    pub configured_maximum_context_tokens: Option<u32>,
    pub default_maximum_context_tokens: u32,
    pub configured_chunking_fields: astronomical_config::ConfiguredChunkingFields,
    pub acceleration_availability: RuntimeModelAccelerationAvailability,
    pub worker_model_configuration: WorkerModelConfiguration,
}

pub(crate) fn runtime_model_generation_defaults(
    resolved_model_config: &ResolvedModelConfig,
) -> RuntimeModelGenerationDefaults {
    RuntimeModelGenerationDefaults {
        maximum_output_tokens: u16::try_from(resolved_model_config.maximum_output_tokens())
            .unwrap_or(u16::MAX),
        configured_maximum_output_tokens: resolved_model_config
            .configured_maximum_output_tokens()
            .and_then(|maximum_output_tokens| u16::try_from(maximum_output_tokens).ok()),
        temperature_thousandths: resolved_model_config
            .temperature()
            .map(sampling_parameter_thousandths),
        top_p_thousandths: resolved_model_config
            .top_p()
            .map(sampling_parameter_thousandths),
    }
}

fn sampling_parameter_thousandths(sampling_parameter: f32) -> u16 {
    (sampling_parameter * 1_000.0).round() as u16
}

pub(crate) fn worker_chunking_configuration(
    chunking: &astronomical_config::ChunkingConfig,
) -> WorkerChunkingConfiguration {
    WorkerChunkingConfiguration {
        fixed_prompt_processing_chunk_size_tokens: chunking
            .fixed_prompt_processing_chunk_size_tokens(),
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: chunking
            .fixed_ssd_streaming_prompt_processing_chunk_size_tokens(),
        full_attention_key_value_growth_tokens: chunking.full_attention_key_value_growth_tokens(),
        speculative_prefill_draft_forward_tokens: chunking
            .speculative_prefill_draft_forward_tokens(),
        prefill_graph_submission_layer_interval: chunking.prefill_graph_submission_layer_interval(),
        experimental_ssd_paging_prefill_graph_submission_layer_interval: chunking
            .experimental_ssd_paging_prefill_graph_submission_layer_interval(),
        experimental_ssd_paging_generation_graph_submission_layer_interval: chunking
            .experimental_ssd_paging_generation_graph_submission_layer_interval(),
        prompt_cache_block_tokens: chunking.prompt_cache_block_tokens(),
        prompt_cache_common_prefix_stride_blocks: chunking
            .prompt_cache_common_prefix_stride_blocks(),
    }
}
