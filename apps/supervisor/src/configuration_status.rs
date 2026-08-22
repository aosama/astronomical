//! Builds the path-free configured, resolved, and worker-effective status contract.

use astronomical_ipc_protocol::{
    WorkerLoadedModelRuntimeConfiguration, WorkerSpeculativePrefillRuntimeConfiguration,
};
use serde::Serialize;

use crate::application::ApplicationState;
use crate::{ResolvedRuntimeConfig, WorkerHealthSnapshot};

#[derive(Serialize)]
pub(crate) struct ConfigurationValue<T> {
    configured: Option<T>,
    default: Option<T>,
    effective: Option<T>,
}

#[derive(Serialize)]
struct NullableConfigurationValue<T> {
    is_configured: bool,
    configured: Option<T>,
    default: Option<T>,
    effective: Option<T>,
}

#[derive(Serialize)]
pub(crate) struct ConfigurationStatusSummary {
    configured_generation: Option<String>,
    resolved_generation: Option<String>,
    effective_generation: Option<String>,
    is_effective: bool,
    restart_required: bool,
    validation_error: Option<String>,
    model_discovery_diagnostics: Vec<ModelDiscoveryDiagnosticSummary>,
    unmatched_model_config_ids: Vec<String>,
    ready_model: Option<ReadyModelConfigurationSummary>,
    prompt_cache: PromptCacheConfigurationSummary,
    memory: MemoryConfigurationSummary,
}

#[derive(Serialize)]
struct ModelDiscoveryDiagnosticSummary {
    code: &'static str,
    model_id: String,
    configured_root_numbers: Vec<usize>,
}

#[derive(Serialize)]
struct ReadyModelConfigurationSummary {
    model_id: String,
    maximum_context_tokens: ConfigurationValue<u32>,
    maximum_output_default_tokens: ConfigurationValue<u32>,
    temperature: ConfigurationValue<f64>,
    top_p: ConfigurationValue<f64>,
    chunking: ChunkingConfigurationSummary,
    mtp_draft_depth: ConfigurationValue<u8>,
    mtp_head_model_id: ConfigurationValue<String>,
    mtp_head_unavailable_reason: Option<String>,
    speculative_prefill: SpeculativePrefillConfigurationSummary,
    speculative_prefill_unavailable_reason: Option<String>,
}

#[derive(Serialize)]
struct SpeculativePrefillConfigurationSummary {
    enabled: ConfigurationValue<bool>,
    draft_model_id: ConfigurationValue<String>,
    keep_percentage: ConfigurationValue<u32>,
    minimum_prompt_tokens: ConfigurationValue<u32>,
}

#[derive(Serialize)]
struct ChunkingConfigurationSummary {
    fixed_prompt_processing_chunk_size_tokens: ConfigurationValue<u32>,
    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: ConfigurationValue<u32>,
    full_attention_key_value_growth_tokens: ConfigurationValue<u32>,
    speculative_prefill_draft_forward_tokens: ConfigurationValue<u32>,
    prefill_graph_submission_layer_interval: ConfigurationValue<u32>,
    experimental_ssd_paging_generation_graph_submission_layer_interval: ConfigurationValue<u32>,
    prompt_cache_block_tokens: NullableConfigurationValue<u32>,
    prompt_cache_common_prefix_stride_blocks: ConfigurationValue<u32>,
}

#[derive(Serialize)]
struct PromptCacheConfigurationSummary {
    enabled: ConfigurationValue<bool>,
    capacity_bytes: ConfigurationValue<u64>,
}

#[derive(Serialize)]
struct MemoryConfigurationSummary {
    configured_maximum_bytes: Option<u64>,
    effective_maximum_bytes: u64,
    pending_maximum_bytes: Option<u64>,
    error: Option<String>,
}

impl ConfigurationStatusSummary {
    pub(crate) fn from_application(
        application_state: &ApplicationState,
        worker_health_snapshot: &WorkerHealthSnapshot,
    ) -> Self {
        let configured_config = application_state
            .configured_config_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.read().ok().map(|config| config.clone()));
        let validation_error = application_state
            .configuration_validation_error
            .read()
            .ok()
            .and_then(|validation_error| validation_error.clone());
        let resolved_config = application_state
            .reloadable_config
            .as_ref()
            .and_then(|snapshot| snapshot.read().ok().map(|config| config.clone()));
        let worker_configuration = worker_health_snapshot
            .worker_runtime_feature_configuration
            .as_ref();
        let configured_generation = configured_config
            .as_ref()
            .map(|config| config.configuration_generation.clone());
        let effective_generation = worker_configuration
            .map(|configuration| configuration.configuration_generation.clone());
        let resolved_generation = resolved_config
            .as_ref()
            .map(|config| config.configuration_generation.clone());
        let is_effective = configured_generation.is_some()
            && validation_error.is_none()
            && configured_generation == resolved_generation
            && configured_generation == effective_generation;
        let restart_required = worker_health_snapshot.status != crate::WorkerHealthStatus::Loading
            && validation_error.is_none()
            && configured_generation.is_some()
            && !is_effective
            && worker_health_snapshot
                .pending_mlx_memory_ceiling_bytes
                .is_none();
        Self {
            configured_generation,
            resolved_generation,
            effective_generation,
            is_effective,
            restart_required,
            validation_error,
            model_discovery_diagnostics: configured_config
                .as_ref()
                .map(|config| {
                    config
                        .model_discovery_diagnostics
                        .iter()
                        .map(|diagnostic| ModelDiscoveryDiagnosticSummary {
                            code: "ambiguous_model_identity",
                            model_id: diagnostic.model_id.clone(),
                            configured_root_numbers: diagnostic.configured_root_numbers.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            unmatched_model_config_ids: configured_config
                .as_ref()
                .map(|config| config.unmatched_model_config_ids.clone())
                .unwrap_or_default(),
            ready_model: ready_model_summary(
                configured_config.as_ref(),
                resolved_config.as_ref(),
                worker_health_snapshot.ready_model_id.as_deref(),
                worker_configuration.and_then(|configuration| configuration.loaded_model.as_ref()),
                worker_health_snapshot
                    .mtp_depth_status
                    .effective_execution_draft_depth,
            ),
            prompt_cache: prompt_cache_summary(configured_config.as_ref(), worker_configuration),
            memory: MemoryConfigurationSummary {
                configured_maximum_bytes: configured_config
                    .as_ref()
                    .and_then(|config| config.maximum_mlx_memory_bytes),
                effective_maximum_bytes: worker_health_snapshot.mlx_memory_ceiling_bytes,
                pending_maximum_bytes: worker_health_snapshot.pending_mlx_memory_ceiling_bytes,
                error: worker_health_snapshot.mlx_memory_limit_error.clone(),
            },
        }
    }
}

fn ready_model_summary(
    configured_config: Option<&ResolvedRuntimeConfig>,
    resolved_config: Option<&ResolvedRuntimeConfig>,
    ready_model_id: Option<&str>,
    effective_model: Option<&WorkerLoadedModelRuntimeConfiguration>,
    effective_mtp_draft_depth: Option<u8>,
) -> Option<ReadyModelConfigurationSummary> {
    let ready_model_id = ready_model_id?;
    let configured_policy =
        configured_config.and_then(|config| config.model_policy_catalog.get(ready_model_id));
    let resolved_policy =
        resolved_config.and_then(|config| config.model_policy_catalog.get(ready_model_id));
    let configured_worker_model =
        configured_policy.map(|policy| &policy.worker_model_configuration);
    let effective_model = effective_model.filter(|model| model.model_id() == ready_model_id);
    let effective_autoregressive_model = effective_model.and_then(|model| model.autoregressive());
    let configured_autoregressive_model =
        configured_worker_model.and_then(|model| model.autoregressive());
    let configured_speculative_prefill = configured_policy.and_then(|policy| {
        policy
            .acceleration_availability
            .configured_speculative_prefill
            .as_ref()
    });
    let effective_speculative_prefill =
        effective_autoregressive_model.and_then(|model| model.speculative_prefill.as_ref());
    Some(ReadyModelConfigurationSummary {
        model_id: ready_model_id.to_owned(),
        maximum_context_tokens: ConfigurationValue {
            configured: configured_policy
                .and_then(|policy| policy.configured_maximum_context_tokens),
            default: configured_policy.map(|policy| policy.default_maximum_context_tokens),
            effective: effective_autoregressive_model.map(|model| model.maximum_context_tokens),
        },
        maximum_output_default_tokens: ConfigurationValue {
            configured: configured_policy.and_then(|policy| {
                policy
                    .generation_defaults
                    .configured_maximum_output_tokens
                    .map(u32::from)
            }),
            default: configured_policy.map(|policy| {
                astronomical_config::DEFAULT_MAXIMUM_OUTPUT_TOKENS.min(
                    policy
                        .configured_maximum_context_tokens
                        .unwrap_or(policy.default_maximum_context_tokens)
                        .saturating_sub(1),
                )
            }),
            effective: resolved_policy
                .map(|policy| u32::from(policy.generation_defaults.maximum_output_tokens)),
        },
        temperature: sampling_value(
            configured_policy.and_then(|policy| policy.generation_defaults.temperature_thousandths),
            resolved_policy.and_then(|policy| policy.generation_defaults.temperature_thousandths),
        ),
        top_p: sampling_value(
            configured_policy.and_then(|policy| policy.generation_defaults.top_p_thousandths),
            resolved_policy.and_then(|policy| policy.generation_defaults.top_p_thousandths),
        ),
        chunking: chunking_summary(configured_policy, effective_model),
        mtp_draft_depth: ConfigurationValue {
            configured: configured_autoregressive_model.and_then(|model| model.mtp_draft_depth),
            default: None,
            effective: effective_mtp_draft_depth,
        },
        mtp_head_model_id: ConfigurationValue {
            configured: configured_policy.and_then(|policy| {
                policy
                    .acceleration_availability
                    .configured_mtp_head_model_id
                    .clone()
            }),
            default: None,
            effective: effective_autoregressive_model
                .and_then(|model| model.mtp_head_model_id.clone()),
        },
        mtp_head_unavailable_reason: configured_policy.and_then(|policy| {
            policy
                .acceleration_availability
                .mtp_head_unavailable_reason
                .clone()
        }),
        speculative_prefill: speculative_prefill_summary(
            configured_speculative_prefill,
            effective_speculative_prefill,
        ),
        speculative_prefill_unavailable_reason: configured_policy.and_then(|policy| {
            policy
                .acceleration_availability
                .speculative_prefill_unavailable_reason
                .clone()
        }),
    })
}

fn chunking_summary(
    configured_policy: Option<&crate::RuntimeModelPolicy>,
    effective_model: Option<&WorkerLoadedModelRuntimeConfiguration>,
) -> ChunkingConfigurationSummary {
    let configured_fields = configured_policy
        .map(|policy| policy.configured_chunking_fields)
        .unwrap_or_default();
    let configured_chunking = configured_policy
        .and_then(|policy| policy.worker_model_configuration.autoregressive())
        .map(|configuration| &configuration.chunking);
    let effective_chunking = effective_model
        .and_then(WorkerLoadedModelRuntimeConfiguration::autoregressive)
        .map(|configuration| &configuration.chunking);
    ChunkingConfigurationSummary {
        fixed_prompt_processing_chunk_size_tokens: ConfigurationValue {
            configured: configured_fields
                .fixed_prompt_processing_chunk_size_tokens
                .then(|| {
                    configured_chunking
                        .map(|chunking| chunking.fixed_prompt_processing_chunk_size_tokens)
                })
                .flatten(),
            default: Some(
                astronomical_config::DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
            ),
            effective: effective_chunking
                .map(|chunking| chunking.fixed_prompt_processing_chunk_size_tokens),
        },
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: ConfigurationValue {
            configured: configured_fields
                .fixed_ssd_streaming_prompt_processing_chunk_size_tokens
                .then(|| {
                    configured_chunking.and_then(|chunking| {
                        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens
                    })
                })
                .flatten(),
            default: None,
            effective: effective_chunking.and_then(|chunking| {
                chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens
            }),
        },
        full_attention_key_value_growth_tokens: ConfigurationValue {
            configured: configured_fields
                .full_attention_key_value_growth_tokens
                .then(|| {
                    configured_chunking
                        .map(|chunking| chunking.full_attention_key_value_growth_tokens)
                })
                .flatten(),
            default: Some(astronomical_config::DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS),
            effective: effective_chunking
                .map(|chunking| chunking.full_attention_key_value_growth_tokens),
        },
        speculative_prefill_draft_forward_tokens: ConfigurationValue {
            configured: configured_fields
                .speculative_prefill_draft_forward_tokens
                .then(|| {
                    configured_chunking
                        .map(|chunking| chunking.speculative_prefill_draft_forward_tokens)
                })
                .flatten(),
            default: Some(
                astronomical_config::DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS,
            ),
            effective: effective_chunking
                .map(|chunking| chunking.speculative_prefill_draft_forward_tokens),
        },
        prefill_graph_submission_layer_interval: ConfigurationValue {
            configured: configured_fields
                .prefill_graph_submission_layer_interval
                .then(|| {
                    configured_chunking
                        .map(|chunking| chunking.prefill_graph_submission_layer_interval)
                })
                .flatten(),
            default: Some(astronomical_config::DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL),
            effective: effective_chunking
                .map(|chunking| chunking.prefill_graph_submission_layer_interval),
        },
        experimental_ssd_paging_generation_graph_submission_layer_interval: ConfigurationValue {
            configured: configured_fields
                .experimental_ssd_paging_generation_graph_submission_layer_interval
                .then(|| {
                    configured_chunking.map(|chunking| {
                        chunking
                            .experimental_ssd_paging_generation_graph_submission_layer_interval
                    })
                })
                .flatten(),
            default: Some(
                astronomical_config::DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
            ),
            effective: effective_chunking.map(|chunking| {
                chunking.experimental_ssd_paging_generation_graph_submission_layer_interval
            }),
        },
        prompt_cache_block_tokens: NullableConfigurationValue {
            is_configured: configured_fields.prompt_cache_block_tokens,
            configured: configured_fields
                .prompt_cache_block_tokens
                .then(|| configured_chunking.and_then(|chunking| chunking.prompt_cache_block_tokens))
                .flatten(),
            default: None,
            effective: effective_chunking.and_then(|chunking| chunking.prompt_cache_block_tokens),
        },
        prompt_cache_common_prefix_stride_blocks: ConfigurationValue {
            configured: configured_fields
                .prompt_cache_common_prefix_stride_blocks
                .then(|| {
                    configured_chunking
                        .map(|chunking| chunking.prompt_cache_common_prefix_stride_blocks)
                })
                .flatten(),
            default: Some(
                astronomical_config::DEFAULT_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS,
            ),
            effective: effective_chunking
                .map(|chunking| chunking.prompt_cache_common_prefix_stride_blocks),
        },
    }
}

fn sampling_value(
    configured_thousandths: Option<u16>,
    effective_thousandths: Option<u16>,
) -> ConfigurationValue<f64> {
    let configured = configured_thousandths.map(|value| f64::from(value) / 1_000.0);
    ConfigurationValue {
        configured,
        default: None,
        effective: effective_thousandths.map(|value| f64::from(value) / 1_000.0),
    }
}

fn speculative_prefill_summary(
    configured: Option<&crate::ConfiguredSpeculativePrefillPolicy>,
    effective: Option<&WorkerSpeculativePrefillRuntimeConfiguration>,
) -> SpeculativePrefillConfigurationSummary {
    SpeculativePrefillConfigurationSummary {
        enabled: ConfigurationValue {
            configured: Some(configured.is_some()),
            default: Some(false),
            effective: Some(effective.is_some()),
        },
        draft_model_id: ConfigurationValue {
            configured: configured.map(|configuration| configuration.draft_model_id.clone()),
            default: None,
            effective: effective.map(|configuration| configuration.draft_model_id.clone()),
        },
        keep_percentage: ConfigurationValue {
            configured: configured.map(|configuration| configuration.keep_percentage),
            default: Some(20),
            effective: effective.map(|configuration| configuration.keep_percentage),
        },
        minimum_prompt_tokens: ConfigurationValue {
            configured: configured.map(|configuration| configuration.minimum_prompt_tokens),
            default: Some(8_192),
            effective: effective.map(|configuration| configuration.minimum_prompt_tokens),
        },
    }
}

fn prompt_cache_summary(
    configured: Option<&ResolvedRuntimeConfig>,
    effective: Option<&astronomical_ipc_protocol::WorkerRuntimeFeatureConfiguration>,
) -> PromptCacheConfigurationSummary {
    PromptCacheConfigurationSummary {
        enabled: ConfigurationValue {
            configured: configured
                .and_then(|config| config.configured_persistent_prompt_cache_enabled),
            default: Some(true),
            effective: effective.map(|configuration| configuration.persistent_prompt_cache_enabled),
        },
        capacity_bytes: ConfigurationValue {
            configured: configured
                .and_then(|config| config.configured_prompt_cache_maximum_size_bytes),
            default: Some(
                astronomical_config::DEFAULT_PROMPT_CACHE_MAXIMUM_SIZE_GB * 1_000_000_000,
            ),
            effective: effective.map(|configuration| configuration.prompt_cache_maximum_size_bytes),
        },
    }
}
