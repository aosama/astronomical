//! Worker-facing startup for the folded ternary family: pack validation,
//! Qwen-dialect processor construction, and the owner-thread engine.

use std::path::Path;

use thiserror::Error;

use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::WorkerChunkingConfiguration;

use crate::MlxInferenceEngine;
use crate::PersistentPromptCacheDiskStoreConfig;
use crate::Qwen3_5GenerationProcessor;
use crate::prism_hadamard_qwen35::artifacts::FoldedPackValidator;
use crate::prism_hadamard_qwen35::engine::{
    PrismHadamardQwen35Engine, PrismHadamardQwen35InferenceExecution,
    PrismHadamardQwen35PendingStartup,
};
use crate::{PerformanceAttribution, PerformanceAttributionLog, PerformanceOperation};

/// Serving settings the worker hands to the folded-ternary startup.
#[derive(Debug, Clone)]
pub struct PrismHadamardQwen35ServingSettings {
    pub maximum_context_tokens: Option<u32>,
    pub maximum_output_tokens: Option<u32>,
    pub prompt_processing_chunk_tokens: u32,
    pub enable_thinking: bool,
    pub chunking: Option<WorkerChunkingConfiguration>,
    pub persistent_prompt_cache_enabled: bool,
    pub prompt_cache_config: Option<PromptCacheConfig>,
    pub performance_attribution_log_path: Option<std::path::PathBuf>,
}

/// Typed folded-ternary startup failure.
#[derive(Debug, Error)]
pub enum PrismHadamardQwen35StartupError {
    #[error("folded ternary pack validation failed: {0}")]
    PackValidation(#[source] crate::prism_hadamard_qwen35::artifacts::FoldedPackValidationError),
    #[error("folded ternary pack does not retain its tokenizer document")]
    MissingTokenizer,
    #[error("the requested serving context exceeds the pack's declared window")]
    ContextWindowExceeded {
        requested_context_tokens: u32,
        declared_context_tokens: u32,
    },
    #[error("folded ternary engine owner failed: {reason}")]
    EngineOwner { reason: String },
}

/// Prepares the folded-ternary generation processor and owner-thread engine.
///
/// The pack validation and tokenizer load run on the caller thread (CPU-only
/// document work); MLX runtime initialization and weight loading stay inside
/// `load` on the owner thread because MLX handles are not `Send`.
pub fn initialize_prism_hadamard_qwen35_model_with_serving_settings(
    model_directory: &Path,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
    serving_settings: PrismHadamardQwen35ServingSettings,
) -> Result<(Qwen3_5GenerationProcessor, PrismHadamardQwen35Engine), PrismHadamardQwen35StartupError>
{
    let mut performance_attribution = if performance_attribution_enabled {
        PerformanceAttribution::enabled()
    } else {
        PerformanceAttribution::disabled()
    };
    let pack = performance_attribution
        .measure_operation(PerformanceOperation::ArtifactValidation, |_| {
            FoldedPackValidator::new().validate(
                model_directory,
                serving_settings
                    .maximum_output_tokens
                    .unwrap_or(u32::from(u16::MAX)),
            )
        })
        .map_err(PrismHadamardQwen35StartupError::PackValidation)?;
    let declared_context_tokens = pack.config().text().max_position_embeddings;
    let context_window = serving_settings
        .maximum_context_tokens
        .unwrap_or(declared_context_tokens)
        .min(declared_context_tokens);
    if context_window == 0 {
        return Err(PrismHadamardQwen35StartupError::ContextWindowExceeded {
            requested_context_tokens: context_window,
            declared_context_tokens,
        });
    }
    let max_output_tokens = serving_settings
        .maximum_output_tokens
        .unwrap_or_else(|| u32::from(u16::MAX).min(context_window - 1));
    let model_id = pack.model_id().to_owned();
    let model_revision = pack.revision().to_owned();
    let model_vocabulary_size = pack.config().text().vocab_size;
    let tokenizer_bytes = pack
        .tokenizer_bytes()
        .ok_or(PrismHadamardQwen35StartupError::MissingTokenizer)?
        .to_vec();
    let generation_processor = performance_attribution
        .measure_operation(PerformanceOperation::TokenizerInitialization, |_| {
            Qwen3_5GenerationProcessor::from_folded_pack_text_dialect(
                &tokenizer_bytes,
                &model_id,
                model_vocabulary_size,
                context_window,
                max_output_tokens,
                serving_settings.enable_thinking,
                performance_attribution_enabled,
            )
        })
        .map_err(
            |tokenizer_error| PrismHadamardQwen35StartupError::EngineOwner {
                reason: tokenizer_error.to_string(),
            },
        )?;
    let performance_attribution_log = match &serving_settings.performance_attribution_log_path {
        Some(log_path) => {
            PerformanceAttributionLog::open(log_path, performance_attribution_enabled).map_err(
                |io_error| PrismHadamardQwen35StartupError::EngineOwner {
                    reason: format!("attribution log failed: {io_error}"),
                },
            )?
        }
        None => PerformanceAttributionLog::disabled(),
    };
    let prompt_cache_disk_store_config = serving_settings
        .prompt_cache_config
        .as_ref()
        .filter(|_| serving_settings.persistent_prompt_cache_enabled)
        .map(|prompt_cache_config| {
            let per_model_prompt_cache_config =
                prompt_cache_config.for_model(&model_id, &model_revision);
            PersistentPromptCacheDiskStoreConfig::new(
                per_model_prompt_cache_config
                    .active_model_prompt_cache_directory()
                    .clone(),
                per_model_prompt_cache_config
                    .global_prompt_cache_root_directory()
                    .clone(),
                per_model_prompt_cache_config.global_prompt_cache_maximum_size_bytes(),
            )
        });
    // The durable cache block size and checkpoint stride are prompt-cache
    // configuration, not the prefill chunk size: they are contract identity
    // (the storage fingerprint embeds the block count) and must satisfy the
    // layout's persistence alignment, which an arbitrary chunk size does not.
    let configured_prompt_cache_block_token_count =
        serving_settings.chunking.as_ref().and_then(|chunking| {
            chunking
                .prompt_cache_block_tokens
                .map(|block_token_count| block_token_count as usize)
        });
    let prompt_cache_common_prefix_stride_blocks = serving_settings
        .chunking
        .as_ref()
        .map(|chunking| chunking.prompt_cache_common_prefix_stride_blocks)
        .unwrap_or(1)
        .max(1);
    let pending_startup = PrismHadamardQwen35PendingStartup {
        validated_pack: pack,
        effective_mlx_memory_ceiling_bytes,
        allocator_cache_memory_limit_bytes,
        maximum_context_tokens: context_window,
        prompt_processing_chunk_tokens: serving_settings.prompt_processing_chunk_tokens,
        prompt_cache_disk_store_config,
        configured_prompt_cache_block_token_count,
        prompt_cache_common_prefix_stride_blocks,
        performance_attribution,
        performance_attribution_log,
        attribution_model_id: model_id,
        attribution_model_revision: model_revision,
    };
    let engine = MlxInferenceEngine::new(move || {
        PrismHadamardQwen35InferenceExecution::pending(pending_startup)
    })
    .map_err(|error| PrismHadamardQwen35StartupError::EngineOwner {
        reason: error.to_string(),
    })?;
    Ok((generation_processor, engine))
}
