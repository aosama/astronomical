//! ModernBERT embedding engine over direct MLX arrays.
//!
//! One command maps to one lazy bidirectional-encoder graph evaluated once:
//! quantized token lookup, embedding LayerNorm, alternating global/local
//! attention layers, gated MLP, final LayerNorm, and host-side pooling with
//! L2 normalization. Pooling and normalization live in Astronomical; no
//! third-party embedding package is linked.

use std::path::{Path, PathBuf};
use std::time::Instant;

use astronomical_ipc_protocol::{
    EmbeddingsCommand, EmbeddingsFailureReason, WorkerEmbeddingCapabilities,
};
use astronomical_runtime_integration::{
    MlxMemoryLimits, MlxRuntime, MlxRuntimeError, MlxSafetensors,
};
use tokenizers::Tokenizer;

use crate::modernbert::artifact::ModernBertArtifact;
use crate::modernbert::configuration::ModernBertConfiguration;
use crate::modernbert::forward::embed_token_ids;
use crate::modernbert::tokenizer::encode_embedding_input;
use crate::performance_attribution::{
    ModelLoadingPerformanceAttributionMetadata, PerformanceAttribution, PerformanceAttributionLog,
    PerformanceAttributionOutcome, PerformanceOperation,
};
use crate::{EmbeddingEngine, EmbeddingEngineLoadResult, EmbeddingEngineOutput};

/// Loaded ModernBERT runtime bound to one artifact directory.
pub struct ModernBertEmbeddingEngine {
    model_id: String,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
    performance_attribution_log: PerformanceAttributionLog,
    preloaded_artifact: Option<ModernBertArtifact>,
    loaded_state: Option<LoadedModernBertState>,
}

struct LoadedModernBertState {
    runtime: MlxRuntime,
    tensors: MlxSafetensors,
    configuration: ModernBertConfiguration,
    tokenizer: Tokenizer,
}

impl ModernBertEmbeddingEngine {
    /// Creates an unloaded engine from validated on-disk artifact evidence.
    pub fn new(
        model_id: impl Into<String>,
        model_directory: &Path,
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        performance_attribution_enabled: bool,
        performance_attribution_log_path: PathBuf,
    ) -> Result<Self, EmbeddingsFailureReason> {
        let artifact = ModernBertArtifact::load(model_directory)?;
        let performance_attribution_log = PerformanceAttributionLog::open(
            &performance_attribution_log_path,
            performance_attribution_enabled,
        )
        .unwrap_or_else(|_| PerformanceAttributionLog::disabled());
        Ok(Self {
            model_id: model_id.into(),
            effective_mlx_memory_ceiling_bytes,
            allocator_cache_memory_limit_bytes,
            performance_attribution_enabled,
            performance_attribution_log,
            preloaded_artifact: Some(artifact),
            loaded_state: None,
        })
    }

    fn new_attribution(&self) -> PerformanceAttribution {
        if self.performance_attribution_enabled {
            PerformanceAttribution::enabled()
        } else {
            PerformanceAttribution::disabled()
        }
    }

    fn record_attribution_report(
        &mut self,
        attribution_report: Option<crate::performance_attribution::PerformanceAttributionReport>,
    ) {
        let Some(attribution_report) = attribution_report else {
            return;
        };
        let _ignored_log_result = self.performance_attribution_log.record(&attribution_report);
    }
}

impl EmbeddingEngine for ModernBertEmbeddingEngine {
    fn load(&mut self) -> Result<EmbeddingEngineLoadResult, EmbeddingsFailureReason> {
        let Some(artifact) = self.preloaded_artifact.take() else {
            return Err(EmbeddingsFailureReason::FatalExecution {
                reason: "embedding engine load was requested twice".to_owned(),
            });
        };
        let mut performance_attribution = self.new_attribution();
        let memory_limits = MlxMemoryLimits::new(
            self.effective_mlx_memory_ceiling_bytes,
            self.allocator_cache_memory_limit_bytes,
        )
        .map_err(|_| EmbeddingsFailureReason::FatalExecution {
            reason: "embedding runtime memory limits are invalid".to_owned(),
        })?;
        let runtime = performance_attribution.measure_operation(
            PerformanceOperation::MlxRuntimeInitialization,
            |_performance_attribution| {
                MlxRuntime::initialize(memory_limits).map_err(|_| {
                    EmbeddingsFailureReason::FatalExecution {
                        reason: "embedding runtime initialization failed".to_owned(),
                    }
                })
            },
        )?;
        let tensors = performance_attribution.measure_operation(
            PerformanceOperation::ModelSafetensorsMapping,
            |_performance_attribution| {
                runtime
                    .load_safetensors(artifact.weights_file, None)
                    .map_err(runtime_failure)
            },
        )?;
        let tokenizer = performance_attribution.measure_operation(
            PerformanceOperation::TokenizerInitialization,
            |_performance_attribution| {
                Tokenizer::from_bytes(&artifact.tokenizer_bytes).map_err(|tokenizer_error| {
                    EmbeddingsFailureReason::FatalExecution {
                        reason: format!("embedding tokenizer failed to load: {tokenizer_error}")
                            .chars()
                            .take(256)
                            .collect(),
                    }
                })
            },
        )?;
        let load_result = EmbeddingEngineLoadResult::new(
            self.model_id.clone(),
            WorkerEmbeddingCapabilities {
                vector_width: artifact.configuration.hidden_size,
                max_input_tokens: artifact.configuration.maximum_position_count,
            },
        );
        self.loaded_state = Some(LoadedModernBertState {
            runtime,
            tensors,
            configuration: artifact.configuration,
            tokenizer,
        });
        let attribution_report = performance_attribution.finish_model_loading(
            ModelLoadingPerformanceAttributionMetadata {
                outcome: PerformanceAttributionOutcome::Success,
                model_id: Some(self.model_id.clone()),
                model_revision: None,
                prefill_transient_observation_completed: false,
                prefill_observed_transient_high_water_bytes: 0,
                total_artifact_payload_bytes: None,
                resident_model_payload_bytes: None,
                model_shard_count: Some(1),
                mlx_active_memory_bytes: None,
                mlx_allocator_cache_memory_bytes: None,
                mlx_peak_memory_bytes: None,
                failure_description: None,
            },
        );
        self.record_attribution_report(attribution_report);
        Ok(load_result)
    }

    fn embed(
        &mut self,
        embeddings_command: &EmbeddingsCommand,
    ) -> Result<EmbeddingEngineOutput, EmbeddingsFailureReason> {
        embeddings_command.validate().map_err(|validation_error| {
            EmbeddingsFailureReason::invalid_request(validation_error.to_string())
        })?;
        let started_at = Instant::now();
        let mut performance_attribution = self.new_attribution();
        let loaded_state =
            self.loaded_state
                .as_mut()
                .ok_or_else(|| EmbeddingsFailureReason::FatalExecution {
                    reason: "embedding engine embed was called before load".to_owned(),
                })?;
        let encoded_inputs = performance_attribution.measure_operation(
            PerformanceOperation::EmbeddingsTokenization,
            |_performance_attribution| {
                let mut encoded_inputs = Vec::with_capacity(embeddings_command.inputs.len());
                for input_text in &embeddings_command.inputs {
                    encoded_inputs.push(encode_embedding_input(
                        &loaded_state.tokenizer,
                        &loaded_state.configuration,
                        input_text,
                    )?);
                }
                Ok::<_, EmbeddingsFailureReason>(encoded_inputs)
            },
        )?;
        let (output_embeddings, input_token_counts) = performance_attribution.measure_operation(
            PerformanceOperation::EmbeddingsForwardSpan,
            |_performance_attribution| {
                let mut output_embeddings = Vec::with_capacity(encoded_inputs.len());
                let mut input_token_counts = Vec::with_capacity(encoded_inputs.len());
                for encoded_input in &encoded_inputs {
                    output_embeddings.push(embed_token_ids(
                        &loaded_state.runtime,
                        &loaded_state.tensors,
                        &loaded_state.configuration,
                        &encoded_input.token_ids,
                        embeddings_command.dimensions,
                    )?);
                    input_token_counts
                        .push(u32::try_from(encoded_input.token_ids.len()).unwrap_or(u32::MAX));
                }
                Ok::<_, EmbeddingsFailureReason>((output_embeddings, input_token_counts))
            },
        )?;
        let elapsed_millis = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        let total_input_tokens = input_token_counts
            .iter()
            .fold(0u32, |total, count| total.saturating_add(*count));
        let vector_width = output_embeddings
            .first()
            .map(|components| u32::try_from(components.len()).unwrap_or(u32::MAX))
            .unwrap_or(0);
        let attribution_report = performance_attribution.finish_embeddings(
            PerformanceAttributionOutcome::Success,
            embeddings_command.request_id.value(),
            self.model_id.clone(),
            embeddings_command.inputs.len(),
            total_input_tokens,
            vector_width,
            None,
        );
        self.record_attribution_report(attribution_report);
        Ok(EmbeddingEngineOutput {
            embeddings: output_embeddings,
            input_token_counts,
            elapsed_millis,
        })
    }
}

fn runtime_failure(runtime_error: MlxRuntimeError) -> EmbeddingsFailureReason {
    EmbeddingsFailureReason::FatalExecution {
        reason: format!("embedding runtime operation failed: {runtime_error}")
            .chars()
            .take(256)
            .collect(),
    }
}
