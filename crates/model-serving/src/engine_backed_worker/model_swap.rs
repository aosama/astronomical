//! Transactional typed runtime replacement for autoregressive and image models.

use astronomical_ipc_protocol::{
    ImageGenerationFailureReason, MlxMemorySnapshotSource, ProtocolWriter, WorkerEvent,
    WorkerModelCapabilities, WorkerModelConfiguration,
};
use tokio::io::AsyncWrite;

use super::support::{ModelFactory, ModelFactoryRuntime, SelectedModel, WorkerRuntimeError};
use super::{EngineBackedWorker, LoadedModel, LoadedRuntime};
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{EnginePlacement, ImageGenerationEngine, InferenceEngine};

/// Emits paired timing boundaries for one model-swap phase only when the
/// factory has performance attribution enabled, mirroring the prompt-cache
/// clear attribution log. The two phases are the worker-level envelope around
/// the engine's own finer-grained attribution segments: factory creation
/// (model-family classification, artifact validation, tokenizer
/// initialization, engine selection and construction) and the engine load
/// (runtime initialization, safetensors mapping, tensor binding).
struct ModelSwapPhasePerformanceLog {
    operation: &'static str,
    model_id: Option<String>,
    started_at: Option<std::time::Instant>,
}

impl ModelSwapPhasePerformanceLog {
    fn start(is_enabled: bool, operation: &'static str, model_id: &str) -> Self {
        if !is_enabled {
            return Self {
                operation,
                model_id: None,
                started_at: None,
            };
        }
        tracing::info!(
            operation,
            phase = "start",
            model_id,
            "performance attribution operation started"
        );
        Self {
            operation,
            model_id: Some(model_id.to_owned()),
            started_at: Some(std::time::Instant::now()),
        }
    }
}

impl Drop for ModelSwapPhasePerformanceLog {
    fn drop(&mut self) {
        if let Some(started_at) = self.started_at {
            tracing::info!(
                operation = self.operation,
                phase = "end",
                elapsed_millis = started_at.elapsed().as_millis(),
                model_id = self.model_id.as_deref(),
                "performance attribution operation completed"
            );
        }
    }
}

// The struct lives at module scope: an impl block cannot contain items.
/// One replacement runtime fully loaded: the runtime itself, the
/// ModelSwapped event to publish, the ceiling it enforces, and the
/// autoregressive facts the worker state records.
struct LoadedReplacement<
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine + Send + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: crate::EmbeddingEngine,
> {
    replacement_runtime: LoadedRuntime<Processor, Engine, ImageEngine, EmbeddingsEngine>,
    model_swapped_event: WorkerEvent,
    minimum_mlx_memory_ceiling_bytes: u64,
    runtime_configuration: astronomical_ipc_protocol::WorkerLoadedModelRuntimeConfiguration,
    engine_placement: Option<EnginePlacement>,
    startup_residency_facts: Option<crate::StreamingStartupResidencyFacts>,
}

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine, EmbeddingsEngine> + Send + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: crate::EmbeddingEngine,
{
    pub(super) async fn swap_model<WriteTransport>(
        &mut self,
        model_directory: &str,
        model_configuration: WorkerModelConfiguration,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let Some(model_factory) = self.model_factory.as_ref() else {
            return Err(WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason: "model swapping is unavailable".to_owned(),
            });
        };
        let performance_attribution_enabled = model_factory.performance_attribution_enabled();
        tracing::info!(
            model_directory,
            model_id = model_configuration.model_id(),
            "starting model swap"
        );
        let factory_runtime = {
            let _factory_create_attribution = ModelSwapPhasePerformanceLog::start(
                performance_attribution_enabled,
                "model_swap_factory_create",
                model_configuration.model_id(),
            );
            model_factory
                .create(model_directory, model_configuration.clone())
                .await
                .map_err(
                    |model_load_failure_reason| WorkerRuntimeError::ModelSwapFailed {
                        model_load_failure_reason,
                    },
                )?
        };

        if !factory_runtime_matches_configuration(&factory_runtime, &model_configuration) {
            return Err(WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason:
                    "model factory returned a runtime for the wrong modality".to_owned(),
            });
        }
        if matches!(
            factory_runtime,
            ModelFactoryRuntime::Autoregressive {
                engine_placement: None,
                ..
            }
        ) {
            return Err(WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason:
                    "model factory returned a swap-only runtime without a startup engine placement"
                        .to_owned(),
            });
        }

        // Creation and modality validation preserve the prior runtime on a rejected selection.
        // Once validated, release it before load so large model payloads never overlap.
        drop(self.loaded_runtime.take());
        let LoadedReplacement {
            replacement_runtime,
            model_swapped_event,
            minimum_mlx_memory_ceiling_bytes,
            runtime_configuration,
            engine_placement,
            startup_residency_facts,
        } = self
            .load_replacement_runtime(factory_runtime, &model_configuration)
            .await?;

        self.loaded_runtime = Some(replacement_runtime);
        self.selected_model = Some(SelectedModel {
            model_directory: model_directory.to_owned(),
            model_configuration: model_configuration.clone(),
        });
        self.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
        self.engine_placement = engine_placement;
        self.startup_residency_facts = startup_residency_facts;
        tracing::info!(
            model_id = model_configuration.model_id(),
            engine_placement = self.engine_placement.map_or(
                "no_autoregressive_engine",
                crate::EnginePlacement::log_label
            ),
            "engine placement recorded after model swap"
        );
        event_writer.send_event(&model_swapped_event).await?;
        if let Some(mut worker_runtime_feature_configuration) =
            self.worker_runtime_feature_configuration()
        {
            worker_runtime_feature_configuration.loaded_model = Some(runtime_configuration);
            event_writer
                .send_event(&WorkerEvent::RuntimeFeatureConfigurationApplied {
                    worker_runtime_feature_configuration,
                })
                .await?;
        }
        self.emit_mlx_memory_sample(MlxMemorySnapshotSource::ModelLoaded, event_writer)
            .await?;
        self.emit_persistent_prompt_cache_stats(event_writer)
            .await?;
        Ok(())
    }

    /// Loads whichever replacement runtime the factory produced: the
    /// autoregressive engine loads under attribution while the image and
    /// embedding engines load synchronously with identity verification.
    async fn load_replacement_runtime(
        &self,
        factory_runtime: ModelFactoryRuntime<Processor, Engine, ImageEngine, EmbeddingsEngine>,
        model_configuration: &WorkerModelConfiguration,
    ) -> Result<
        LoadedReplacement<Processor, Engine, ImageEngine, EmbeddingsEngine>,
        WorkerRuntimeError,
    > {
        match factory_runtime {
            ModelFactoryRuntime::Autoregressive {
                processor,
                engine,
                engine_placement,
            } => {
                self.load_autoregressive_replacement(
                    processor,
                    engine,
                    engine_placement,
                    model_configuration,
                )
                .await
            }
            ModelFactoryRuntime::Image(image_engine) => {
                self.load_image_replacement(image_engine, model_configuration)
                    .await
            }
            ModelFactoryRuntime::Embeddings(embedding_engine) => {
                self.load_embedding_replacement(embedding_engine, model_configuration)
                    .await
            }
        }
    }

    /// Loads the autoregressive replacement: the engine load is attributed, its
    /// ready event is derived from the processor, and the startup residency
    /// facts carry into the worker state.
    async fn load_autoregressive_replacement(
        &self,
        processor: Processor,
        engine: Engine,
        engine_placement: Option<EnginePlacement>,
        model_configuration: &WorkerModelConfiguration,
    ) -> Result<
        LoadedReplacement<Processor, Engine, ImageEngine, EmbeddingsEngine>,
        WorkerRuntimeError,
    > {
        let mut replacement_model = LoadedModel { processor, engine };
        let engine_load_result = replacement_model.engine.load().await.map_err(|engine_error| {
                tracing::error!(model_directory = ?model_configuration.model_id(), error = %engine_error, "model engine load failed after swap creation");
                WorkerRuntimeError::ModelSwapFailed {
                    model_load_failure_reason: "model engine initialization failed".to_owned(),
                }
            })?;
        let minimum = engine_load_result.minimum_mlx_memory_ceiling_bytes();
        let event = model_swapped_from_ready_event(
            replacement_model.processor.ready_event(),
            engine_load_result.expert_memory_mode(),
            minimum,
        )?;
        let runtime_configuration = model_configuration.runtime_configuration();
        Ok(LoadedReplacement {
            replacement_runtime: LoadedRuntime::Autoregressive(replacement_model),
            model_swapped_event: event,
            minimum_mlx_memory_ceiling_bytes: minimum,
            runtime_configuration,
            engine_placement,
            startup_residency_facts: engine_load_result.startup_residency_facts(),
        })
    }

    /// Loads the image replacement and verifies the loaded identity matches the
    /// selected model.
    async fn load_image_replacement(
        &self,
        mut image_engine: ImageEngine,
        model_configuration: &WorkerModelConfiguration,
    ) -> Result<
        LoadedReplacement<Processor, Engine, ImageEngine, EmbeddingsEngine>,
        WorkerRuntimeError,
    > {
        let image_load_result = image_engine.load().map_err(|failure_reason| {
                tracing::error!(model_directory = ?model_configuration.model_id(), reason = ?failure_reason, "image engine load failed after swap creation");
                WorkerRuntimeError::ModelSwapFailed {
                    model_load_failure_reason: bounded_image_engine_load_failure(failure_reason),
                }
            })?;
        if image_load_result.model_id() != model_configuration.model_id() {
            return Err(WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason:
                    "loaded image identity does not match the selected model".to_owned(),
            });
        }
        let minimum = image_load_result.minimum_mlx_memory_ceiling_bytes();
        let event = WorkerEvent::ModelSwapped {
            model_id: image_load_result.model_id().to_owned(),
            capabilities: WorkerModelCapabilities::image_generation(
                image_load_result.capabilities().clone(),
            ),
            expert_memory_mode: None,
            minimum_mlx_memory_ceiling_bytes: minimum,
        };
        Ok(LoadedReplacement {
            replacement_runtime: LoadedRuntime::Image(image_engine),
            model_swapped_event: event,
            minimum_mlx_memory_ceiling_bytes: minimum,
            runtime_configuration: model_configuration.runtime_configuration(),
            // No autoregressive engine is loaded, so there is no placement.
            engine_placement: None,
            startup_residency_facts: None,
        })
    }

    /// Loads the embeddings replacement and verifies the loaded identity matches
    /// the selected model.
    async fn load_embedding_replacement(
        &self,
        mut embedding_engine: EmbeddingsEngine,
        model_configuration: &WorkerModelConfiguration,
    ) -> Result<
        LoadedReplacement<Processor, Engine, ImageEngine, EmbeddingsEngine>,
        WorkerRuntimeError,
    > {
        let embedding_load_result = embedding_engine.load().map_err(|failure_reason| {
                tracing::error!(model_directory = ?model_configuration.model_id(), reason = ?failure_reason, "embedding engine load failed after swap creation");
                WorkerRuntimeError::ModelSwapFailed {
                    model_load_failure_reason: bounded_embedding_engine_load_failure(failure_reason),
                }
            })?;
        if embedding_load_result.model_id() != model_configuration.model_id() {
            return Err(WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason:
                    "loaded embedding identity does not match the selected model".to_owned(),
            });
        }
        let minimum = embedding_load_result.minimum_mlx_memory_ceiling_bytes();
        let event = WorkerEvent::ModelSwapped {
            model_id: embedding_load_result.model_id().to_owned(),
            capabilities: WorkerModelCapabilities::embeddings(
                *embedding_load_result.capabilities(),
            ),
            expert_memory_mode: None,
            minimum_mlx_memory_ceiling_bytes: minimum,
        };
        Ok(LoadedReplacement {
            replacement_runtime: LoadedRuntime::Embeddings(embedding_engine),
            model_swapped_event: event,
            minimum_mlx_memory_ceiling_bytes: minimum,
            runtime_configuration: model_configuration.runtime_configuration(),
            // No autoregressive engine is loaded, so there is no placement.
            engine_placement: None,
            startup_residency_facts: None,
        })
    }
}

fn bounded_image_engine_load_failure(failure_reason: ImageGenerationFailureReason) -> String {
    let failure_detail = match failure_reason {
        ImageGenerationFailureReason::InvalidRequest { reason }
        | ImageGenerationFailureReason::EncodingFailed { reason }
        | ImageGenerationFailureReason::FatalExecution { reason } => reason,
        ImageGenerationFailureReason::ModelDoesNotSupportImageGeneration => {
            "the selected model does not support image generation".to_owned()
        }
        ImageGenerationFailureReason::EngineBusy => "the image engine is busy".to_owned(),
        ImageGenerationFailureReason::Cancelled => "image engine loading was cancelled".to_owned(),
    };
    format!("image engine initialization failed: {failure_detail}")
        .chars()
        .take(512)
        .collect()
}

fn bounded_embedding_engine_load_failure(
    failure_reason: astronomical_ipc_protocol::EmbeddingsFailureReason,
) -> String {
    let failure_detail = match failure_reason {
        astronomical_ipc_protocol::EmbeddingsFailureReason::InvalidRequest { reason }
        | astronomical_ipc_protocol::EmbeddingsFailureReason::FatalExecution { reason } => reason,
        astronomical_ipc_protocol::EmbeddingsFailureReason::MalformedModelOutput => {
            "the embedding engine produced a malformed output".to_owned()
        }
        astronomical_ipc_protocol::EmbeddingsFailureReason::ContextLengthExceeded {
            actual_total_context_tokens,
            maximum_context_tokens,
        } => format!(
            "embedding input has {actual_total_context_tokens} tokens, exceeding the {maximum_context_tokens}-token encoder context"
        ),
        astronomical_ipc_protocol::EmbeddingsFailureReason::EngineBusy => {
            "the embedding engine is busy".to_owned()
        }
    };
    format!("embedding engine initialization failed: {failure_detail}")
        .chars()
        .take(512)
        .collect()
}

pub(super) fn factory_runtime_matches_configuration<
    Processor,
    Engine,
    ImageEngine,
    EmbeddingsEngine,
>(
    factory_runtime: &ModelFactoryRuntime<Processor, Engine, ImageEngine, EmbeddingsEngine>,
    model_configuration: &WorkerModelConfiguration,
) -> bool {
    matches!(
        (factory_runtime, model_configuration),
        (
            ModelFactoryRuntime::Autoregressive { .. },
            WorkerModelConfiguration::Autoregressive(_)
        ) | (
            ModelFactoryRuntime::Image(_),
            WorkerModelConfiguration::Flux2Klein(_)
        ) | (
            ModelFactoryRuntime::Image(_),
            WorkerModelConfiguration::QwenImage21(_)
        ) | (
            ModelFactoryRuntime::Embeddings(_),
            WorkerModelConfiguration::Embeddings(_)
        )
    )
}

pub(super) fn model_swapped_from_ready_event(
    ready_event: WorkerEvent,
    expert_memory_mode: Option<astronomical_ipc_protocol::ExpertMemoryMode>,
    minimum_mlx_memory_ceiling_bytes: u64,
) -> Result<WorkerEvent, WorkerRuntimeError> {
    match ready_event {
        WorkerEvent::Ready {
            model_id,
            capabilities,
        } => Ok(WorkerEvent::ModelSwapped {
            model_id,
            capabilities,
            expert_memory_mode,
            minimum_mlx_memory_ceiling_bytes,
        }),
        other_event => {
            tracing::error!(
                ?other_event,
                "expected Ready event from new processor after swap"
            );
            Err(WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason: "model processor did not become ready".to_owned(),
            })
        }
    }
}
