//! Transactional typed runtime replacement for autoregressive and image models.

use astronomical_ipc_protocol::{
    ImageGenerationFailureReason, MlxMemorySnapshotSource, MtpRuntimeState, ProtocolWriter,
    SpeculativePrefillRuntimeState, WorkerEvent, WorkerLoadedModelRuntimeConfiguration,
    WorkerModelCapabilities, WorkerModelConfiguration,
};
use tokio::io::AsyncWrite;

use super::support::{ModelFactory, ModelFactoryRuntime, WorkerRuntimeError};
use super::{EngineBackedWorker, LoadedModel, LoadedRuntime};
use crate::{ImageGenerationEngine, InferenceEngine, ModelGenerationProcessor};

impl<Processor, Engine, Factory, ImageEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine> + Send + 'static,
    ImageEngine: ImageGenerationEngine,
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
        tracing::info!(
            model_directory,
            model_id = model_configuration.model_id(),
            "starting model swap"
        );
        let factory_runtime = model_factory
            .create(model_directory, model_configuration.clone())
            .await
            .map_err(
                |model_load_failure_reason| WorkerRuntimeError::ModelSwapFailed {
                    model_load_failure_reason,
                },
            )?;

        if !factory_runtime_matches_configuration(&factory_runtime, &model_configuration) {
            return Err(WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason:
                    "model factory returned a runtime for the wrong modality".to_owned(),
            });
        }

        // Creation and modality validation preserve the prior runtime on a rejected selection.
        // Once validated, release it before load so large model payloads never overlap.
        drop(self.loaded_runtime.take());
        let (
            replacement_runtime,
            model_swapped_event,
            minimum_mlx_memory_ceiling_bytes,
            runtime_configuration,
        ) = match factory_runtime {
            ModelFactoryRuntime::Autoregressive { processor, engine } => {
                let mut replacement_model = LoadedModel { processor, engine };
                let engine_load_result = replacement_model.engine.load().await.map_err(|engine_error| {
                        tracing::error!(model_directory, error = %engine_error, "model engine load failed after swap creation");
                        WorkerRuntimeError::ModelSwapFailed {
                            model_load_failure_reason: "model engine initialization failed".to_owned(),
                        }
                    })?;
                let minimum = engine_load_result.minimum_mlx_memory_ceiling_bytes();
                let event = model_swapped_from_ready_event(
                    replacement_model.processor.ready_event(
                        engine_load_result.mtp_runtime_state(),
                        engine_load_result
                            .mtp_unavailable_reason()
                            .map(String::from),
                        engine_load_result.mtp_depth_status(),
                        engine_load_result.speculative_prefill_runtime_state(),
                        engine_load_result
                            .speculative_prefill_unavailable_reason()
                            .map(String::from),
                        engine_load_result
                            .speculative_prefill_draft_model_id()
                            .map(String::from),
                        engine_load_result
                            .speculative_prefill_draft_model_revision()
                            .map(String::from),
                    ),
                    engine_load_result.expert_memory_mode(),
                    minimum,
                )?;
                let mut runtime_configuration = model_configuration.runtime_configuration();
                if let WorkerLoadedModelRuntimeConfiguration::Autoregressive(configuration) =
                    &mut runtime_configuration
                {
                    configuration.speculative_prefill_enabled = !matches!(
                        engine_load_result.speculative_prefill_runtime_state(),
                        SpeculativePrefillRuntimeState::Disabled
                    );
                    if !configuration.speculative_prefill_enabled {
                        configuration.speculative_prefill = None;
                    }
                }
                (
                    LoadedRuntime::Autoregressive(replacement_model),
                    event,
                    minimum,
                    runtime_configuration,
                )
            }
            ModelFactoryRuntime::Image(mut image_engine) => {
                let image_load_result = image_engine.load().map_err(|failure_reason| {
                        tracing::error!(model_directory, reason = ?failure_reason, "image engine load failed after swap creation");
                        WorkerRuntimeError::ModelSwapFailed {
                            model_load_failure_reason: bounded_image_engine_load_failure(
                                failure_reason,
                            ),
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
                    mtp_runtime_state: MtpRuntimeState::Disabled,
                    mtp_unavailable_reason: None,
                    mtp_depth_status: Default::default(),
                    speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
                    speculative_prefill_unavailable_reason: None,
                    speculative_prefill_draft_model_id: None,
                    speculative_prefill_draft_model_revision: None,
                };
                (
                    LoadedRuntime::Image(image_engine),
                    event,
                    minimum,
                    model_configuration.runtime_configuration(),
                )
            }
        };

        self.loaded_runtime = Some(replacement_runtime);
        self.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
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

fn factory_runtime_matches_configuration<Processor, Engine, ImageEngine>(
    factory_runtime: &ModelFactoryRuntime<Processor, Engine, ImageEngine>,
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
        )
    )
}

fn model_swapped_from_ready_event(
    ready_event: WorkerEvent,
    expert_memory_mode: Option<astronomical_ipc_protocol::ExpertMemoryMode>,
    minimum_mlx_memory_ceiling_bytes: u64,
) -> Result<WorkerEvent, WorkerRuntimeError> {
    match ready_event {
        WorkerEvent::Ready {
            model_id,
            capabilities,
            mtp_runtime_state,
            mtp_unavailable_reason,
            mtp_depth_status,
            speculative_prefill_runtime_state,
            speculative_prefill_unavailable_reason,
            speculative_prefill_draft_model_id,
            speculative_prefill_draft_model_revision,
        } => Ok(WorkerEvent::ModelSwapped {
            model_id,
            capabilities,
            expert_memory_mode,
            minimum_mlx_memory_ceiling_bytes,
            mtp_runtime_state,
            mtp_unavailable_reason,
            mtp_depth_status,
            speculative_prefill_runtime_state,
            speculative_prefill_unavailable_reason,
            speculative_prefill_draft_model_id,
            speculative_prefill_draft_model_revision,
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
