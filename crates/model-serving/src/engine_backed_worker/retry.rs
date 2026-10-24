use astronomical_ipc_protocol::RequestId;

use super::model_swap::factory_runtime_matches_configuration;
use super::support::{ModelFactory, ModelFactoryRuntime, SelectedModel};
use super::{EngineBackedWorker, LoadedModel, LoadedRuntime};
use crate::inference_engine::contract::PreparedInferenceRequest;
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{EmbeddingEngine, ImageGenerationEngine, InferenceEngine, InferenceEngineError};

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
    EngineBackedWorker<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<Processor, Engine, ImageEngine, EmbeddingsEngine> + Send + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: EmbeddingEngine,
{
    pub(super) async fn retry_resident_generation_on_streaming(
        &mut self,
        request_id: RequestId,
        mut retry_request: Processor::InferenceRequest,
        demotion_cause: crate::StreamingEngineRequirement,
        demoted_total_context_token_count: usize,
        resident_fork_reason: String,
    ) -> Result<crate::inference_engine::contract::EngineGenerationStart, InferenceEngineError>
    {
        let Some(SelectedModel {
            model_directory,
            model_configuration,
        }) = self.selected_model.clone()
        else {
            return Err(InferenceEngineError::InvalidRequest {
                reason: resident_fork_reason,
            });
        };
        let Some(model_factory) = self.model_factory.as_ref() else {
            return Err(InferenceEngineError::InvalidRequest {
                reason: resident_fork_reason,
            });
        };
        if !model_factory.supports_streaming_retry() {
            return Err(InferenceEngineError::InvalidRequest {
                reason: resident_fork_reason,
            });
        }

        let retry_operation_log = StreamingRetryPerformanceLog::start(
            model_factory.performance_attribution_enabled(),
            request_id,
        );

        drop(self.loaded_runtime.take());
        let factory_runtime = model_factory
            .create_streaming_retry(&model_directory, model_configuration.clone())
            .await
            .map_err(|reason| InferenceEngineError::Fatal {
                reason: format!("streaming retry engine creation failed: {reason}"),
            })?;
        if !factory_runtime_matches_configuration(&factory_runtime, &model_configuration) {
            return Err(InferenceEngineError::Fatal {
                reason: "streaming retry factory returned the wrong runtime modality".to_owned(),
            });
        }
        // A streaming-retry runtime carries no startup placement: the worker
        // records the typed `DemotedForRequest` placement below, once the
        // streaming engine has loaded.
        let ModelFactoryRuntime::Autoregressive {
            processor,
            engine,
            engine_placement: _,
        } = factory_runtime
        else {
            return Err(InferenceEngineError::Fatal {
                reason: "streaming retry factory returned a non-autoregressive runtime".to_owned(),
            });
        };
        let mut replacement_model = LoadedModel { processor, engine };
        let engine_load_result = replacement_model
            .engine
            .load()
            .await
            .map_err(|engine_error| InferenceEngineError::Fatal {
                reason: format!("streaming retry engine initialization failed: {engine_error}"),
            })?;
        self.minimum_mlx_memory_ceiling_bytes =
            engine_load_result.minimum_mlx_memory_ceiling_bytes();
        self.loaded_runtime = Some(LoadedRuntime::Autoregressive(replacement_model));
        // The swap is complete: the placement lifecycle records the typed
        // request demotion, and the streaming engine's own startup residency
        // facts are kept so a later re-promotion gate can replay them.
        let demoted_engine_placement = crate::EnginePlacement::DemotedForRequest {
            cause: demotion_cause,
            demoted_total_context_token_count,
        };
        self.engine_placement = Some(demoted_engine_placement);
        self.startup_residency_facts = engine_load_result.startup_residency_facts();
        tracing::info!(
            request_id = request_id.value(),
            engine_placement = demoted_engine_placement.log_label(),
            demotion_cause = ?demotion_cause,
            demoted_total_context_token_count,
            "engine placement demoted to streaming for this request"
        );

        if let Some(retry_operation_started_at) = retry_operation_log.started_at {
            retry_request.record_streaming_retry_interval(
                retry_operation_started_at,
                std::time::Instant::now(),
            );
        }
        let generation_start = {
            let Some(LoadedRuntime::Autoregressive(loaded_model)) = self.loaded_runtime.as_mut()
            else {
                return Err(InferenceEngineError::Fatal {
                    reason: "streaming retry runtime was removed before request replay".to_owned(),
                });
            };
            loaded_model.engine.start_generation(retry_request).await
        };
        match generation_start {
            Ok(generation_start) => Ok(generation_start),
            // The demotion has already been recorded above, so the streaming
            // replay failing its own admission cannot fork again: the
            // request is answered with the rejection reason instead.
            Err(InferenceEngineError::ResidentForkRequired { reason, .. }) => {
                Err(InferenceEngineError::InvalidRequest { reason })
            }
            Err(engine_error) => Err(engine_error),
        }
    }
}

struct StreamingRetryPerformanceLog {
    request_id: RequestId,
    started_at: Option<std::time::Instant>,
}

impl StreamingRetryPerformanceLog {
    fn start(is_enabled: bool, request_id: RequestId) -> Self {
        let started_at = is_enabled.then(std::time::Instant::now);
        if started_at.is_some() {
            tracing::info!(
                operation = "resident_to_streaming_retry",
                phase = "start",
                request_id = request_id.value(),
                "performance attribution operation started"
            );
        }
        Self {
            request_id,
            started_at,
        }
    }
}

impl Drop for StreamingRetryPerformanceLog {
    fn drop(&mut self) {
        if let Some(started_at) = self.started_at {
            tracing::info!(
                operation = "resident_to_streaming_retry",
                phase = "end",
                request_id = self.request_id.value(),
                elapsed_millis = started_at.elapsed().as_millis(),
                "performance attribution operation completed"
            );
        }
    }
}
