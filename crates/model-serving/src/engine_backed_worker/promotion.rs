//! Bounded re-promotion of a demoted streaming worker to the resident engine.
//!
//! The re-promotion gate in `generation_start` decides *whether* a demoted
//! worker may return to complete residency; this module performs the swap
//! once that decision is `PromoteToResident`. The swap mirrors the
//! streaming-retry swap's transactional ordering exactly — create and
//! validate the replacement runtime before releasing the live one — so a
//! failure before the drop leaves the streaming engine untouched and the
//! request simply continues streaming, while a failure after the drop is
//! fatal for the worker.

use astronomical_ipc_protocol::RequestId;

use super::model_swap::factory_runtime_matches_configuration;
use super::support::{ModelFactory, ModelFactoryRuntime, SelectedModel};
use super::{EngineBackedWorker, LoadedModel, LoadedRuntime};
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
    /// Swaps a demoted streaming runtime for the resident engine.
    ///
    /// Returns `Ok(true)` after the promotion is recorded. Returns
    /// `Ok(false)`, with the streaming runtime untouched and the reason
    /// logged, when the resident runtime cannot even be created. Returns
    /// `Err` only when the worker is left without a runtime, which the
    /// caller must treat as a fatal engine error for the request.
    pub(super) async fn promote_demoted_engine_to_resident(
        &mut self,
        request_id: RequestId,
    ) -> Result<bool, InferenceEngineError> {
        let Some(SelectedModel {
            model_directory,
            model_configuration,
        }) = self.selected_model.clone()
        else {
            tracing::warn!(
                request_id = request_id.value(),
                "resident promotion skipped: no selected model is recorded"
            );
            return Ok(false);
        };
        let Some(model_factory) = self.model_factory.as_ref() else {
            tracing::warn!(
                request_id = request_id.value(),
                "resident promotion skipped: no model factory is configured"
            );
            return Ok(false);
        };
        // Creation failures happen before the live runtime is released, so
        // the demoted streaming engine keeps serving this request.
        let factory_runtime = match model_factory
            .create_resident_promotion(&model_directory, model_configuration.clone())
            .await
        {
            Ok(factory_runtime) => factory_runtime,
            Err(creation_failure_reason) => {
                tracing::warn!(
                    request_id = request_id.value(),
                    creation_failure_reason,
                    "resident promotion skipped: resident engine creation failed"
                );
                return Ok(false);
            }
        };
        if !factory_runtime_matches_configuration(&factory_runtime, &model_configuration) {
            tracing::warn!(
                request_id = request_id.value(),
                "resident promotion skipped: factory returned the wrong runtime modality"
            );
            return Ok(false);
        }
        // Creation and modality validation preserve the streaming runtime on
        // a rejected selection. Once validated, release it before load so
        // large model payloads never overlap.
        drop(self.loaded_runtime.take());
        let ModelFactoryRuntime::Autoregressive {
            processor,
            engine,
            engine_placement: _,
        } = factory_runtime
        else {
            return Err(InferenceEngineError::Fatal {
                reason: "resident promotion factory returned a non-autoregressive runtime"
                    .to_owned(),
            });
        };
        let mut replacement_model = LoadedModel { processor, engine };
        let engine_load_result = replacement_model
            .engine
            .load()
            .await
            .map_err(|engine_error| InferenceEngineError::Fatal {
                reason: format!("resident promotion engine initialization failed: {engine_error}"),
            })?;
        self.minimum_mlx_memory_ceiling_bytes =
            engine_load_result.minimum_mlx_memory_ceiling_bytes();
        self.loaded_runtime = Some(LoadedRuntime::Autoregressive(replacement_model));
        // The gate already proved complete residency fits at the effective
        // ceiling, and the promotion runtime was built by the resident
        // selection, so the lifecycle records the promotion itself.
        self.engine_placement = Some(crate::EnginePlacement::PromotedAfterPressureResolved);
        self.startup_residency_facts = engine_load_result.startup_residency_facts();
        Ok(true)
    }
}
