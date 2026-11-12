//! Loads a speech model swapped into a running worker: the engine loads,
//! must report the identity the policy selected, and is announced with the
//! voices it offers.

use astronomical_ipc_protocol::{WorkerEvent, WorkerModelConfiguration};

use super::model_swap::LoadedReplacement;
use super::support::{ModelFactory, ModelFactoryRuntime, WorkerRuntimeError};
use super::{EngineBackedWorker, LoadedRuntime};
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{ImageGenerationEngine, InferenceEngine};

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine, DecisionsEngine, TextToSpeechEngine>
    EngineBackedWorker<
        Processor,
        Engine,
        Factory,
        ImageEngine,
        EmbeddingsEngine,
        DecisionsEngine,
        TextToSpeechEngine,
    >
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingsEngine,
            DecisionsEngine,
            TextToSpeechEngine,
        > + Send
        + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: crate::EmbeddingEngine,
    DecisionsEngine: crate::DecisionEngine,
    TextToSpeechEngine: crate::SpeechEngine,
{
    /// Loads a speech replacement; `load_replacement_runtime` loads every
    /// kind other than the forward-pass ones itself.
    pub(super) fn load_speech_replacement(
        &self,
        factory_runtime: ModelFactoryRuntime<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingsEngine,
            DecisionsEngine,
            TextToSpeechEngine,
        >,
        model_configuration: &WorkerModelConfiguration,
    ) -> Result<
        LoadedReplacement<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingsEngine,
            DecisionsEngine,
            TextToSpeechEngine,
        >,
        WorkerRuntimeError,
    > {
        let ModelFactoryRuntime::Speech(mut speech_engine) = factory_runtime else {
            return Err(WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason: "the selected model is not a forward-pass model"
                    .to_owned(),
            });
        };
        let speech_load_result = speech_engine.load().map_err(|load_failure| {
            tracing::error!(model_id = ?model_configuration.model_id(), reason = ?load_failure, "speech engine load failed after swap creation");
            WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason: load_failure.to_string(),
            }
        })?;
        if speech_load_result.model_id != model_configuration.model_id() {
            return Err(WorkerRuntimeError::ModelSwapFailed {
                model_load_failure_reason:
                    "loaded speech model identity does not match the selected model".to_owned(),
            });
        }
        let minimum = speech_engine.minimum_mlx_memory_ceiling_bytes();
        let event = WorkerEvent::ModelSwapped {
            model_id: speech_load_result.model_id.clone(),
            capabilities: super::speech::speech_model_capabilities(&speech_load_result),
            expert_memory_mode: None,
            minimum_mlx_memory_ceiling_bytes: minimum,
        };
        Ok(LoadedReplacement {
            replacement_runtime: LoadedRuntime::Speech(speech_engine),
            model_swapped_event: event,
            minimum_mlx_memory_ceiling_bytes: minimum,
            runtime_configuration: model_configuration.runtime_configuration(),
            engine_placement: None,
            startup_residency_facts: None,
            neural_engine_prefill_event: None,
        })
    }
}
