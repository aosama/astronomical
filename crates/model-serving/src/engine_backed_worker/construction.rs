use super::support::ModelFactory;
use super::{EngineBackedWorker, LoadedModel, LoadedRuntime};
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{ImageGenerationEngine, InferenceEngine};

impl<Processor, Engine> EngineBackedWorker<Processor, Engine, ()>
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
{
    pub fn new(processor: Processor, engine: Engine) -> Self {
        Self {
            loaded_runtime: Some(LoadedRuntime::Autoregressive(LoadedModel {
                processor,
                engine,
            })),
            model_factory: None,
            machine_mlx_memory_ceiling_bytes: 0,
            effective_mlx_memory_ceiling_bytes: 0,
            minimum_mlx_memory_ceiling_bytes: 1,
            wired_residency_budget_bytes: None,
            worker_runtime_feature_configuration: None,
            selected_model: None,
        }
    }
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
    pub fn with_model_factory(
        processor: Processor,
        engine: Engine,
        model_factory: Factory,
    ) -> Self {
        Self {
            loaded_runtime: Some(LoadedRuntime::Autoregressive(LoadedModel {
                processor,
                engine,
            })),
            model_factory: Some(model_factory),
            machine_mlx_memory_ceiling_bytes: 0,
            effective_mlx_memory_ceiling_bytes: 0,
            minimum_mlx_memory_ceiling_bytes: 1,
            wired_residency_budget_bytes: None,
            worker_runtime_feature_configuration: None,
            selected_model: None,
        }
    }

    pub fn idle_with_model_factory(model_factory: Factory, mlx_memory_ceiling_bytes: u64) -> Self {
        Self::idle_with_model_factory_and_machine_mlx_memory_ceiling(
            model_factory,
            mlx_memory_ceiling_bytes,
            mlx_memory_ceiling_bytes,
        )
    }

    pub fn idle_with_model_factory_and_machine_mlx_memory_ceiling(
        model_factory: Factory,
        machine_mlx_memory_ceiling_bytes: u64,
        effective_mlx_memory_ceiling_bytes: u64,
    ) -> Self {
        Self {
            loaded_runtime: None,
            model_factory: Some(model_factory),
            machine_mlx_memory_ceiling_bytes,
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes: 1,
            wired_residency_budget_bytes: None,
            worker_runtime_feature_configuration: None,
            selected_model: None,
        }
    }

    /// Records the wired-residency budget granted at worker startup so the
    /// daemon's status surface can name the operating-system bound.
    #[must_use]
    pub fn with_wired_residency_budget_bytes(
        mut self,
        wired_residency_budget_bytes: Option<u64>,
    ) -> Self {
        self.wired_residency_budget_bytes = wired_residency_budget_bytes;
        self
    }

    /// Attaches the worker-startup feature policy that must be acknowledged to the daemon.
    #[must_use]
    pub fn with_worker_runtime_feature_configuration(
        mut self,
        worker_runtime_feature_configuration: astronomical_ipc_protocol::WorkerRuntimeFeatureConfiguration,
    ) -> Self {
        self.worker_runtime_feature_configuration = Some(worker_runtime_feature_configuration);
        self
    }
}
