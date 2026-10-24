use crate::{EmbeddingUnavailableEngine, ImageGenerationUnavailableEngine};

pub struct EngineBackedWorker<
    Processor,
    Engine,
    Factory = (),
    ImageEngine = ImageGenerationUnavailableEngine,
    EmbeddingEngine = EmbeddingUnavailableEngine,
> {
    pub(crate) loaded_runtime:
        Option<LoadedRuntime<Processor, Engine, ImageEngine, EmbeddingEngine>>,
    pub(crate) model_factory: Option<Factory>,
    pub(crate) machine_mlx_memory_ceiling_bytes: u64,
    pub(crate) effective_mlx_memory_ceiling_bytes: u64,
    pub(crate) minimum_mlx_memory_ceiling_bytes: u64,
    /// GPU memory macOS granted this worker as a wired pool; `None` when
    /// serving is pageable. Reported in the idle event so the daemon can
    /// name the operating-system bound whenever it clamps the ceiling.
    pub(crate) wired_residency_budget_bytes: Option<u64>,
    pub(crate) worker_runtime_feature_configuration:
        Option<astronomical_ipc_protocol::WorkerRuntimeFeatureConfiguration>,
    pub(crate) selected_model: Option<support::SelectedModel>,
    /// The typed placement lifecycle of the loaded autoregressive engine:
    /// produced by startup engine selection, demoted by a request's
    /// admission, and re-promoted only through the bounded gate. The
    /// streaming-retry path records the demotion; nothing else may write
    /// free-form states because the type has none.
    pub(crate) engine_placement: Option<crate::EnginePlacement>,
    /// Startup residency facts of a loaded streaming engine; the
    /// re-promotion gate replays them against the current effective
    /// ceiling. `None` while a resident engine (or no engine) is loaded.
    pub(crate) startup_residency_facts: Option<crate::StreamingStartupResidencyFacts>,
}

pub(crate) enum LoadedRuntime<Processor, Engine, ImageEngine, EmbeddingEngine> {
    Autoregressive(LoadedModel<Processor, Engine>),
    Image(ImageEngine),
    Embeddings(EmbeddingEngine),
}

pub(crate) struct LoadedModel<Processor, Engine> {
    pub(crate) processor: Processor,
    pub(crate) engine: Engine,
}

mod construction;
mod embeddings;
mod fatal;
mod generation_advance;
mod generation_start;
mod idle_command;
mod image_generation;
mod memory_limit;
mod model_swap;
mod output;
mod promotion;
mod protocol;
mod retry;
mod support;

pub use support::{ModelFactory, ModelFactoryRuntime, WorkerRuntimeError};
