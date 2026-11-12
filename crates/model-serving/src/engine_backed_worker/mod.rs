use astronomical_ipc_protocol::WorkerEvent;

use crate::{EmbeddingUnavailableEngine, ImageGenerationUnavailableEngine};

pub struct EngineBackedWorker<
    Processor,
    Engine,
    Factory = (),
    ImageEngine = ImageGenerationUnavailableEngine,
    EmbeddingEngine = EmbeddingUnavailableEngine,
    DecisionsEngine = crate::DecisionUnavailableEngine,
    TextToSpeechEngine = crate::SpeechUnavailableEngine,
> {
    pub(crate) loaded_runtime: Option<
        LoadedRuntime<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingEngine,
            DecisionsEngine,
            TextToSpeechEngine,
        >,
    >,
    pub(crate) model_factory: Option<Factory>,
    pub(crate) machine_mlx_memory_ceiling_bytes: u64,
    pub(crate) effective_mlx_memory_ceiling_bytes: u64,
    pub(crate) minimum_mlx_memory_ceiling_bytes: u64,
    /// GPU memory macOS granted this worker as a wired pool; `None` when
    /// serving is pageable. Reported in the idle event so the daemon can
    /// name the operating-system bound whenever it clamps the ceiling.
    pub(crate) wired_residency_budget_bytes: Option<u64>,
    /// The highest ceiling a loaded model may be raised to so it can serve at
    /// its floor (#1902): the user's own limit when they set one below the
    /// machine, otherwise unbounded. Zero until startup reports it, which
    /// never raises.
    pub(crate) floor_raise_limit_bytes: u64,
    /// The ceiling the model factory holds: never raised to a model's floor,
    /// so each newly loaded model is judged against it rather than against a
    /// floor an earlier model was raised to.
    pub(crate) factory_mlx_memory_ceiling_bytes: u64,
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
    /// Programs are still being prepared beside the serving model.
    pub(crate) is_neural_engine_preparation_running: bool,
    /// Only a request can teach anything about running programs, so an idle worker
    /// that has served none never crosses to the owner thread to ask.
    pub(crate) has_request_run_since_settling: bool,
    /// A status from a load made during a generation, which has no event stream to
    /// hand, held so the idle loop remains the one place these reports leave from.
    pub(crate) unsent_neural_engine_prefill_event: Option<WorkerEvent>,
}

pub(crate) enum LoadedRuntime<
    Processor,
    Engine,
    ImageEngine,
    EmbeddingEngine,
    DecisionsEngine,
    TextToSpeechEngine,
> {
    Autoregressive(LoadedModel<Processor, Engine>),
    Image(ImageEngine),
    Embeddings(EmbeddingEngine),
    Decisions(DecisionsEngine),
    /// Held boxed rather than as another type parameter: every family is
    /// one trait object, and the worker's generic list stays as it is.
    Speech(TextToSpeechEngine),
}

pub(crate) struct LoadedModel<Processor, Engine> {
    pub(crate) processor: Processor,
    pub(crate) engine: Engine,
}

mod construction;
mod decisions;
mod embeddings;
mod fatal;
mod forward_pass_memory;
mod generation_advance;
mod generation_advance_prefill;
mod generation_start;
mod idle_command;
mod image_generation;
mod load_announcement;
mod memory_limit;
mod model_swap;
mod neural_engine_preparation_relay;
mod output;
mod promotion;
mod protocol;
mod retry;
mod speech;
mod speech_model_swap;
mod stepped_request;
mod support;
mod token_count;

pub use support::{ModelFactory, ModelFactoryRuntime, WorkerRuntimeError};

#[cfg(test)]
mod tests;
