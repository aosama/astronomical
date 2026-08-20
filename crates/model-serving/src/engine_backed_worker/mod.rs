use crate::ImageGenerationUnavailableEngine;

pub struct EngineBackedWorker<
    Processor,
    Engine,
    Factory = (),
    ImageEngine = ImageGenerationUnavailableEngine,
> {
    pub(crate) loaded_runtime: Option<LoadedRuntime<Processor, Engine, ImageEngine>>,
    pub(crate) model_factory: Option<Factory>,
    pub(crate) machine_mlx_memory_ceiling_bytes: u64,
    pub(crate) effective_mlx_memory_ceiling_bytes: u64,
    pub(crate) minimum_mlx_memory_ceiling_bytes: u64,
    pub(crate) worker_runtime_feature_configuration:
        Option<astronomical_ipc_protocol::WorkerRuntimeFeatureConfiguration>,
}

pub(crate) enum LoadedRuntime<Processor, Engine, ImageEngine> {
    Autoregressive(LoadedModel<Processor, Engine>),
    Image(ImageEngine),
}

pub(crate) struct LoadedModel<Processor, Engine> {
    pub(crate) processor: Processor,
    pub(crate) engine: Engine,
}

mod construction;
mod fatal;
mod generation_advance;
mod generation_start;
mod idle_command;
mod image_generation;
mod memory_limit;
mod model_swap;
mod output;
mod protocol;
mod support;

pub use support::{ModelFactory, ModelFactoryRuntime, WorkerRuntimeError};
