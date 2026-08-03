pub struct EngineBackedWorker<Processor, Engine, Factory = ()> {
    pub(crate) loaded_model: Option<LoadedModel<Processor, Engine>>,
    pub(crate) model_factory: Option<Factory>,
    pub(crate) machine_mlx_memory_ceiling_bytes: u64,
    pub(crate) effective_mlx_memory_ceiling_bytes: u64,
    pub(crate) minimum_mlx_memory_ceiling_bytes: u64,
}

pub(crate) struct LoadedModel<Processor, Engine> {
    pub(crate) processor: Processor,
    pub(crate) engine: Engine,
}

mod generation_advance;
mod generation_start;
mod idle_command;
mod memory_limit;
mod model_swap;
