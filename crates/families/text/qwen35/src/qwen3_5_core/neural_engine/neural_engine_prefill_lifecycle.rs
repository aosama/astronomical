//! What an engine keeps about Neural Engine prompt reading for its loaded
//! model, and the three moments it tells it about: the model finished
//! loading, the GPU was used, and the engine is idle between requests.
//!
//! Both text engines hold one of these and call it at the same moments, so
//! the programs behave the same wherever a dense model is held (issue #2585).

use std::path::PathBuf;

use ipc_protocol::{NeuralEnginePrefillStatus, WorkerChunkingConfiguration};

use super::neural_engine_host::NeuralEngineHost;
use super::neural_engine_prefill_loading::{
    NeuralEnginePrefillPreparation, NeuralEnginePreparationStep,
    begin_neural_engine_prefill_preparation,
};
use super::neural_engine_program_settlement::settle_neural_engine_programs;

pub struct NeuralEnginePrefillLifecycle {
    /// Where this model revision's compiled programs are kept between loads;
    /// `None` compiles them again on every load.
    astro_neural_engine_program_cache_directory: Option<PathBuf>,
    /// What the opt-in prompt reading is doing for the loaded model.
    status: NeuralEnginePrefillStatus,
    /// The programs still on their way to the loaded model, which serves on
    /// the GPU until they arrive.
    preparation: Option<NeuralEnginePrefillPreparation>,
}

impl NeuralEnginePrefillLifecycle {
    #[must_use]
    pub const fn new(astro_neural_engine_program_cache_directory: Option<PathBuf>) -> Self {
        Self {
            astro_neural_engine_program_cache_directory,
            status: NeuralEnginePrefillStatus::Off,
            preparation: None,
        }
    }

    #[must_use]
    pub const fn status(&self) -> &NeuralEnginePrefillStatus {
        &self.status
    }

    /// Starts the programs for a model whose weights are in place. They come
    /// last in a load so their admission sees the model's own memory, and
    /// are only started here: the model serves on the GPU while they build.
    pub fn begin(&mut self, host: &dyn NeuralEngineHost, chunking: &WorkerChunkingConfiguration) {
        (self.status, self.preparation) = begin_neural_engine_prefill_preparation(
            host,
            chunking,
            self.astro_neural_engine_program_cache_directory.as_deref(),
            None,
        );
    }

    /// A request's start and every forward say so on the owner thread, which
    /// together cover the GPU work a waiting measurement must not read
    /// through.
    pub fn note_gpu_used(&mut self) {
        if let Some(preparation) = self.preparation.as_mut() {
            preparation.note_gpu_used();
        }
    }

    /// Moves the programs one step along, or settles what the last request
    /// learned about programs already running, and returns a status a reader
    /// has not yet seen. Only called while no request is active, which is
    /// when programs can be released.
    pub fn advance(
        &mut self,
        host: &mut dyn NeuralEngineHost,
    ) -> Option<NeuralEnginePrefillStatus> {
        let Some(preparation) = self.preparation.as_mut() else {
            let settled_status =
                settle_neural_engine_programs(host.model_base_mut(), &self.status)?;
            self.status = settled_status.clone();
            return Some(settled_status);
        };
        let reported_status = match preparation.advance(host) {
            NeuralEnginePreparationStep::Unchanged => return None,
            NeuralEnginePreparationStep::Progressed(status) => status,
            NeuralEnginePreparationStep::ReadyToFinish => self.preparation.take()?.finish(),
        };
        self.status = reported_status.clone();
        Some(reported_status)
    }
}

/// A text engine's bridge from the engine-neutral execution contract onto
/// its state's methods: the methods both Qwen 3.5 engines share, and the
/// Neural Engine programs a dense model held in memory builds beside itself
/// and advances while the engine is idle. Each engine's state has an
/// `advance_neural_engine_preparation` of its own, which this forwards to.
macro_rules! qwen3_5_inference_execution_bridge {
    ($inference_execution:ty, $engine_state:ty) => {
        impl $crate::MlxInferenceExecution for $inference_execution {
            type Request = $crate::TextInferenceRequest;

            const PREPARES_NEURAL_ENGINE_PROGRAMS: bool = true;

            text_generation::text_shared_inference_execution_methods!($engine_state);

            fn advance_neural_engine_preparation(
                &mut self,
            ) -> Option<ipc_protocol::NeuralEnginePrefillStatus> {
                <$engine_state>::advance_neural_engine_preparation(self)
            }
        }
    };
}
pub(crate) use qwen3_5_inference_execution_bridge;
