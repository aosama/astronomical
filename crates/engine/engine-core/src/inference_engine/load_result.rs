use ipc_protocol::{DraftModelStatus, ExpertMemoryMode, NeuralEnginePrefillStatus};

/// Immutable readiness metadata captured after all engine load transitions.
///
/// The worker publishes this snapshot with `Ready` or `ModelSwapped`; callers do
/// not infer runtime mode from artifact names, configuration, or memory totals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineLoadResult {
    expert_memory_mode: Option<ExpertMemoryMode>,
    minimum_mlx_memory_ceiling_bytes: u64,
    neural_engine_prefill_status: NeuralEnginePrefillStatus,
    /// `None` for an engine that never looks for a drafter, so it announces nothing.
    draft_model_status: Option<DraftModelStatus>,
}

impl EngineLoadResult {
    #[must_use]
    pub fn new() -> Self {
        Self {
            expert_memory_mode: None,
            minimum_mlx_memory_ceiling_bytes: 1,
            neural_engine_prefill_status: NeuralEnginePrefillStatus::Off,
            draft_model_status: None,
        }
    }

    /// Sets the loaded model's expert-memory mode.
    #[must_use]
    pub const fn with_expert_memory_mode(
        mut self,
        expert_memory_mode: Option<ExpertMemoryMode>,
    ) -> Self {
        self.expert_memory_mode = expert_memory_mode;
        self
    }

    /// Sets the loaded model's safe idle MLX minimum in exact bytes.
    #[must_use]
    pub const fn with_minimum_mlx_memory_ceiling_bytes(
        mut self,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> Self {
        self.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
        self
    }

    /// Sets what the loaded model's Neural Engine prefill ended up doing.
    #[must_use]
    pub fn with_neural_engine_prefill_status(
        mut self,
        neural_engine_prefill_status: NeuralEnginePrefillStatus,
    ) -> Self {
        self.neural_engine_prefill_status = neural_engine_prefill_status;
        self
    }

    /// For an engine that has no Neural Engine path: a user who opted in still
    /// gets told why it is not running, rather than seeing it reported as off.
    #[must_use]
    pub fn neural_engine_prefill_unsupported_when_enabled(
        is_enabled: bool,
    ) -> NeuralEnginePrefillStatus {
        if is_enabled {
            NeuralEnginePrefillStatus::Unavailable {
                reason: "this model is running on an engine without Neural Engine programs, either its architecture's or the low-memory one".to_owned(),
            }
        } else {
            NeuralEnginePrefillStatus::Off
        }
    }

    /// Whether this load left programs being prepared beside the serving model.
    #[must_use]
    pub const fn is_neural_engine_preparation_running(&self) -> bool {
        matches!(
            self.neural_engine_prefill_status,
            NeuralEnginePrefillStatus::Preparing { .. }
        )
    }

    /// The event announcing a setting that was on, or `None` when it was off,
    /// so workers that never opt in publish exactly what they always did.
    #[must_use]
    pub fn neural_engine_prefill_event(&self) -> Option<ipc_protocol::WorkerEvent> {
        if self.neural_engine_prefill_status == NeuralEnginePrefillStatus::Off {
            return None;
        }
        Some(ipc_protocol::WorkerEvent::NeuralEnginePrefillChanged {
            neural_engine_prefill: self.neural_engine_prefill_status.clone(),
        })
    }

    /// Sets what the engine found when it looked for a drafter.
    #[must_use]
    pub fn with_draft_model_status(mut self, draft_model_status: DraftModelStatus) -> Self {
        self.draft_model_status = Some(draft_model_status);
        self
    }

    /// The event announcing the drafter state, or `None` for an engine that
    /// never looks for a drafter, so such engines publish exactly what they always did.
    #[must_use]
    pub fn draft_model_event(&self) -> Option<ipc_protocol::WorkerEvent> {
        self.draft_model_status
            .clone()
            .map(|draft_model| ipc_protocol::WorkerEvent::DraftModelChanged { draft_model })
    }

    /// Returns the expert-memory mode selected before readiness.
    #[must_use]
    pub const fn expert_memory_mode(&self) -> Option<ExpertMemoryMode> {
        self.expert_memory_mode
    }

    /// Returns the exact safe idle MLX minimum for the loaded engine.
    #[must_use]
    pub const fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.minimum_mlx_memory_ceiling_bytes
    }
}

impl Default for EngineLoadResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "load_result_tests.rs"]
mod tests;
