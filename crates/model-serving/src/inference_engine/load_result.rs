use astronomical_ipc_protocol::ExpertMemoryMode;

/// Immutable readiness metadata captured after all engine load transitions.
///
/// The worker publishes this snapshot with `Ready` or `ModelSwapped`; callers do
/// not infer runtime mode from artifact names, configuration, or memory totals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineLoadResult {
    expert_memory_mode: Option<ExpertMemoryMode>,
    minimum_mlx_memory_ceiling_bytes: u64,
}

impl EngineLoadResult {
    #[must_use]
    pub fn new() -> Self {
        Self {
            expert_memory_mode: None,
            minimum_mlx_memory_ceiling_bytes: 1,
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
