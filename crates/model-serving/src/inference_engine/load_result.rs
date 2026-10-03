use astronomical_ipc_protocol::{ExpertMemoryMode, MtpDepthStatus, MtpRuntimeState};

/// Immutable readiness metadata captured after all engine load transitions.
///
/// The worker publishes this snapshot with `Ready` or `ModelSwapped`; callers do
/// not infer runtime mode from artifact names, configuration, or memory totals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineLoadResult {
    expert_memory_mode: Option<ExpertMemoryMode>,
    mtp_runtime_state: MtpRuntimeState,
    mtp_unavailable_reason: Option<String>,
    mtp_depth_status: MtpDepthStatus,
    minimum_mlx_memory_ceiling_bytes: u64,
}

impl EngineLoadResult {
    /// Creates a load result with the default MTP state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expert_memory_mode: None,
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            mtp_depth_status: MtpDepthStatus::default(),
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

    /// Sets the MTP runtime state.
    #[must_use]
    pub fn with_mtp_runtime_state(mut self, mtp_runtime_state: MtpRuntimeState) -> Self {
        self.mtp_runtime_state = mtp_runtime_state;
        self
    }

    /// Sets the MTP unavailable reason when the runtime state is Unavailable.
    #[must_use]
    pub fn with_mtp_unavailable_reason(mut self, reason: String) -> Self {
        self.mtp_unavailable_reason = Some(reason);
        self
    }

    #[must_use]
    pub const fn with_mtp_depth_status(mut self, mtp_depth_status: MtpDepthStatus) -> Self {
        self.mtp_depth_status = mtp_depth_status;
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

    /// Returns the MTP runtime state.
    #[must_use]
    pub const fn mtp_runtime_state(&self) -> MtpRuntimeState {
        self.mtp_runtime_state
    }

    /// Returns the expert-memory mode selected before readiness.
    #[must_use]
    pub const fn expert_memory_mode(&self) -> Option<ExpertMemoryMode> {
        self.expert_memory_mode
    }

    /// Returns the MTP unavailable reason, if any.
    #[must_use]
    pub fn mtp_unavailable_reason(&self) -> Option<&str> {
        self.mtp_unavailable_reason.as_deref()
    }

    #[must_use]
    pub const fn mtp_depth_status(&self) -> MtpDepthStatus {
        self.mtp_depth_status
    }

    /// Returns the exact safe idle MLX minimum for the loaded engine.
    #[must_use]
    pub const fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.minimum_mlx_memory_ceiling_bytes
    }
}
