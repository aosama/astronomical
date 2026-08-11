use astronomical_ipc_protocol::{
    ExpertMemoryMode, MtpRuntimeState, SpeculativePrefillRuntimeState,
};

/// Immutable readiness metadata captured after all engine load transitions.
///
/// The worker publishes this snapshot with `Ready` or `ModelSwapped`; callers do
/// not infer runtime mode from artifact names, configuration, or memory totals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineLoadResult {
    expert_memory_mode: Option<ExpertMemoryMode>,
    mtp_runtime_state: MtpRuntimeState,
    mtp_unavailable_reason: Option<String>,
    speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
    speculative_prefill_unavailable_reason: Option<String>,
    speculative_prefill_draft_model_id: Option<String>,
    speculative_prefill_draft_model_revision: Option<String>,
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
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
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

    /// Sets optional draft-assisted speculative-prefill load metadata.
    #[must_use]
    pub fn with_speculative_prefill_runtime(
        mut self,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        speculative_prefill_unavailable_reason: Option<String>,
        speculative_prefill_draft_model_id: Option<String>,
        speculative_prefill_draft_model_revision: Option<String>,
    ) -> Self {
        self.speculative_prefill_runtime_state = speculative_prefill_runtime_state;
        self.speculative_prefill_unavailable_reason = speculative_prefill_unavailable_reason;
        self.speculative_prefill_draft_model_id = speculative_prefill_draft_model_id;
        self.speculative_prefill_draft_model_revision = speculative_prefill_draft_model_revision;
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

    /// Returns the optional draft-assisted speculative-prefill runtime state.
    #[must_use]
    pub const fn speculative_prefill_runtime_state(&self) -> SpeculativePrefillRuntimeState {
        self.speculative_prefill_runtime_state
    }

    /// Returns the optional draft-assisted speculative-prefill load failure reason.
    #[must_use]
    pub fn speculative_prefill_unavailable_reason(&self) -> Option<&str> {
        self.speculative_prefill_unavailable_reason.as_deref()
    }

    /// Returns the configured draft model identity, when present.
    #[must_use]
    pub fn speculative_prefill_draft_model_id(&self) -> Option<&str> {
        self.speculative_prefill_draft_model_id.as_deref()
    }

    /// Returns the validated request-scoped draft model revision, when active.
    #[must_use]
    pub fn speculative_prefill_draft_model_revision(&self) -> Option<&str> {
        self.speculative_prefill_draft_model_revision.as_deref()
    }

    /// Returns the exact safe idle MLX minimum for the loaded engine.
    #[must_use]
    pub const fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.minimum_mlx_memory_ceiling_bytes
    }
}
