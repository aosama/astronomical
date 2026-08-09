use astronomical_ipc_protocol::ExpertMemoryMode;

/// Reconciled ownership view of one MLX active-memory measurement.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MlxActiveMemoryBreakdown {
    /// Sparse-expert payload bytes represented within active MLX memory.
    pub expert_payload_bytes: u64,
    /// Resident non-expert model payload represented within active MLX memory.
    pub model_core_payload_bytes: u64,
    /// Request context payload represented within active MLX memory.
    pub context_state_payload_bytes: u64,
    /// Complete active MLX memory attributed to request-scoped draft scoring.
    pub speculative_prefill_draft_memory_bytes: u64,
}

impl MlxActiveMemoryBreakdown {
    #[must_use]
    /// Reconciles logical owners in priority order so their sum equals active memory.
    pub fn reconcile(
        mlx_active_memory_bytes: u64,
        expert_payload_bytes: u64,
        model_core_payload_bytes: u64,
        context_state_payload_bytes: u64,
    ) -> Self {
        let reconciled_expert_payload_bytes = expert_payload_bytes.min(mlx_active_memory_bytes);
        let remaining_after_experts =
            mlx_active_memory_bytes.saturating_sub(reconciled_expert_payload_bytes);
        let reconciled_model_core_payload_bytes =
            model_core_payload_bytes.min(remaining_after_experts);
        let remaining_after_model_core =
            remaining_after_experts.saturating_sub(reconciled_model_core_payload_bytes);
        let reconciled_context_state_payload_bytes =
            context_state_payload_bytes.min(remaining_after_model_core);
        Self {
            expert_payload_bytes: reconciled_expert_payload_bytes,
            model_core_payload_bytes: reconciled_model_core_payload_bytes,
            context_state_payload_bytes: reconciled_context_state_payload_bytes,
            speculative_prefill_draft_memory_bytes: 0,
        }
    }

    #[must_use]
    /// Reconciles target owners and assigns every remaining active byte to draft scoring.
    pub fn reconcile_with_speculative_prefill_draft(
        mlx_active_memory_bytes: u64,
        expert_payload_bytes: u64,
        model_core_payload_bytes: u64,
        context_state_payload_bytes: u64,
        speculative_prefill_draft_memory_bytes: u64,
    ) -> Self {
        let target_memory_breakdown = Self::reconcile(
            mlx_active_memory_bytes,
            expert_payload_bytes,
            model_core_payload_bytes,
            context_state_payload_bytes,
        );
        let target_attributed_memory_bytes = target_memory_breakdown
            .expert_payload_bytes
            .saturating_add(target_memory_breakdown.model_core_payload_bytes)
            .saturating_add(target_memory_breakdown.context_state_payload_bytes);
        let remaining_active_memory_bytes =
            mlx_active_memory_bytes.saturating_sub(target_attributed_memory_bytes);
        let reconciled_known_speculative_prefill_draft_memory_bytes =
            speculative_prefill_draft_memory_bytes.min(remaining_active_memory_bytes);
        let unclassified_speculative_prefill_draft_work_bytes = remaining_active_memory_bytes
            .saturating_sub(reconciled_known_speculative_prefill_draft_memory_bytes);
        Self {
            speculative_prefill_draft_memory_bytes:
                reconciled_known_speculative_prefill_draft_memory_bytes
                    .saturating_add(unclassified_speculative_prefill_draft_work_bytes),
            ..target_memory_breakdown
        }
    }
}

/// One MLX memory measurement with its reconciled active-memory ownership view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxMemoryTelemetry {
    pub active_memory_bytes: u64,
    pub allocator_cache_memory_bytes: u64,
    pub peak_memory_bytes: u64,
    pub active_memory_breakdown: MlxActiveMemoryBreakdown,
}

impl MlxMemoryTelemetry {
    #[must_use]
    pub const fn new(
        active_memory_bytes: u64,
        allocator_cache_memory_bytes: u64,
        peak_memory_bytes: u64,
        active_memory_breakdown: MlxActiveMemoryBreakdown,
    ) -> Self {
        Self {
            active_memory_bytes,
            allocator_cache_memory_bytes,
            peak_memory_bytes,
            active_memory_breakdown,
        }
    }
}

/// Final state reported after a live MLX memory-ceiling change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxMemoryLimitAdjustment {
    effective_mlx_memory_ceiling_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
    expert_memory_mode: ExpertMemoryMode,
    mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
}

impl MlxMemoryLimitAdjustment {
    /// Creates one successfully applied live-limit report.
    #[must_use]
    pub const fn new(
        effective_mlx_memory_ceiling_bytes: u64,
        minimum_mlx_memory_ceiling_bytes: u64,
        expert_memory_mode: ExpertMemoryMode,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
    ) -> Self {
        Self {
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            expert_memory_mode,
            mlx_memory_telemetry,
        }
    }

    #[must_use]
    pub const fn effective_mlx_memory_ceiling_bytes(self) -> u64 {
        self.effective_mlx_memory_ceiling_bytes
    }

    #[must_use]
    pub const fn minimum_mlx_memory_ceiling_bytes(self) -> u64 {
        self.minimum_mlx_memory_ceiling_bytes
    }

    #[must_use]
    pub const fn expert_memory_mode(self) -> ExpertMemoryMode {
        self.expert_memory_mode
    }

    #[must_use]
    pub const fn mlx_memory_telemetry(self) -> Option<MlxMemoryTelemetry> {
        self.mlx_memory_telemetry
    }
}
