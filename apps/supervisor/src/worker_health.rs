use super::serving_session_snapshot::ServingSessionSnapshot;
use astronomical_ipc_protocol::{
    ChatModelCapabilities, ExpertMemoryMode, MtpDepthStatus, MtpRuntimeState,
    SpeculativePrefillRuntimeState, WorkerEvent, WorkerMlxMemorySnapshot,
    WorkerPromptProcessingPhase, WorkerRuntimeFeatureConfiguration,
};
use tokio::time::Instant;

mod publisher;

pub(crate) use publisher::{
    clear_active_request_progress, clear_latest_mlx_memory_snapshot,
    publish_active_request_progress, publish_activity, publish_expert_memory_mode,
    publish_expert_residency, publish_health, publish_latest_mlx_memory_snapshot,
    publish_mlx_memory_limit_changed, publish_mlx_memory_limit_rejection,
    publish_pending_mlx_memory_ceiling, publish_pending_prompt_cache_clear,
    publish_persistent_prompt_cache_stats, publish_worker_expert_residency,
    record_prompt_work_reuse, record_serving_session,
};

/// Coarse worker availability state exposed by the supervisor readiness endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkerHealthStatus {
    /// The worker process is responsive but its inference engine is still loading.
    Loading,
    /// The worker is available for new requests.
    Ready,
    /// The worker is absent or otherwise unavailable.
    Unavailable,
}

impl WorkerHealthStatus {
    /// Stable text used by readiness responses and diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Unavailable => "unavailable",
        }
    }

    /// Whether the worker should currently receive new requests.
    #[must_use]
    pub const fn is_ready(self) -> bool {
        matches!(self, Self::Ready)
    }
}

/// Current activity of the one local worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkerActivity {
    /// No generation request is active.
    Idle,
    /// A request is active but no output has reached the supervisor yet.
    PromptProcessing,
    /// Prompt processing finished and expert ownership is being prepared for decode.
    GenerationPreparation,
    /// At least one generated output has reached the supervisor.
    Generating,
}

impl WorkerActivity {
    /// Stable text used by the status endpoint and menu bar display.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::PromptProcessing => "prompt_processing",
            Self::GenerationPreparation => "generation_preparation",
            Self::Generating => "generating",
        }
    }
}

/// Per-request progress snapshot copied directly from one worker progress event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ActiveRequestProgress {
    /// Accumulated prompt-processing progress for the active request.
    Prefill {
        prompt_processing_phase: WorkerPromptProcessingPhase,
        processed_tokens: u32,
        total_tokens: u32,
        /// Supervisor wall-clock origin used only for live status presentation.
        request_started_at: Instant,
        elapsed_millis: u64,
        /// Present only after the engine completes and measures a prefill chunk.
        completed_prefill_chunk_tokens: Option<u32>,
    },
    /// Prefill completed and the worker is preparing the first decode forward.
    GenerationPreparation {
        request_started_at: Instant,
        preparation_started_at: Instant,
        total_layer_count: u32,
        complete_layer_count: u32,
        partial_layer_count: u32,
    },
    /// Accumulated generation progress for the active request.
    Generation {
        generated_token_count: u32,
        maximum_output_tokens: u32,
        elapsed_millis: u64,
    },
}

/// Latest worker-reported complete-versus-partial sparse-expert topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertResidencySnapshot {
    pub total_layer_count: u32,
    pub complete_layer_count: u32,
    pub complete_layer_payload_bytes: u64,
    pub partial_layer_count: u32,
    pub partial_layer_payload_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PersistentPromptCacheSummary {
    pub hits: u64,
    pub misses: u64,
    pub tokens_saved: u64,
    pub block_token_count: u64,
    pub sequence_state_block_count: u64,
    pub boundary_state_snapshot_count: u64,
    pub visual_embedding_count: u64,
    pub total_size_bytes: u64,
    pub visual_embedding_total_size_bytes: u64,
    pub maximum_size_bytes: u64,
    pub visual_embedding_hits: u64,
    pub visual_embedding_misses: u64,
    pub visual_embedding_rows_loaded: u64,
}

/// Scope of the newest cache clear waiting for generation to become idle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingPromptCacheClear {
    pub model_id: Option<String>,
}

impl PersistentPromptCacheSummary {
    pub fn from_worker_event(persistent_prompt_cache_stats: Option<&WorkerEvent>) -> Self {
        let Some(WorkerEvent::PersistentPromptCacheStats {
            persistent_prompt_cache_hits,
            persistent_prompt_cache_misses,
            persistent_prompt_cache_tokens_saved,
            persistent_prompt_cache_block_token_count,
            persistent_prompt_cache_sequence_state_block_count,
            persistent_prompt_cache_boundary_state_snapshot_count,
            persistent_prompt_cache_visual_embedding_count,
            persistent_prompt_cache_total_size_bytes,
            persistent_prompt_cache_visual_embedding_total_size_bytes,
            persistent_prompt_cache_maximum_size_bytes,
            persistent_prompt_cache_visual_embedding_hits,
            persistent_prompt_cache_visual_embedding_misses,
            persistent_prompt_cache_visual_embedding_rows_loaded,
        }) = persistent_prompt_cache_stats
        else {
            return Self::default();
        };
        Self {
            hits: *persistent_prompt_cache_hits,
            misses: *persistent_prompt_cache_misses,
            tokens_saved: *persistent_prompt_cache_tokens_saved,
            block_token_count: *persistent_prompt_cache_block_token_count,
            sequence_state_block_count: *persistent_prompt_cache_sequence_state_block_count,
            boundary_state_snapshot_count: *persistent_prompt_cache_boundary_state_snapshot_count,
            visual_embedding_count: *persistent_prompt_cache_visual_embedding_count,
            total_size_bytes: *persistent_prompt_cache_total_size_bytes,
            visual_embedding_total_size_bytes:
                *persistent_prompt_cache_visual_embedding_total_size_bytes,
            maximum_size_bytes: *persistent_prompt_cache_maximum_size_bytes,
            visual_embedding_hits: *persistent_prompt_cache_visual_embedding_hits,
            visual_embedding_misses: *persistent_prompt_cache_visual_embedding_misses,
            visual_embedding_rows_loaded: *persistent_prompt_cache_visual_embedding_rows_loaded,
        }
    }

    pub fn hit_rate(self) -> f64 {
        let query_count = self.hits.saturating_add(self.misses);
        if query_count == 0 {
            return 0.0;
        }
        let hit_rate = self.hits as f64 / query_count as f64;
        (hit_rate * 10_000.0).round() / 10_000.0
    }
}

/// Snapshot of the supervisor's current worker health assessment.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerHealthSnapshot {
    /// Coarse worker availability state.
    pub status: WorkerHealthStatus,
    /// Fine-grained activity while the worker remains ready.
    pub activity: WorkerActivity,
    /// Exact worker-reported model identity when the worker is ready.
    pub ready_model_id: Option<String>,
    /// Worker-reported structured-chat capabilities when the worker is ready.
    pub ready_model_capabilities: Option<ChatModelCapabilities>,
    /// Optional per-request progress for the currently active generation.
    pub active_request_progress: Option<ActiveRequestProgress>,
    /// Latest worker-reported sparse-expert residency.
    pub expert_memory_mode: Option<ExpertMemoryMode>,
    /// Concrete topology retained independently of transient request progress.
    pub expert_residency: Option<ExpertResidencySnapshot>,
    /// Actual MTP runtime state: Disabled, TargetOnly, Active, or Unavailable.
    pub mtp_runtime_state: MtpRuntimeState,
    /// Concise reason when MTP runtime state is Unavailable.
    pub mtp_unavailable_reason: Option<String>,
    /// Configured, artifact, resolved, and current execution depth metadata.
    pub mtp_depth_status: MtpDepthStatus,
    /// Actual optional draft-assisted speculative-prefill runtime state.
    pub speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
    /// Concise reason when speculative prefill is Unavailable.
    pub speculative_prefill_unavailable_reason: Option<String>,
    /// Configured draft model identity reported by the worker.
    pub speculative_prefill_draft_model_id: Option<String>,
    /// Validated request-scoped draft revision reported by the worker.
    pub speculative_prefill_draft_model_revision: Option<String>,
    /// Feature settings explicitly acknowledged by the currently running worker.
    pub worker_runtime_feature_configuration: Option<WorkerRuntimeFeatureConfiguration>,
    /// Latest persistent prompt-cache observability stats from the worker.
    pub persistent_prompt_cache_stats: Option<WorkerEvent>,
    /// Latest worker-owned MLX allocator observation, regardless of request lifecycle phase.
    pub latest_mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    /// Immutable maximum reported by the worker's macOS/MLX startup probe.
    pub machine_mlx_memory_ceiling_bytes: u64,
    /// Safe idle lower bound reported by the loaded model or idle worker.
    pub minimum_mlx_memory_ceiling_bytes: u64,
    /// Latest accepted limit queued until the active generation finalizes.
    pub pending_mlx_memory_ceiling_bytes: Option<u64>,
    /// Newest cache-clear request queued behind the active generation.
    pub pending_prompt_cache_clear: Option<PendingPromptCacheClear>,
    /// Bounded last live-memory control failure. Cleared by a later success.
    pub mlx_memory_limit_error: Option<String>,
    pub mlx_memory_ceiling_bytes: u64,
    pub serving_session: ServingSessionSnapshot,
}

impl WorkerHealthSnapshot {
    /// Builds a ready snapshot from the worker's enriched readiness event.
    #[must_use]
    pub fn ready_with_model(
        model_id: String,
        capabilities: ChatModelCapabilities,
        mtp_runtime_state: MtpRuntimeState,
        mtp_unavailable_reason: Option<String>,
    ) -> Self {
        Self {
            status: WorkerHealthStatus::Ready,
            activity: WorkerActivity::Idle,
            ready_model_id: Some(model_id),
            ready_model_capabilities: Some(capabilities),
            active_request_progress: None,
            expert_memory_mode: None,
            expert_residency: None,
            mtp_runtime_state,
            mtp_unavailable_reason,
            mtp_depth_status: MtpDepthStatus::EMPTY,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
            worker_runtime_feature_configuration: None,
            persistent_prompt_cache_stats: None,
            latest_mlx_memory_snapshot: None,
            machine_mlx_memory_ceiling_bytes: 0,
            minimum_mlx_memory_ceiling_bytes: 1,
            pending_mlx_memory_ceiling_bytes: None,
            pending_prompt_cache_clear: None,
            mlx_memory_limit_error: None,
            mlx_memory_ceiling_bytes: 0,
            serving_session: ServingSessionSnapshot::empty(),
        }
    }

    /// Builds a fresh resident-model snapshot without resetting daemon-session totals.
    #[must_use]
    pub fn ready_with_replacement_model(
        model_id: String,
        capabilities: ChatModelCapabilities,
        minimum_mlx_memory_ceiling_bytes: u64,
        mtp_runtime_state: MtpRuntimeState,
        mtp_unavailable_reason: Option<String>,
        previous_health_snapshot: &Self,
    ) -> Self {
        let mut replacement_health_snapshot = Self::ready_with_model(
            model_id,
            capabilities,
            mtp_runtime_state,
            mtp_unavailable_reason,
        );
        replacement_health_snapshot.mlx_memory_ceiling_bytes =
            previous_health_snapshot.mlx_memory_ceiling_bytes;
        replacement_health_snapshot.machine_mlx_memory_ceiling_bytes =
            previous_health_snapshot.machine_mlx_memory_ceiling_bytes;
        replacement_health_snapshot.minimum_mlx_memory_ceiling_bytes =
            minimum_mlx_memory_ceiling_bytes;
        replacement_health_snapshot.pending_prompt_cache_clear =
            previous_health_snapshot.pending_prompt_cache_clear.clone();
        replacement_health_snapshot.serving_session =
            previous_health_snapshot.serving_session.clone();
        replacement_health_snapshot
    }

    /// Builds a ready snapshot for an idle worker that has no resident model.
    #[must_use]
    pub const fn ready_without_model(mlx_memory_ceiling_bytes: u64) -> Self {
        Self::ready_without_model_with_memory_limits(
            mlx_memory_ceiling_bytes,
            mlx_memory_ceiling_bytes,
            1,
        )
    }

    #[must_use]
    pub const fn ready_without_model_with_memory_limits(
        machine_mlx_memory_ceiling_bytes: u64,
        mlx_memory_ceiling_bytes: u64,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> Self {
        Self {
            status: WorkerHealthStatus::Ready,
            activity: WorkerActivity::Idle,
            ready_model_id: None,
            ready_model_capabilities: None,
            active_request_progress: None,
            expert_memory_mode: None,
            expert_residency: None,
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            mtp_depth_status: MtpDepthStatus::EMPTY,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
            worker_runtime_feature_configuration: None,
            persistent_prompt_cache_stats: None,
            latest_mlx_memory_snapshot: None,
            machine_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            pending_mlx_memory_ceiling_bytes: None,
            pending_prompt_cache_clear: None,
            mlx_memory_limit_error: None,
            mlx_memory_ceiling_bytes,
            serving_session: ServingSessionSnapshot::empty(),
        }
    }

    /// Builds a non-ready snapshot.
    #[must_use]
    pub const fn unavailable(status: WorkerHealthStatus) -> Self {
        Self {
            status,
            activity: WorkerActivity::Idle,
            ready_model_id: None,
            ready_model_capabilities: None,
            active_request_progress: None,
            expert_memory_mode: None,
            expert_residency: None,
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            mtp_depth_status: MtpDepthStatus::EMPTY,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
            worker_runtime_feature_configuration: None,
            persistent_prompt_cache_stats: None,
            latest_mlx_memory_snapshot: None,
            machine_mlx_memory_ceiling_bytes: 0,
            minimum_mlx_memory_ceiling_bytes: 1,
            pending_mlx_memory_ceiling_bytes: None,
            pending_prompt_cache_clear: None,
            mlx_memory_limit_error: None,
            mlx_memory_ceiling_bytes: 0,
            serving_session: ServingSessionSnapshot::empty(),
        }
    }

    #[must_use]
    pub const fn mtp_runtime_state(&self) -> MtpRuntimeState {
        self.mtp_runtime_state
    }

    #[must_use]
    pub fn mtp_unavailable_reason(&self) -> Option<&str> {
        self.mtp_unavailable_reason.as_deref()
    }

    #[must_use]
    pub const fn with_mtp_depth_status(mut self, mtp_depth_status: MtpDepthStatus) -> Self {
        self.mtp_depth_status = mtp_depth_status;
        self
    }

    /// Adds worker-reported optional draft-assisted speculative-prefill metadata.
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

    /// Records the startup feature policy acknowledged by this exact worker.
    #[must_use]
    pub fn with_worker_runtime_feature_configuration(
        mut self,
        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration,
    ) -> Self {
        self.worker_runtime_feature_configuration = Some(worker_runtime_feature_configuration);
        self
    }
}
