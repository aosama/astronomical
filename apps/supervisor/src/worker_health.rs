use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{
    ChatModelCapabilities, ExpertMemoryMode, MtpRuntimeState, SpeculativePrefillRuntimeState,
    WorkerEvent, WorkerMlxMemorySnapshot, WorkerPrefillOptimizerInsight,
    WorkerPromptProcessingPhase, WorkerPromptWorkReuse,
};
use tokio::time::Instant;

use super::serving_session_snapshot::ServingSessionSnapshot;

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
        completed_prefill_chunck_tokens: Option<u32>,
    },
    /// Accumulated generation progress for the active request.
    Generation {
        generated_token_count: u32,
        maximum_output_tokens: u32,
        elapsed_millis: u64,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PersistentPromptCacheSummary {
    pub hits: u64,
    pub misses: u64,
    pub tokens_saved: u64,
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

impl PersistentPromptCacheSummary {
    pub fn from_worker_event(persistent_prompt_cache_stats: Option<&WorkerEvent>) -> Self {
        let Some(WorkerEvent::PersistentPromptCacheStats {
            persistent_prompt_cache_hits,
            persistent_prompt_cache_misses,
            persistent_prompt_cache_tokens_saved,
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
    /// Actual MTP runtime state: Disabled, TargetOnly, Active, or Unavailable.
    pub mtp_runtime_state: MtpRuntimeState,
    /// Concise reason when MTP runtime state is Unavailable.
    pub mtp_unavailable_reason: Option<String>,
    /// Actual optional draft-assisted speculative-prefill runtime state.
    pub speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
    /// Concise reason when speculative prefill is Unavailable.
    pub speculative_prefill_unavailable_reason: Option<String>,
    /// Configured draft model identity reported by the worker.
    pub speculative_prefill_draft_model_id: Option<String>,
    /// Validated request-scoped draft revision reported by the worker.
    pub speculative_prefill_draft_model_revision: Option<String>,
    /// Latest persistent prompt-cache observability stats from the worker.
    pub persistent_prompt_cache_stats: Option<WorkerEvent>,
    /// Latest worker-owned MLX allocator observation, regardless of request lifecycle phase.
    pub latest_mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    /// Bounded recent adaptive-prefill decisions for the Observatory.
    pub prefill_optimizer_insights: Vec<WorkerPrefillOptimizerInsight>,
    /// Immutable maximum reported by the worker's macOS/MLX startup probe.
    pub machine_mlx_memory_ceiling_bytes: u64,
    /// Safe idle lower bound reported by the loaded model or idle worker.
    pub minimum_mlx_memory_ceiling_bytes: u64,
    /// Latest accepted limit queued until the active generation finalizes.
    pub pending_mlx_memory_ceiling_bytes: Option<u64>,
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
            mtp_runtime_state,
            mtp_unavailable_reason,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
            persistent_prompt_cache_stats: None,
            latest_mlx_memory_snapshot: None,
            prefill_optimizer_insights: Vec::new(),
            machine_mlx_memory_ceiling_bytes: 0,
            minimum_mlx_memory_ceiling_bytes: 1,
            pending_mlx_memory_ceiling_bytes: None,
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
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
            persistent_prompt_cache_stats: None,
            latest_mlx_memory_snapshot: None,
            prefill_optimizer_insights: Vec::new(),
            machine_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            pending_mlx_memory_ceiling_bytes: None,
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
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
            persistent_prompt_cache_stats: None,
            latest_mlx_memory_snapshot: None,
            prefill_optimizer_insights: Vec::new(),
            machine_mlx_memory_ceiling_bytes: 0,
            minimum_mlx_memory_ceiling_bytes: 1,
            pending_mlx_memory_ceiling_bytes: None,
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
}

pub(crate) fn publish_health(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    worker_health_snapshot: WorkerHealthSnapshot,
) {
    if let Ok(mut health_snapshot) = health_snapshot.write() {
        *health_snapshot = worker_health_snapshot;
    }
}

pub(crate) fn publish_activity(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    worker_activity: WorkerActivity,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.activity = worker_activity;
    }
}

pub(crate) fn publish_active_request_progress(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    progress: ActiveRequestProgress,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.active_request_progress = Some(progress);
    }
}

pub(crate) fn clear_active_request_progress(health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.active_request_progress = None;
    }
}

pub(crate) fn publish_expert_memory_mode(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    expert_memory_mode: ExpertMemoryMode,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.expert_memory_mode = Some(expert_memory_mode);
    }
}

pub(crate) fn publish_persistent_prompt_cache_stats(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    persistent_prompt_cache_stats: WorkerEvent,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.persistent_prompt_cache_stats = Some(persistent_prompt_cache_stats);
    }
}

pub(crate) fn publish_latest_mlx_memory_snapshot(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    mlx_memory_snapshot: WorkerMlxMemorySnapshot,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.latest_mlx_memory_snapshot = Some(mlx_memory_snapshot);
    }
}

pub(crate) fn clear_latest_mlx_memory_snapshot(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.latest_mlx_memory_snapshot = None;
    }
}

pub(crate) fn publish_pending_mlx_memory_ceiling(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    pending_mlx_memory_ceiling_bytes: Option<u64>,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.pending_mlx_memory_ceiling_bytes = pending_mlx_memory_ceiling_bytes;
    }
}

pub(crate) fn publish_mlx_memory_limit_changed(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    effective_mlx_memory_ceiling_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
    expert_memory_mode: ExpertMemoryMode,
    mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.mlx_memory_ceiling_bytes = effective_mlx_memory_ceiling_bytes;
        worker_health_snapshot.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
        worker_health_snapshot.pending_mlx_memory_ceiling_bytes = None;
        worker_health_snapshot.mlx_memory_limit_error = None;
        worker_health_snapshot.expert_memory_mode = worker_health_snapshot
            .ready_model_id
            .as_ref()
            .map(|_| expert_memory_mode);
        worker_health_snapshot.latest_mlx_memory_snapshot = mlx_memory_snapshot;
    }
}

pub(crate) fn publish_mlx_memory_limit_rejection(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    minimum_mlx_memory_ceiling_bytes: u64,
    reason: String,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
        worker_health_snapshot.pending_mlx_memory_ceiling_bytes = None;
        worker_health_snapshot.mlx_memory_limit_error = Some(reason);
    }
}

pub(crate) fn record_serving_session(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    prompt_token_count: u32,
    cached_token_count: u32,
    prefill_tok_per_second: Option<f64>,
    generation_tok_per_second: Option<f64>,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot
            .serving_session
            .record_completed_request(
                prompt_token_count,
                cached_token_count,
                prefill_tok_per_second,
                generation_tok_per_second,
            );
    }
}

pub(crate) fn record_prompt_work_reuse(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    prompt_work_reuse: WorkerPromptWorkReuse,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot
            .serving_session
            .record_prompt_work_reuse(prompt_work_reuse);
    }
}
