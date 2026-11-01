//! Daemon-side accumulation of worker expert-streaming statistics.
//!
//! The worker pushes one `ExpertStreamingStats` event per finalized
//! generation, scoped to its engine load. This accumulator turns that stream
//! into observatory state: the latest snapshot, daemon-accumulated session
//! token totals, and a bounded ring of expert-memory-mode transitions that
//! becomes the observatory decision feed. All methods are pure state
//! transitions — timestamps arrive as arguments so tests stay deterministic.

use astronomical_ipc_protocol::{ExpertMemoryMode, WorkerEvent, WorkerExpertStreamingStats};

/// Transitions kept for the observatory decision feed before the oldest
/// is dropped. Twenty-four covers a long interactive session's promotions
/// and demotions without unbounded growth.
pub(crate) const MAXIMUM_MODE_TRANSITION_HISTORY: usize = 24;

/// One observed expert-memory-mode change, timestamped by the daemon.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertMemoryModeTransition {
    pub observed_at_unix_millis: u64,
    pub previous_mode: Option<ExpertMemoryMode>,
    pub next_mode: ExpertMemoryMode,
}

/// Accumulated observatory state derived from worker events.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertStreamingObservatory {
    /// The newest worker snapshot; `None` until the first stats event.
    pub latest_stats: Option<WorkerExpertStreamingStats>,
    /// The mode the worker most recently reported, from a stats event or an
    /// out-of-band mode change. Tracked independently of `latest_stats`
    /// because a mode change can arrive before any generation finishes;
    /// reading the previous mode off `latest_stats` left early transitions
    /// labelled `previous_mode: None` even after the mode was known
    /// (issue #1715).
    current_expert_memory_mode: Option<ExpertMemoryMode>,
    /// Session totals accumulated across every stats event; the worker
    /// reports per-request counts by contract.
    pub accumulated_prompt_token_count: u64,
    pub accumulated_generated_token_count: u64,
    /// Bounded, oldest-first ring of mode transitions for the decision feed.
    pub mode_transitions: Vec<ExpertMemoryModeTransition>,
    /// Daemon wall-clock of the newest stats event.
    pub updated_at_unix_millis: Option<u64>,
}

impl ExpertStreamingObservatory {
    /// Const-empty because the health snapshot's constructors are const fns.
    pub(crate) const fn empty() -> Self {
        Self {
            latest_stats: None,
            current_expert_memory_mode: None,
            accumulated_prompt_token_count: 0,
            accumulated_generated_token_count: 0,
            mode_transitions: Vec::new(),
            updated_at_unix_millis: None,
        }
    }

    /// Folds one worker stats event into observatory state. A different
    /// model id than the latest snapshot means a new engine load (worker
    /// restart or model swap), so accumulation restarts with the event: the
    /// worker's counters are engine-load-scoped by contract.
    pub(crate) fn record_stats(
        &mut self,
        worker_expert_streaming_stats: WorkerExpertStreamingStats,
        observed_at_unix_millis: u64,
    ) {
        let model_changed_from_latest = match (
            self.latest_stats.as_ref(),
            worker_expert_streaming_stats.model_id.as_ref(),
        ) {
            (Some(latest_stats), Some(incoming_model_id)) => {
                latest_stats.model_id.as_ref() != Some(incoming_model_id)
            }
            _ => false,
        };
        if model_changed_from_latest {
            self.mode_transitions.clear();
            self.accumulated_prompt_token_count = 0;
            self.accumulated_generated_token_count = 0;
            self.latest_stats = None;
            // A new engine load restarts its own mode story; the first stats
            // event of the load opens it without a predecessor.
            self.current_expert_memory_mode = None;
        }
        let previous_mode = self.current_expert_memory_mode;
        if previous_mode != Some(worker_expert_streaming_stats.expert_memory_mode) {
            self.push_mode_transition(ExpertMemoryModeTransition {
                observed_at_unix_millis,
                previous_mode,
                next_mode: worker_expert_streaming_stats.expert_memory_mode,
            });
        }
        self.current_expert_memory_mode = Some(worker_expert_streaming_stats.expert_memory_mode);
        self.accumulated_prompt_token_count = self
            .accumulated_prompt_token_count
            .saturating_add(worker_expert_streaming_stats.prompt_token_count);
        self.accumulated_generated_token_count = self
            .accumulated_generated_token_count
            .saturating_add(worker_expert_streaming_stats.generated_token_count);
        self.updated_at_unix_millis = Some(observed_at_unix_millis);
        self.latest_stats = Some(worker_expert_streaming_stats);
    }

    /// Records an out-of-band mode change so the feed stays truthful between
    /// generation boundaries; the next stats event reconciles the snapshot.
    pub(crate) fn record_mode_change(
        &mut self,
        next_mode: ExpertMemoryMode,
        observed_at_unix_millis: u64,
    ) {
        let previous_mode = self.current_expert_memory_mode;
        if previous_mode == Some(next_mode) {
            return;
        }
        self.push_mode_transition(ExpertMemoryModeTransition {
            observed_at_unix_millis,
            previous_mode,
            next_mode,
        });
        // The tracked mode moves even without a snapshot, so a later
        // transition before any stats event still knows its predecessor.
        self.current_expert_memory_mode = Some(next_mode);
        if let Some(latest_stats) = self.latest_stats.as_mut() {
            latest_stats.expert_memory_mode = next_mode;
        }
    }

    /// Appends one transition, dropping the oldest beyond the bound.
    fn push_mode_transition(&mut self, mode_transition: ExpertMemoryModeTransition) {
        if self.mode_transitions.len() >= MAXIMUM_MODE_TRANSITION_HISTORY {
            self.mode_transitions.remove(0);
        }
        self.mode_transitions.push(mode_transition);
    }
}

/// Extracts the stats payload from the event for the publisher.
pub(crate) fn expert_streaming_stats_from_event(
    expert_streaming_stats_event: &WorkerEvent,
) -> Option<WorkerExpertStreamingStats> {
    match expert_streaming_stats_event {
        WorkerEvent::ExpertStreamingStats { stats } => Some(stats.clone()),
        _ => None,
    }
}
