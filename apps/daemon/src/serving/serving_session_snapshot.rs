use ipc_protocol::WorkerPromptWorkReuse;

use super::neural_engine_session_totals::NeuralEngineSessionTotals;

/// What one completed request contributes to the session's two speeds: the
/// tokens each phase processed and how long that phase took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletedRequestWork {
    pub prompt_token_count: u32,
    pub astro_ssd_prompt_cache_restored_token_count: u32,
    /// The prompt tokens the model computed during its prompt time, as the
    /// worker's prompt progress counted them: the prompt less every prefix
    /// restored or adopted rather than computed.
    pub computed_prompt_token_count: u32,
    pub generated_token_count: u16,
    pub prefill_elapsed_millis: u64,
    pub generation_elapsed_millis: u64,
}

/// One phase's work so far: tokens and the time they took.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PhaseWork {
    token_count: u64,
    elapsed_millis: u64,
}

impl PhaseWork {
    /// The phase's speed with `running` added, or `None` while neither has
    /// both tokens and time: a session with nothing measured has no speed.
    fn tokens_per_second_with(self, running: Self) -> Option<f64> {
        let token_count = self.token_count.saturating_add(running.token_count);
        let elapsed_millis = self.elapsed_millis.saturating_add(running.elapsed_millis);
        if token_count == 0 || elapsed_millis == 0 {
            return None;
        }
        Some(token_count as f64 / (elapsed_millis as f64 / 1000.0))
    }
}

/// Compact lifetime-of-daemon serving summary rendered by local interfaces.
///
/// The two speeds are averages over the whole session, the running request
/// included: completed work plus the running request's work so far, so a
/// person watching a long reply sees the session figure move with it rather
/// than wait for the reply to end.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingSessionSnapshot {
    pub completed_request_count: u64,
    pub total_prompt_token_count: u64,
    pub total_reused_prompt_token_count: u64,
    pub target_prompt_work_token_count: u64,
    pub target_reused_prompt_work_token_count: u64,
    pub average_prefill_tok_per_second: f64,
    pub average_generation_tok_per_second: f64,
    pub neural_engine: NeuralEngineSessionTotals,
    completed_prefill: PhaseWork,
    completed_generation: PhaseWork,
    running_prefill: PhaseWork,
    running_generation: PhaseWork,
}

impl ServingSessionSnapshot {
    pub const fn empty() -> Self {
        Self {
            completed_request_count: 0,
            total_prompt_token_count: 0,
            total_reused_prompt_token_count: 0,
            target_prompt_work_token_count: 0,
            target_reused_prompt_work_token_count: 0,
            average_prefill_tok_per_second: 0.0,
            average_generation_tok_per_second: 0.0,
            neural_engine: NeuralEngineSessionTotals::empty(),
            completed_prefill: PhaseWork {
                token_count: 0,
                elapsed_millis: 0,
            },
            completed_generation: PhaseWork {
                token_count: 0,
                elapsed_millis: 0,
            },
            running_prefill: PhaseWork {
                token_count: 0,
                elapsed_millis: 0,
            },
            running_generation: PhaseWork {
                token_count: 0,
                elapsed_millis: 0,
            },
        }
    }

    /// Folds one request into the session and drops its running work in the
    /// same step, so a reader never sees it counted twice or not at all.
    /// Each speed is the session's total tokens over its total time, not the
    /// mean of the requests' own speeds: a mean lets a five-token reply, whose
    /// speed is mostly noise, move the figure as far as a two-thousand-token
    /// one. The reading speed counts only computed prompt tokens, the same
    /// count the running work used, so a request's end never makes it jump.
    pub fn record_completed_request(&mut self, completed_request_work: CompletedRequestWork) {
        let prompt_token_count = completed_request_work.prompt_token_count;
        let restored_token_count = completed_request_work
            .astro_ssd_prompt_cache_restored_token_count
            .min(prompt_token_count);
        self.completed_request_count = self.completed_request_count.saturating_add(1);
        self.total_prompt_token_count = self
            .total_prompt_token_count
            .saturating_add(u64::from(prompt_token_count));
        self.total_reused_prompt_token_count = self
            .total_reused_prompt_token_count
            .saturating_add(u64::from(restored_token_count));
        fold_measured_work(
            &mut self.completed_prefill,
            u64::from(completed_request_work.computed_prompt_token_count),
            completed_request_work.prefill_elapsed_millis,
        );
        fold_measured_work(
            &mut self.completed_generation,
            u64::from(completed_request_work.generated_token_count),
            completed_request_work.generation_elapsed_millis,
        );
        self.running_prefill = PhaseWork::default();
        self.running_generation = PhaseWork::default();
        self.refresh_averages();
    }

    /// The running request's prompt work so far: the tokens computed and the
    /// worker's measured prompt time, both cumulative.
    pub fn record_running_prefill(&mut self, computed_token_count: u32, elapsed_millis: u64) {
        self.running_prefill = PhaseWork {
            token_count: u64::from(computed_token_count),
            elapsed_millis,
        };
        self.refresh_averages();
    }

    /// The running request's reply so far: tokens written and the writing
    /// time up to the last of them.
    pub fn record_running_generation(&mut self, generated_token_count: u32, elapsed_millis: u64) {
        self.running_generation = PhaseWork {
            token_count: u64::from(generated_token_count),
            elapsed_millis,
        };
        self.refresh_averages();
    }

    /// Drops the running request's work when it ends without completing
    /// (failure, cancellation, a replaced worker): what it did is not served
    /// work the session can stand behind.
    pub fn clear_running_request(&mut self) {
        self.running_prefill = PhaseWork::default();
        self.running_generation = PhaseWork::default();
        self.refresh_averages();
    }

    /// The two averages from completed plus running work. A phase with
    /// nothing measured keeps 0, which readers show as no speed.
    fn refresh_averages(&mut self) {
        self.average_prefill_tok_per_second = self
            .completed_prefill
            .tokens_per_second_with(self.running_prefill)
            .unwrap_or(0.0);
        self.average_generation_tok_per_second = self
            .completed_generation
            .tokens_per_second_with(self.running_generation)
            .unwrap_or(0.0);
    }

    pub fn record_prompt_work_reuse(&mut self, prompt_work_reuse: WorkerPromptWorkReuse) {
        self.target_prompt_work_token_count = self
            .target_prompt_work_token_count
            .saturating_add(prompt_work_reuse.target_eligible_token_count);
        self.target_reused_prompt_work_token_count =
            self.target_reused_prompt_work_token_count.saturating_add(
                prompt_work_reuse
                    .target_restored_token_count
                    .min(prompt_work_reuse.target_eligible_token_count),
            );
    }
}

/// Adds one request's phase to the completed work. A phase with no time or
/// no tokens has nothing to measure and leaves the work as it was.
fn fold_measured_work(phase_work: &mut PhaseWork, token_count: u64, elapsed_millis: u64) {
    if token_count == 0 || elapsed_millis == 0 {
        return;
    }
    phase_work.token_count = phase_work.token_count.saturating_add(token_count);
    phase_work.elapsed_millis = phase_work.elapsed_millis.saturating_add(elapsed_millis);
}

#[cfg(test)]
#[path = "serving_session_snapshot_tests.rs"]
mod serving_session_snapshot_tests;
