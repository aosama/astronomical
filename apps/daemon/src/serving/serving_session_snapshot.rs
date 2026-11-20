use ipc_protocol::WorkerPromptWorkReuse;

use super::neural_engine_session_totals::NeuralEngineSessionTotals;

/// What one completed request contributes to the session's two speeds: the
/// tokens each phase processed and how long that phase took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletedRequestWork {
    pub prompt_token_count: u32,
    pub astro_ssd_prompt_cache_restored_token_count: u32,
    pub generated_token_count: u16,
    pub prefill_elapsed_millis: u64,
    pub generation_elapsed_millis: u64,
}

/// Compact lifetime-of-daemon serving summary rendered by local interfaces.
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
    prefilled_token_count: u64,
    prefill_elapsed_millis: u64,
    generated_token_count: u64,
    generation_elapsed_millis: u64,
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
            prefilled_token_count: 0,
            prefill_elapsed_millis: 0,
            generated_token_count: 0,
            generation_elapsed_millis: 0,
        }
    }

    /// Folds one request into the session. Each speed is the session's total
    /// tokens over its total time, not the mean of the requests' own speeds:
    /// a mean lets a five-token reply, whose speed is mostly noise, move the
    /// figure as far as a two-thousand-token one. Tokens restored from the
    /// `astro_ssd_prompt_cache` are left out of the prefill speed, because
    /// they were read back and not computed during the prefill time.
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

        self.record_prefill_work(
            prompt_token_count - restored_token_count,
            completed_request_work.prefill_elapsed_millis,
        );
        self.record_generation_work(
            completed_request_work.generated_token_count,
            completed_request_work.generation_elapsed_millis,
        );
    }

    /// Adds one request's prefill to the session's reading speed. A request
    /// with no prefill time or no computed token has nothing to measure and
    /// leaves the speed as it was.
    fn record_prefill_work(&mut self, prefilled_token_count: u32, prefill_elapsed_millis: u64) {
        if prefill_elapsed_millis == 0 || prefilled_token_count == 0 {
            return;
        }
        self.prefilled_token_count = self
            .prefilled_token_count
            .saturating_add(u64::from(prefilled_token_count));
        self.prefill_elapsed_millis = self
            .prefill_elapsed_millis
            .saturating_add(prefill_elapsed_millis);
        self.average_prefill_tok_per_second =
            tokens_per_second(self.prefilled_token_count, self.prefill_elapsed_millis);
    }

    /// Adds one request's generation to the session's writing speed, with the
    /// same rule for a request that has nothing to measure.
    fn record_generation_work(
        &mut self,
        generated_token_count: u16,
        generation_elapsed_millis: u64,
    ) {
        if generation_elapsed_millis == 0 || generated_token_count == 0 {
            return;
        }
        self.generated_token_count = self
            .generated_token_count
            .saturating_add(u64::from(generated_token_count));
        self.generation_elapsed_millis = self
            .generation_elapsed_millis
            .saturating_add(generation_elapsed_millis);
        self.average_generation_tok_per_second =
            tokens_per_second(self.generated_token_count, self.generation_elapsed_millis);
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

fn tokens_per_second(token_count: u64, elapsed_millis: u64) -> f64 {
    token_count as f64 / (elapsed_millis as f64 / 1000.0)
}
