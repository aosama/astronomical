use astronomical_ipc_protocol::WorkerPromptWorkReuse;

/// Compact lifetime-of-daemon serving summary rendered by local interfaces.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServingSessionSnapshot {
    pub completed_request_count: u64,
    pub total_prompt_token_count: u64,
    pub total_reused_prompt_token_count: u64,
    pub target_prompt_work_token_count: u64,
    pub target_reused_prompt_work_token_count: u64,
    pub drafter_prompt_work_token_count: u64,
    pub drafter_reused_prompt_work_token_count: u64,
    pub average_prefill_tok_per_second: f64,
    pub average_generation_tok_per_second: f64,
    prefill_measurement_count: u64,
    generation_measurement_count: u64,
}

impl ServingSessionSnapshot {
    pub const fn empty() -> Self {
        Self {
            completed_request_count: 0,
            total_prompt_token_count: 0,
            total_reused_prompt_token_count: 0,
            target_prompt_work_token_count: 0,
            target_reused_prompt_work_token_count: 0,
            drafter_prompt_work_token_count: 0,
            drafter_reused_prompt_work_token_count: 0,
            average_prefill_tok_per_second: 0.0,
            average_generation_tok_per_second: 0.0,
            prefill_measurement_count: 0,
            generation_measurement_count: 0,
        }
    }

    pub fn record_completed_request(
        &mut self,
        prompt_token_count: u32,
        cached_token_count: u32,
        prefill_tok_per_second: Option<f64>,
        generation_tok_per_second: Option<f64>,
    ) {
        self.completed_request_count = self.completed_request_count.saturating_add(1);
        self.total_prompt_token_count = self
            .total_prompt_token_count
            .saturating_add(u64::from(prompt_token_count));
        self.total_reused_prompt_token_count = self
            .total_reused_prompt_token_count
            .saturating_add(u64::from(cached_token_count.min(prompt_token_count)));
        if let Some(prefill_tok_per_second) = prefill_tok_per_second {
            self.average_prefill_tok_per_second = rolling_average(
                self.average_prefill_tok_per_second,
                self.prefill_measurement_count,
                prefill_tok_per_second,
            );
            self.prefill_measurement_count = self.prefill_measurement_count.saturating_add(1);
        }
        if let Some(generation_tok_per_second) = generation_tok_per_second {
            self.average_generation_tok_per_second = rolling_average(
                self.average_generation_tok_per_second,
                self.generation_measurement_count,
                generation_tok_per_second,
            );
            self.generation_measurement_count = self.generation_measurement_count.saturating_add(1);
        }
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
        self.drafter_prompt_work_token_count = self
            .drafter_prompt_work_token_count
            .saturating_add(prompt_work_reuse.drafter_eligible_token_count);
        self.drafter_reused_prompt_work_token_count =
            self.drafter_reused_prompt_work_token_count.saturating_add(
                prompt_work_reuse
                    .drafter_restored_token_count
                    .min(prompt_work_reuse.drafter_eligible_token_count),
            );
    }
}

fn rolling_average(current_average: f64, prior_count: u64, new_measurement: f64) -> f64 {
    let prior_total = current_average * prior_count as f64;
    (prior_total + new_measurement) / prior_count.saturating_add(1) as f64
}
