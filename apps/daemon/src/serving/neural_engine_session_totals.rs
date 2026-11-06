//! What the Neural Engine has done since the daemon started, for people asking whether
//! it is earning its memory.

use astronomical_ipc_protocol::{
    NeuralEngineChunkPlacement, NeuralEngineChunkReport, NeuralEnginePrefillStatus,
};

const WHOLE_SPEED_PERCENT: u64 = 100;

/// Time saved has to be an estimate, because a chunk cannot run both ways at once: what a
/// warm chunk on both engines took, times what the model's measured speed says the GPU
/// alone would have added. Chunks of prompts that woke cold programs earn nothing, because
/// a 2B model measured 0.94× across such prompts while its later chunks still gained, so
/// this can undercount a large model and will not claim time a small one lost.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NeuralEngineSessionTotals {
    pub prompt_token_count: u64,
    pub neural_engine_token_count: u64,
    pub estimated_saved_millis: u64,
    /// The latest chunk's, cleared with the active request's progress, however that
    /// request ended, so it only describes one that is running.
    pub last_chunk_placement: Option<NeuralEngineChunkPlacement>,
    measured_speed_percent: Option<u32>,
    /// Warm time from a model still being compared, which earns its credit only if the
    /// comparison finds the programs worthwhile.
    warm_millis_awaiting_speed: u64,
}

impl NeuralEngineSessionTotals {
    pub const fn empty() -> Self {
        Self {
            prompt_token_count: 0,
            neural_engine_token_count: 0,
            estimated_saved_millis: 0,
            last_chunk_placement: None,
            measured_speed_percent: None,
            warm_millis_awaiting_speed: 0,
        }
    }

    pub fn record_chunk(&mut self, report: NeuralEngineChunkReport) {
        self.prompt_token_count = self
            .prompt_token_count
            .saturating_add(u64::from(report.chunk_token_count));
        self.neural_engine_token_count = self
            .neural_engine_token_count
            .saturating_add(u64::from(report.neural_engine_token_count));
        self.last_chunk_placement = Some(report.placement);
        if report.placement != NeuralEngineChunkPlacement::BothEngines || !report.is_prompt_warm {
            return;
        }
        match self.measured_speed_percent {
            Some(measured_speed_percent) => {
                self.credit_saved_time(report.elapsed_millis, measured_speed_percent);
            }
            None => {
                self.warm_millis_awaiting_speed = self
                    .warm_millis_awaiting_speed
                    .saturating_add(report.elapsed_millis);
            }
        }
    }

    /// Follows the loaded model's status. A measured speed settles the time waiting for
    /// it; anything else ends that model's claim to it.
    pub fn note_prefill_status(&mut self, status: Option<&NeuralEnginePrefillStatus>) {
        match status {
            Some(NeuralEnginePrefillStatus::Active {
                measured_speed_percent: Some(measured_speed_percent),
                ..
            }) => {
                let waiting_millis = std::mem::take(&mut self.warm_millis_awaiting_speed);
                self.credit_saved_time(waiting_millis, *measured_speed_percent);
                self.measured_speed_percent = Some(*measured_speed_percent);
            }
            Some(NeuralEnginePrefillStatus::Active {
                measured_speed_percent: None,
                ..
            }) => self.measured_speed_percent = None,
            _ => {
                self.warm_millis_awaiting_speed = 0;
                self.measured_speed_percent = None;
            }
        }
    }

    pub const fn clear_last_chunk_placement(&mut self) {
        self.last_chunk_placement = None;
    }

    fn credit_saved_time(&mut self, both_engines_millis: u64, measured_speed_percent: u32) {
        let gained_percent = u64::from(measured_speed_percent).saturating_sub(WHOLE_SPEED_PERCENT);
        self.estimated_saved_millis = self.estimated_saved_millis.saturating_add(
            both_engines_millis.saturating_mul(gained_percent) / WHOLE_SPEED_PERCENT,
        );
    }
}

#[cfg(test)]
#[path = "neural_engine_session_totals_tests.rs"]
mod tests;
