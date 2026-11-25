//! What is in memory when a forward starts.
//!
//! The memory governor sets aside what a forward needs before it runs, so
//! nothing is decided here (issue #2221). This file samples MLX and resets
//! its peak counter, so the sample taken after the forward measures that
//! one forward, and hands back what the measurement compares against.

use crate::InferenceEngineError;

use text_generation::generation_loop::forward_memory_governance::{
    TextForwardKind, TextForwardMemoryGovernance,
};

use super::{Qwen3_5EngineState, fatal_runtime_error};

/// What was held when a forward began.
pub(in crate::qwen3_5_streaming) struct ForwardMemoryBaseline {
    pub(in crate::qwen3_5_streaming) active_memory_bytes: usize,
}

impl Qwen3_5EngineState {
    /// Samples what is held and resets MLX's peak counter. That counter is
    /// process-global: resetting it here means the next sample measures
    /// this one forward and not model loading or an earlier forward.
    pub(in crate::qwen3_5_streaming) fn begin_forward_memory_measurement(
        &self,
    ) -> Result<ForwardMemoryBaseline, InferenceEngineError> {
        let model = self.core.loaded_model()?;
        let memory_snapshot_before_forward = model
            .runtime()
            .memory_snapshot()
            .map_err(fatal_runtime_error)?;
        model
            .runtime()
            .reset_peak_memory()
            .map_err(fatal_runtime_error)?;
        model.begin_measuring_a_new_forward();
        Ok(ForwardMemoryBaseline {
            active_memory_bytes: memory_snapshot_before_forward.active_memory_bytes(),
        })
    }
}

/// What the streaming engine carries across one forward: the memory
/// baseline taken before it and what the forward is, for the governor.
pub struct StreamingForwardInProgress {
    memory_baseline: ForwardMemoryBaseline,
    measured_forward: super::completed_forward_memory::MeasuredForward,
}

/// The streaming engine never refuses a forward: it measures memory before
/// it and tells its governor afterwards what the forward used.
impl TextForwardMemoryGovernance for Qwen3_5EngineState {
    type ForwardInProgress = StreamingForwardInProgress;

    fn begin_governed_forward(
        &mut self,
        forward_kind: TextForwardKind,
        _request_decoder_state: &text_generation::RequestDecoderStateStack,
    ) -> Result<StreamingForwardInProgress, InferenceEngineError> {
        let measured_forward = match forward_kind {
            TextForwardKind::PromptChunk { token_count } => {
                super::completed_forward_memory::MeasuredForward::prompt(token_count)
            }
            TextForwardKind::ReplyToken => {
                super::completed_forward_memory::MeasuredForward::reply_token()
            }
        };
        Ok(StreamingForwardInProgress {
            memory_baseline: self.begin_forward_memory_measurement()?,
            measured_forward,
        })
    }

    fn complete_governed_forward(
        &self,
        forward_in_progress: StreamingForwardInProgress,
        stored_state_bytes: u64,
        performance_attribution: &mut crate::PerformanceAttribution,
    ) -> Result<runtime_integration::MlxMemorySnapshot, InferenceEngineError> {
        let model = self.core.loaded_model()?;
        Ok(super::completed_forward_memory::measure_completed_forward(
            model,
            forward_in_progress.measured_forward,
            stored_state_bytes,
            forward_in_progress.memory_baseline.active_memory_bytes,
            performance_attribution,
        )?
        .mlx_memory_snapshot)
    }
}
