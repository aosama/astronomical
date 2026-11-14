//! What is in memory when a forward starts.
//!
//! The memory governor sets aside what a forward needs before it runs, so
//! nothing is decided here (issue #2221). This file samples MLX and resets
//! its peak counter, so the sample taken after the forward measures that
//! one forward, and hands back what the measurement compares against.

use crate::{InferenceEngineError, PerformanceAttribution};

use super::{Qwen3_5EngineState, qwen3_5_runtime_error};

/// What was held when a forward began.
pub(in crate::qwen3_5_streaming) struct ForwardMemoryBaseline {
    pub(in crate::qwen3_5_streaming) active_memory_bytes: usize,
    pub(in crate::qwen3_5_streaming) retained_expert_payload_bytes: u64,
    /// Cumulative logical expert-streaming payload bytes when the forward
    /// began. The delta to the value after the forward is the expert pages
    /// that forward read (issue #691).
    pub(in crate::qwen3_5_streaming) streamed_expert_page_bytes: u64,
}

impl Qwen3_5EngineState {
    /// Samples what is held and resets MLX's peak counter. That counter is
    /// process-global: resetting it here means the next sample measures
    /// this one forward and not model loading or an earlier forward.
    pub(in crate::qwen3_5_streaming) fn begin_forward_memory_measurement(
        &self,
        performance_attribution: &PerformanceAttribution,
    ) -> Result<ForwardMemoryBaseline, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let memory_snapshot_before_forward = model
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        model
            .runtime()
            .reset_peak_memory()
            .map_err(qwen3_5_runtime_error)?;
        model.begin_measuring_a_new_forward();
        Ok(ForwardMemoryBaseline {
            active_memory_bytes: memory_snapshot_before_forward.active_memory_bytes(),
            retained_expert_payload_bytes: model.reclaimable_retained_payload_bytes(),
            streamed_expert_page_bytes: performance_attribution
                .expert_streaming_payload_byte_count(),
        })
    }
}
