//! The folded-ternary engine's answers to the shared Qwen 3.5 feedback
//! injection (issue #2473, Prism step 10).
//!
//! The shared injection checks the feedback against the vocabulary and the
//! pack's position range, reads its prefix as one prompt chunk, forwards its
//! last token to sample the next reply token, and finalizes the request when
//! any of that fails. The folded-ternary engine keeps no memory governor and
//! never drafts, so its answers are the plain ones.

use runtime_integration::MlxMemorySnapshot;

use super::{PrismHadamardQwen35ActiveGeneration, PrismHadamardQwen35InferenceExecution};
use crate::RequestDecoderStateStack;
use crate::{InferenceEngineError, PerformanceAttribution};
use text_generation::generation_loop::forward_memory_governance::{
    TextForwardKind, TextForwardMemoryGovernance,
};
use text_generation::generation_loop::input_injection::TextInputInjectionHooks;
use text_generation::model::text_model::TextModel;

/// The dense pack is resident from load on, so no forward is admitted or
/// measured against a governor; the reading after a forward is the plain MLX
/// memory reading.
impl TextForwardMemoryGovernance for PrismHadamardQwen35InferenceExecution {
    type ForwardInProgress = ();

    fn begin_governed_forward(
        &mut self,
        _forward_kind: TextForwardKind,
        _request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }

    fn complete_governed_forward(
        &self,
        _forward_in_progress: (),
        _stored_state_bytes: u64,
        _performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxMemorySnapshot, InferenceEngineError> {
        self.execution_core
            .loaded_model()?
            .memory_snapshot_for_telemetry()
    }
}

impl TextInputInjectionHooks for PrismHadamardQwen35InferenceExecution {
    /// There is no drafting to suspend and no governed context to grow; the
    /// shared projection against the pack's position range is the whole
    /// check.
    fn prepare_for_injected_input(
        &mut self,
        _active_generation: &mut PrismHadamardQwen35ActiveGeneration,
        _projected_context_tokens: usize,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "input_injection_hooks_tests.rs"]
mod input_injection_hooks_tests;
