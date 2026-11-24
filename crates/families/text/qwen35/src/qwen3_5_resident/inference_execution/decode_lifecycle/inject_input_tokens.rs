//! The resident engine's part of model-visible feedback injection. The
//! injection itself is shared with the streaming engine in
//! `qwen3_5_core::shared_execution::input_injection`.

use crate::qwen3_5_core::shared_execution::input_injection::Qwen3_5InputInjectionHooks;

use crate::InferenceEngineError;

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;

impl Qwen3_5InputInjectionHooks for Qwen3_5EngineState {
    /// The projected context must fit beside the weights. The injected
    /// prefill bypasses the tap-capturing forward that grows the draft's
    /// context key-value state, so speculation suspends for the session's
    /// remainder; the plain decode step owns feedback exactly.
    fn prepare_for_injected_input(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        projected_context_tokens: usize,
    ) -> Result<(), InferenceEngineError> {
        self.require_request_context_to_fit(projected_context_tokens)?;
        active_request.suspend_speculative_session("feedback_injection");
        Ok(())
    }
}
