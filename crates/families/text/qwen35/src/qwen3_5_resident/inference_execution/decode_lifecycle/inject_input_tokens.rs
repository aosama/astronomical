//! The resident engine's part of model-visible feedback injection. The
//! injection itself is shared with the streaming engine in
//! `text_generation::generation_loop::input_injection`.

use crate::qwen3_5_resident::inference_execution::engine_request::ResidentSpeculativeRequest;
use text_generation::generation_loop::input_injection::TextInputInjectionHooks;

use crate::InferenceEngineError;

use super::Qwen3_5EngineState;
use super::engine_request::ResidentEngineRequest;

impl TextInputInjectionHooks for Qwen3_5EngineState {
    /// The projected context must fit beside the weights. The injected
    /// prefill bypasses the tap-capturing forward that grows the draft's
    /// context key-value state, so speculation suspends for the session's
    /// remainder; the plain decode step owns feedback exactly.
    fn prepare_for_injected_input(
        &mut self,
        active_request: &mut ResidentEngineRequest,
        projected_context_tokens: usize,
    ) -> Result<(), InferenceEngineError> {
        self.require_request_context_to_fit(projected_context_tokens)?;
        active_request.suspend_speculative_session("feedback_injection");
        Ok(())
    }
}
