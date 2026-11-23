//! The streaming engine's part of model-visible feedback injection. The
//! injection itself is shared with the resident engine in
//! `qwen3_5_core::shared_execution::input_injection`.

use crate::qwen3_5_core::shared_execution::input_injection::Qwen3_5InputInjectionHooks;

use crate::InferenceEngineError;

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;

impl Qwen3_5InputInjectionHooks for Qwen3_5EngineState {
    /// Injection extends the request's stored state between measured steps.
    /// The memory governor grows the context account to the projected extent
    /// first, so the kept weights have made room before the injected tokens
    /// run.
    fn prepare_for_injected_input(
        &mut self,
        _active_request: &mut Qwen3_5EngineRequest,
        projected_context_tokens: usize,
    ) -> Result<(), InferenceEngineError> {
        let model = self
            .core
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let projected_context_bytes = projected_context_tokens
            .saturating_mul(self.core.context_memory_reservation_bytes_per_token);
        model.govern_context_extent(u64::try_from(projected_context_bytes).unwrap_or(u64::MAX));
        Ok(())
    }
}
