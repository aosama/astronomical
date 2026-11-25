//! Methods the resident and streaming Qwen 3.5 models answer the same way
//! (issue #2473). The two models are different types, and the evaluation
//! methods are inherent while the text-model methods implement a trait whose
//! implementation each model owns, so neither a shared impl nor a blanket one
//! can hold them; each model expands these lists inside its own `impl`. The
//! expansion is the code each model held before, so the decode path compiles
//! to the same calls.

/// Expands, inside a model's inherent `impl`, the evaluation methods both
/// models share. The model must answer `select_highest_logit_token`,
/// `evaluate_forward_state_with_performance_attribution` and reach its
/// runtime as `self.runtime`.
macro_rules! qwen3_5_shared_model_evaluation_methods {
    () => {
        /// Returns the highest-logit token ID for one final-position output.
        pub fn highest_logit_token_id(
            &self,
            final_position_logits: &::mlx_bridge::MlxArray,
        ) -> Result<u32, ::text_generation::execution_error::TextExecutionError> {
            let selected_token = self.select_highest_logit_token(final_position_logits)?;
            Ok(selected_token.item_u32()?)
        }

        /// Materializes final logits and mutable decoder state for one forward pass.
        pub fn evaluate_forward_state(
            &self,
            final_logits: &::mlx_bridge::MlxArray,
            request_decoder_state: &::text_generation::RequestDecoderStateStack,
        ) -> Result<(), ::text_generation::execution_error::TextExecutionError> {
            let mut disabled_performance_attribution = $crate::PerformanceAttribution::disabled();
            self.evaluate_forward_state_with_performance_attribution(
                final_logits,
                request_decoder_state,
                &mut disabled_performance_attribution,
            )?;
            Ok(())
        }

        /// Submits decode evaluation without synchronizing the graphics processor.
        pub fn async_evaluate_generation(
            &self,
            generated_token: &::mlx_bridge::MlxArray,
            request_decoder_state: &::text_generation::RequestDecoderStateStack,
        ) -> Result<(), ::text_generation::execution_error::TextExecutionError> {
            let evaluation_arrays =
                $crate::qwen3_5_core::model_math::forward_contract::forward_state_arrays(
                    generated_token,
                    request_decoder_state,
                )?;
            self.runtime.async_eval_arrays(&evaluation_arrays)?;
            Ok(())
        }
    };
}

/// Expands, inside a model's `impl TextModel`, the methods both models
/// forward to the same place. Calls name the model type, not method syntax,
/// so a renamed inherent method fails to compile instead of recursing into
/// the trait.
macro_rules! qwen3_5_shared_text_model_forwarding {
    ($model:ident) => {
        fn expert_residency_telemetry_for_breakdown(
            &self,
            active_memory_breakdown: &$crate::memory::telemetry::MlxActiveMemoryBreakdown,
        ) -> $crate::inference_engine::contract::ExpertResidencyTelemetry {
            $model::expert_residency_telemetry_for_breakdown(self, active_memory_breakdown)
        }

        fn latest_memory_ledger_reading(&self) -> Option<::memory_policy::ledger::LedgerReading> {
            $model::latest_memory_ledger_reading(self)
        }

        fn minimum_mlx_memory_ceiling_bytes(&self) -> Result<u64, $crate::InferenceEngineError> {
            $model::minimum_mlx_memory_ceiling_bytes(self)
        }

        fn prefill_chunk_with_performance_attribution(
            &self,
            token_ids: &[u32],
            starting_position_tokens: u32,
            request_decoder_state: &mut ::text_generation::RequestDecoderStateStack,
            performance_attribution: &mut $crate::PerformanceAttribution,
        ) -> Result<(), ::text_generation::execution_error::TextExecutionError> {
            $crate::qwen3_5_core::model::prefill_forward::Qwen3_5PrefillForward::prefill_chunk_with_performance_attribution(
                self,
                token_ids,
                starting_position_tokens,
                request_decoder_state,
                performance_attribution,
            )
        }

        fn build_forward_chunk_with_performance_attribution(
            &self,
            token_ids: &[u32],
            starting_position_tokens: u32,
            request_decoder_state: &mut ::text_generation::RequestDecoderStateStack,
            performance_attribution: &mut $crate::PerformanceAttribution,
        ) -> Result<::mlx_bridge::MlxArray, ::text_generation::execution_error::TextExecutionError> {
            $model::build_forward_chunk_with_performance_attribution(
                self,
                token_ids,
                starting_position_tokens,
                request_decoder_state,
                performance_attribution,
            )
        }

        fn async_evaluate_generation(
            &self,
            generated_token: &::mlx_bridge::MlxArray,
            request_decoder_state: &::text_generation::RequestDecoderStateStack,
        ) -> Result<(), ::text_generation::execution_error::TextExecutionError> {
            $model::async_evaluate_generation(self, generated_token, request_decoder_state)
        }
    };
}

pub(crate) use qwen3_5_shared_model_evaluation_methods;
pub(crate) use qwen3_5_shared_text_model_forwarding;
