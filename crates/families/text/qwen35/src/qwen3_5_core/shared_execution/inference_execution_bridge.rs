//! The methods of the engine-neutral execution contract that both Qwen 3.5
//! engines answer the same way (issue #2473): each forwards to a shared
//! step. Rust lets only the crate that owns a trait or a type implement one
//! for the other, so the two impls cannot share a blanket implementation;
//! each expands this one list inside its own `impl MlxInferenceExecution`,
//! and adds only what differs (the resident engine's Neural Engine
//! preparation). The expansion is the code each impl held before, so the
//! decode dispatch is unchanged.

/// Expands the shared methods inside an engine's `impl MlxInferenceExecution`.
/// The engine's own `Qwen3_5EngineState` must be in scope at the call site,
/// since its inherent `start_generation` is the one called.
macro_rules! qwen3_5_shared_inference_execution_methods {
    () => {
        fn load(&mut self) -> Result<$crate::inference_engine::load_result::EngineLoadResult, $crate::InferenceEngineError> {
            $crate::qwen3_5_core::shared_execution::model_loading_lifecycle::load_model(self)
        }

        fn start_generation(
            &mut self,
            inference_request: Self::Request,
        ) -> Result<$crate::EngineGenerationStart, $crate::InferenceEngineError> {
            Qwen3_5EngineState::start_generation(self, inference_request)
        }

        fn decode_next_token(
            &mut self,
            request_id: ::ipc_protocol::RequestId,
        ) -> Result<$crate::GeneratedToken, $crate::InferenceEngineError> {
            $crate::qwen3_5_core::shared_execution::decode_step::advance_generation(self, request_id)
        }

        fn inject_input_tokens(
            &mut self,
            request_id: ::ipc_protocol::RequestId,
            input_token_ids: Vec<u32>,
        ) -> Result<(), $crate::InferenceEngineError> {
            $crate::qwen3_5_core::shared_execution::input_injection::inject_input_tokens(
                self,
                request_id,
                input_token_ids,
            )
        }

        fn cancel_generation(
            &mut self,
            request_id: ::ipc_protocol::RequestId,
        ) -> Result<$crate::GenerationFinalization, $crate::InferenceEngineError> {
            Ok($crate::qwen3_5_core::shared_execution::generation_finalization_steps::cancel_generation(
                self, request_id,
            ))
        }
        fn collect_astro_ssd_prompt_cache_stats(
            &self,
        ) -> Result<Option<::ipc_protocol::WorkerEvent>, $crate::InferenceEngineError> {
            Ok(self.core.astro_ssd_prompt_cache_stats_event())
        }

        fn clear_astro_ssd_prompt_cache(
            &mut self,
            model_id: Option<String>,
        ) -> Result<Option<::ipc_protocol::WorkerEvent>, $crate::InferenceEngineError> {
            self.core.clear_astro_ssd_prompt_cache(model_id.as_deref())
        }

        fn collect_mlx_memory_telemetry(
            &self,
        ) -> Result<Option<$crate::MlxMemoryTelemetry>, $crate::InferenceEngineError> {
            $crate::qwen3_5_core::shared_execution::generation_finalization_steps::collect_current_mlx_memory_telemetry(self)
        }

        fn update_mlx_memory_limit(
            &mut self,
            requested_mlx_memory_ceiling_bytes: u64,
        ) -> Result<$crate::MlxMemoryLimitAdjustment, $crate::InferenceEngineError> {
            $crate::qwen3_5_core::shared_execution::mlx_memory_limit_change::update_mlx_memory_limit(
                self,
                requested_mlx_memory_ceiling_bytes,
            )
        }
    };
}

pub(crate) use qwen3_5_shared_inference_execution_methods;
