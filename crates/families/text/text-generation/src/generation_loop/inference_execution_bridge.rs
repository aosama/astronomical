//! The methods of the engine-neutral execution contract that every text
//! engine answers the same way (issue #2473): each forwards to a shared
//! step. Rust lets only the crate that owns a trait or a type implement one
//! for the other, so the engines cannot share a blanket implementation;
//! each expands this one list inside its own `impl MlxInferenceExecution`,
//! and adds only what differs (the resident Qwen 3.5 engine's Neural Engine
//! preparation). The expansion is the code each impl held before, so the
//! decode dispatch is unchanged.

/// Expands the shared methods inside an engine's `impl MlxInferenceExecution`.
/// The engine names the type whose inherent `start_generation` begins a
/// request, since that step differs between engines.
#[macro_export]
macro_rules! text_shared_inference_execution_methods {
    ($engine_state:ty) => {
        fn load(&mut self) -> Result<$crate::generation_loop::inference_execution_bridge::contract_types::EngineLoadResult, $crate::generation_loop::inference_execution_bridge::contract_types::InferenceEngineError> {
            $crate::generation_loop::model_loading_lifecycle::load_model(self)
        }

        fn start_generation(
            &mut self,
            inference_request: Self::Request,
        ) -> Result<$crate::generation_loop::inference_execution_bridge::contract_types::EngineGenerationStart, $crate::generation_loop::inference_execution_bridge::contract_types::InferenceEngineError> {
            <$engine_state>::start_generation(self, inference_request)
        }

        fn decode_next_token(
            &mut self,
            request_id: ::ipc_protocol::RequestId,
        ) -> Result<$crate::generation_loop::inference_execution_bridge::contract_types::GeneratedToken, $crate::generation_loop::inference_execution_bridge::contract_types::InferenceEngineError> {
            $crate::generation_loop::decode_step::advance_generation(self, request_id)
        }

        fn inject_input_tokens(
            &mut self,
            request_id: ::ipc_protocol::RequestId,
            input_token_ids: Vec<u32>,
        ) -> Result<(), $crate::generation_loop::inference_execution_bridge::contract_types::InferenceEngineError> {
            $crate::generation_loop::input_injection::inject_input_tokens(
                self,
                request_id,
                input_token_ids,
            )
        }

        fn cancel_generation(
            &mut self,
            request_id: ::ipc_protocol::RequestId,
        ) -> Result<$crate::generation_loop::inference_execution_bridge::contract_types::GenerationFinalization, $crate::generation_loop::inference_execution_bridge::contract_types::InferenceEngineError> {
            Ok($crate::generation_loop::generation_finalization_steps::cancel_generation(
                self, request_id,
            ))
        }
        fn collect_astro_ssd_prompt_cache_stats(
            &self,
        ) -> Result<Option<::ipc_protocol::WorkerEvent>, $crate::generation_loop::inference_execution_bridge::contract_types::InferenceEngineError> {
            Ok($crate::generation_loop::execution_core::TextExecutionCoreOwner::execution_core(self)
                .astro_ssd_prompt_cache_stats_event())
        }

        fn clear_astro_ssd_prompt_cache(
            &mut self,
            model_id: Option<String>,
        ) -> Result<Option<::ipc_protocol::WorkerEvent>, $crate::generation_loop::inference_execution_bridge::contract_types::InferenceEngineError> {
            $crate::generation_loop::execution_core::TextExecutionCoreOwner::execution_core(self)
                .clear_astro_ssd_prompt_cache(model_id.as_deref())
        }

        fn collect_mlx_memory_telemetry(
            &self,
        ) -> Result<Option<$crate::generation_loop::inference_execution_bridge::contract_types::MlxMemoryTelemetry>, $crate::generation_loop::inference_execution_bridge::contract_types::InferenceEngineError> {
            $crate::generation_loop::generation_finalization_steps::collect_current_mlx_memory_telemetry(self)
        }

        fn update_mlx_memory_limit(
            &mut self,
            requested_mlx_memory_ceiling_bytes: u64,
        ) -> Result<$crate::generation_loop::inference_execution_bridge::contract_types::MlxMemoryLimitAdjustment, $crate::generation_loop::inference_execution_bridge::contract_types::InferenceEngineError> {
            $crate::generation_loop::mlx_memory_limit_change::update_mlx_memory_limit(
                self,
                requested_mlx_memory_ceiling_bytes,
            )
        }
    };
}

/// The engine-core types the expansion names, re-exported so it resolves
/// them through this crate from whichever family expands it.
#[doc(hidden)]
pub mod contract_types {
    pub use engine_core::InferenceEngineError;
    pub use engine_core::inference_engine::contract::{
        EngineGenerationStart, GeneratedToken, GenerationFinalization,
    };
    pub use engine_core::inference_engine::load_result::EngineLoadResult;
    pub use memory_policy::telemetry::{MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
}
