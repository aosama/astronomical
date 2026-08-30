use astronomical_runtime_integration::{MlxArray, MlxMemorySnapshot};

use crate::{GeneratedToken, InferenceEngineError, PerformanceCounter, PerformanceOperation};

use super::engine_request::Qwen3_5EngineRequest;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5::Qwen3_5Model;

pub(in crate::qwen3_5) struct GeneratedTokenEmission {
    pub(in crate::qwen3_5) generated_token: GeneratedToken,
    pub(in crate::qwen3_5) is_terminal: bool,
}

impl Qwen3_5EngineState {
    pub(in crate::qwen3_5) fn generated_token_will_be_terminal(
        &self,
        active_request: &Qwen3_5EngineRequest,
        generated_token_id: u32,
    ) -> bool {
        self.end_of_sequence_token_ids.contains(&generated_token_id)
            || active_request.generated_token_count.saturating_add(1)
                >= active_request.maximum_output_tokens
    }

    pub(in crate::qwen3_5) fn build_generated_token_emission(
        &self,
        model: &Qwen3_5Model,
        active_request: &mut Qwen3_5EngineRequest,
        generated_token_id: u32,
        mlx_memory_snapshot: Option<&MlxMemorySnapshot>,
    ) -> Result<GeneratedTokenEmission, InferenceEngineError> {
        active_request.generated_token_count = active_request
            .generated_token_count
            .checked_add(1)
            .ok_or_else(|| fatal_engine_error("generated-token counter overflowed"))?;
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::GeneratedTokenCount, 1);

        let is_reasoning_token =
            active_request.observe_committed_thinking_token(generated_token_id)?;

        let is_terminal = self.end_of_sequence_token_ids.contains(&generated_token_id)
            || active_request.generated_token_count >= active_request.maximum_output_tokens;
        let mlx_memory_telemetry = mlx_memory_snapshot
            .map(|mlx_memory_snapshot| {
                let mlx_active_memory_bytes =
                    u64::try_from(mlx_memory_snapshot.active_memory_bytes()).map_err(|_| {
                        fatal_engine_error("MLX active memory bytes exceed the u64 range")
                    })?;
                Ok::<crate::MlxMemoryTelemetry, InferenceEngineError>(
                    crate::MlxMemoryTelemetry::new(
                        mlx_active_memory_bytes,
                        u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes()).map_err(
                            |_| {
                                fatal_engine_error(
                                    "MLX allocator-cache memory bytes exceed the u64 range",
                                )
                            },
                        )?,
                        u64::try_from(mlx_memory_snapshot.peak_memory_bytes()).map_err(|_| {
                            fatal_engine_error("MLX peak memory bytes exceed the u64 range")
                        })?,
                        model.active_memory_breakdown(
                            &active_request.request_decoder_state,
                            active_request.additional_context_state_payload_bytes(),
                            mlx_active_memory_bytes,
                            0,
                        ),
                    )
                    .with_expert_residency_telemetry(model.expert_residency_telemetry()),
                )
            })
            .transpose()?;
        let generated_token = GeneratedToken::TokenId {
            token_id: generated_token_id,
            is_reasoning_token,
            expert_memory_mode: Some(model.expert_memory_mode()),
            mlx_memory_telemetry,
            first_decode_forward_elapsed_millis: active_request
                .first_decode_forward_elapsed_millis
                .take(),
            generation_finalization: None,
        };
        Ok(GeneratedTokenEmission {
            generated_token,
            is_terminal,
        })
    }
}

pub(super) fn synchronize_generated_token_id(
    active_request: &mut Qwen3_5EngineRequest,
    generated_token: &MlxArray,
) -> Result<u32, InferenceEngineError> {
    active_request
        .performance_attribution
        .measure_operation(
            PerformanceOperation::GeneratedTokenItemSynchronizationWait,
            |_performance_attribution| generated_token.item_u32(),
        )
        .map_err(qwen3_5_runtime_error)
}
