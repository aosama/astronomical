use astronomical_runtime_integration::{MlxArray, MlxMemorySnapshot};

use crate::{GeneratedToken, InferenceEngineError, PerformanceCounter, PerformanceOperation};

use super::engine_request::Qwen3_5MoEEngineRequest;
use super::{Qwen3_5MoEEngineState, fatal_engine_error, qwen3_5_moe_runtime_error};
use crate::qwen3_5_moe::Qwen3_5MoEModel;

pub(super) struct GeneratedTokenEmission {
    pub(super) generated_token: GeneratedToken,
    pub(super) is_terminal: bool,
}

impl Qwen3_5MoEEngineState {
    pub(super) fn generated_token_will_be_terminal(
        &self,
        active_request: &Qwen3_5MoEEngineRequest,
        generated_token_id: u32,
    ) -> bool {
        let emitted_token_id =
            self.preview_thinking_budget_token(active_request, generated_token_id);
        self.end_of_sequence_token_ids.contains(&emitted_token_id)
            || active_request.generated_token_count.saturating_add(1)
                >= active_request.maximum_output_tokens
    }

    pub(super) fn build_generated_token_emission(
        &self,
        model: &Qwen3_5MoEModel,
        active_request: &mut Qwen3_5MoEEngineRequest,
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
            active_request.is_inside_thinking && generated_token_id != self.think_end_token_id;
        let generated_token_id = self.apply_thinking_budget(active_request, generated_token_id)?;

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
                            active_request.mtp_request_state.as_ref(),
                            mlx_active_memory_bytes,
                        ),
                    ),
                )
            })
            .transpose()?;
        let generated_token = GeneratedToken::TokenId {
            token_id: generated_token_id,
            is_reasoning_token,
            expert_memory_mode: Some(model.expert_memory_mode()),
            mlx_memory_telemetry,
            generation_finalization: None,
        };
        Ok(GeneratedTokenEmission {
            generated_token,
            is_terminal,
        })
    }

    fn apply_thinking_budget(
        &self,
        active_request: &mut Qwen3_5MoEEngineRequest,
        generated_token_id: u32,
    ) -> Result<u32, InferenceEngineError> {
        // The next token may already have been computed from the model's actual
        // chosen token, so the KV cache can have one token of drift after this
        // override. MTP avoids starting new verification windows when this
        // boundary is imminent, preserving the existing target-only behavior.
        if active_request.is_inside_thinking && generated_token_id != self.think_end_token_id {
            if let Some(thinking_budget) = active_request.thinking_budget {
                active_request.thinking_token_count = active_request
                    .thinking_token_count
                    .checked_add(1)
                    .ok_or_else(|| fatal_engine_error("thinking-token counter overflowed"))?;
                if active_request.thinking_token_count >= thinking_budget {
                    active_request.is_inside_thinking = false;
                    return Ok(self.think_end_token_id);
                }
            }
            return Ok(generated_token_id);
        }
        if generated_token_id == self.think_end_token_id {
            active_request.is_inside_thinking = false;
        }
        Ok(generated_token_id)
    }

    fn preview_thinking_budget_token(
        &self,
        active_request: &Qwen3_5MoEEngineRequest,
        generated_token_id: u32,
    ) -> u32 {
        if active_request.is_inside_thinking
            && generated_token_id != self.think_end_token_id
            && active_request
                .thinking_budget
                .is_some_and(|thinking_budget| {
                    active_request.thinking_token_count.saturating_add(1) >= thinking_budget
                })
        {
            self.think_end_token_id
        } else {
            generated_token_id
        }
    }
}

pub(super) fn synchronize_generated_token_id(
    active_request: &mut Qwen3_5MoEEngineRequest,
    generated_token: &MlxArray,
) -> Result<u32, InferenceEngineError> {
    active_request
        .performance_attribution
        .measure_operation(
            PerformanceOperation::GeneratedTokenItemSynchronizationWait,
            |_performance_attribution| generated_token.item_u32(),
        )
        .map_err(qwen3_5_moe_runtime_error)
}

/// Returns whether an MTP window could cross the forced thinking boundary.
#[doc(hidden)]
#[must_use]
pub fn qwen3_5_moe_mtp_verification_may_cross_thinking_budget(
    is_inside_thinking: bool,
    thinking_token_count: u16,
    thinking_budget: Option<u16>,
    possible_emitted_token_count: u16,
) -> bool {
    is_inside_thinking
        && thinking_budget.is_some_and(|thinking_budget| {
            thinking_token_count.saturating_add(possible_emitted_token_count) >= thinking_budget
        })
}

/// Returns whether a depth-one MTP window fits output and context boundaries.
#[doc(hidden)]
#[must_use]
pub fn qwen3_5_moe_depth_one_mtp_window_fits(
    generated_token_count: u16,
    maximum_output_tokens: u16,
    next_position_tokens: u32,
    maximum_position_count: usize,
) -> bool {
    maximum_output_tokens.saturating_sub(generated_token_count) >= 2
        && maximum_position_count.saturating_sub(next_position_tokens as usize) >= 2
}
