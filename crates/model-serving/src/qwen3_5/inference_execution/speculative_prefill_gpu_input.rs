#[cfg(feature = "direct-mlx")]
use super::Qwen3_5EngineState;
#[cfg(feature = "direct-mlx")]
use super::engine_request::Qwen3_5EngineRequest;
#[cfg(feature = "direct-mlx")]
use crate::{PerformanceCounter, PerformanceOperation, Qwen3_5ExecutionError};

#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    pub(super) fn prepare_speculative_prefill_prompt_token_indices_on_gpu(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let prompt_token_indices_on_gpu = active_request.performance_attribution.measure_operation(
            PerformanceOperation::SpeculativePrefillSparseInputAssembly,
            |_performance_attribution| {
                let model = self.model.as_ref().ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "Qwen3.5 target model is unavailable for sparse input assembly",
                })?;
                let prompt_token_count_i32 = i32::try_from(active_request.input_token_ids.len())
                    .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                        description: "speculative-prefill prompt token count exceeds the MLX range",
                    })?;
                let signed_prompt_token_ids = active_request
                    .input_token_ids
                    .iter()
                    .map(|prompt_token_id| {
                        i32::try_from(*prompt_token_id).map_err(|_| {
                            Qwen3_5ExecutionError::InvalidInput {
                                description: "speculative-prefill token ID exceeds the MLX int32 range",
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok::<_, Qwen3_5ExecutionError>(model.runtime().array_from_i32(
                    &signed_prompt_token_ids,
                    &[1, prompt_token_count_i32],
                )?)
            },
        )?;
        active_request.speculative_prefill_prompt_token_indices = Some(prompt_token_indices_on_gpu);
        Ok(())
    }

    pub(super) fn record_speculative_prefill_input_assembly_fallback(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
        speculative_prefill_input_assembly_error: impl std::fmt::Display,
    ) {
        active_request.should_use_speculative_prefill = false;
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::SpeculativePrefillFallbackCount, 1);
        tracing::warn!(
            request_id = active_request.request_id.value(),
            error = %speculative_prefill_input_assembly_error,
            "optional speculative-prefill GPU input assembly failed; continuing target-only"
        );
    }
}
