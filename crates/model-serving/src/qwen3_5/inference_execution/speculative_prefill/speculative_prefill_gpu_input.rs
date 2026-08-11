//! Creates the request-owned GPU token array used by sparse target gathering.
//!
//! Selection vectors contain absolute positions, not token identifiers. Sparse
//! execution needs both: it gathers selected identifiers from this complete
//! prompt array and separately forwards original positions for rotary encoding.
//! Building the complete token array once avoids one host-to-GPU upload per
//! target chunk.

#[cfg(feature = "direct-mlx")]
use super::super::Qwen3_5EngineState;
#[cfg(feature = "direct-mlx")]
use super::super::engine_request::Qwen3_5EngineRequest;
#[cfg(feature = "direct-mlx")]
use crate::{PerformanceOperation, Qwen3_5ExecutionError};

#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    /// Uploads the complete prompt's token identifiers as one `[1, token_count]`
    /// signed Int32 MLX array and attaches it to the active request.
    ///
    /// MLX indexing uses Int32 at this boundary. Every length and identifier is
    /// checked before allocation so a lossy cast can never select another token.
    pub(in crate::qwen3_5) fn prepare_speculative_prefill_prompt_token_indices_on_gpu(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<(), Qwen3_5ExecutionError> {
        // Attribute conversion and allocation as sparse input assembly rather
        // than hiding the cost inside the first sparse target forward.
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
                // A leading batch dimension keeps the array compatible with the
                // target embedding/gather path used by ordinary prefill.
                Ok::<_, Qwen3_5ExecutionError>(model.runtime().array_from_i32(
                    &signed_prompt_token_ids,
                    &[1, prompt_token_count_i32],
                )?)
            },
        )?;
        // Install only after complete successful construction. On failure the
        // request retains `None` and cannot accidentally execute stale input.
        active_request.speculative_prefill_prompt_token_indices = Some(prompt_token_indices_on_gpu);
        Ok(())
    }
}
