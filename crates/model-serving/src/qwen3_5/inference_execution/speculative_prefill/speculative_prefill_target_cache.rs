//! Persists and restores selection-bound sparse target decoder state.
//!
//! Ordinary persistent prompt state represents every dense token in a prefix.
//! SpecPrefill target state instead represents an ordered subset of original
//! prompt positions. Correct restoration therefore requires both decoder tensors
//! and the exact UInt32 position vector that produced them, under a contract that
//! binds target revision, drafter revision, tokenizer mapping, and selection policy.

use astronomical_runtime_integration::{MlxArray, MlxDtype};

use crate::{
    PerformanceAttribution, PerformanceCounter, PerformanceOperation, Qwen3_5ExecutionError,
    RequestDecoderStateStack,
};

use super::super::{
    Qwen3_5EngineRequest, Qwen3_5EngineState, Qwen3_5SpeculativePrefillFailureStageForTests,
};

/// Metadata accompanying a successfully reconstructed sparse target prefix.
pub(crate) struct RestoredSpeculativePrefillTargetPrefix {
    /// Logical prompt prefix covered, which advances the ordinary prefill cursor.
    pub(crate) prompt_prefix_token_count: usize,
    /// Actual target rows represented by decoder state, retained for later persistence/accounting.
    pub(crate) selected_target_token_positions: MlxArray,
}

impl Qwen3_5EngineState {
    /// Restores the longest exact sparse target prefix that improves upon any
    /// ordinary prompt-cache prefix already restored for this request.
    pub(crate) fn restore_longest_speculative_prefill_target_prefix(
        &self,
        prompt_token_ids: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        existing_restored_prompt_token_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<
        Option<(
            RequestDecoderStateStack,
            RestoredSpeculativePrefillTargetPrefix,
        )>,
        Qwen3_5ExecutionError,
    > {
        // Either missing storage or an incomplete policy identity is an ordinary
        // miss. No disk lookup is possible without both.
        let Some(target_persistent_prompt_cache) = self.persistent_prompt_cache.as_ref() else {
            return Ok(None);
        };
        let Some(target_state_contract) = self.speculative_prefill_target_state_contract() else {
            return Ok(None);
        };
        let target_model = self
            .model
            .as_ref()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "target model is unavailable during speculative-prefill state restore",
            })?;
        let restored_target_state = performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCachePrefixLookup,
            |performance_attribution| {
                target_persistent_prompt_cache.load_longest_speculative_prefill_target_state(
                    target_model.runtime(),
                    &target_state_contract,
                    prompt_token_ids,
                    ordered_image_sha256_digests,
                    performance_attribution.positional_file_read_metrics(),
                )
            },
        )?;
        let Some(restored_target_state) = restored_target_state else {
            return Ok(None);
        };
        let (prompt_prefix_token_count, selected_target_token_positions, decoder_state_tensors) =
            restored_target_state.into_parts();
        // Prefer existing ordinary state when sparse state advances no farther.
        // Position dtype/rank are semantic contracts required by reconstruction.
        if prompt_prefix_token_count <= existing_restored_prompt_token_count
            || selected_target_token_positions.dtype() != MlxDtype::UInt32
            || selected_target_token_positions.shape().len() != 1
        {
            return Ok(None);
        }
        let mut restored_request_decoder_state =
            RequestDecoderStateStack::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                target_model.decoder_cache_layout(),
                self.full_attention_kv_state_growth_tokens,
            )?;
        // Reconstruct into a fresh stack so failure cannot partially overwrite
        // the request's existing ordinary restored state.
        performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCacheStateReconstruction,
            |_performance_attribution| {
                restored_request_decoder_state.restore_speculative_prefill_target_state_tensors(
                    &decoder_state_tensors,
                    selected_target_token_positions.shape()[0].max(0) as usize,
                )
            },
        )?;
        // MLX restoration is lazy. Materialize before replacing request state so
        // missing/invalid arrays fail inside the measured restore boundary.
        performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCacheStateMaterializationSynchronizationWait,
            |_performance_attribution| {
                restored_request_decoder_state
                    .materialize_restored_persistent_prompt_cache_state(target_model.runtime())
            },
        )?;
        performance_attribution.record_counter(
            PerformanceCounter::SpeculativePrefillTargetPersistentStateRestoredTokenCount,
            selected_target_token_positions.shape()[0].max(0) as u64,
        );
        Ok(Some((
            restored_request_decoder_state,
            RestoredSpeculativePrefillTargetPrefix {
                prompt_prefix_token_count,
                selected_target_token_positions,
            },
        )))
    }

    /// Saves the active request's sparse target prefix before the final
    /// generation-kickoff token is forwarded.
    pub(in crate::qwen3_5) fn save_speculative_prefill_target_prefix(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Result<(), crate::InferenceEngineError> {
        // Ordinary requests own no selection-bound state.
        if !active_request.should_use_speculative_prefill {
            return Ok(());
        }
        let (Some(target_persistent_prompt_cache), Some(target_model)) =
            (self.persistent_prompt_cache.as_ref(), self.model.as_ref())
        else {
            // Disk caching is optional. With no configured store there is no
            // persistence promise to enforce.
            return Ok(());
        };
        let Some(target_state_contract) = self.speculative_prefill_target_state_contract() else {
            // A configured store plus active SpecPrefill requires a complete
            // identity; persisting under a partial identity is forbidden.
            return Err(
                super::speculative_prefill_failure::configured_speculative_prefill_failure(
                    active_request.request_id,
                    "sparse target-state persistence planning",
                    "the target-state contract is unavailable",
                ),
            );
        };
        if active_request.take_forced_speculative_prefill_failure_for_tests(
            Qwen3_5SpeculativePrefillFailureStageForTests::SparseTargetStatePersistence,
        ) {
            return Err(
                super::speculative_prefill_failure::configured_speculative_prefill_failure(
                    active_request.request_id,
                    "sparse target-state persistence",
                    "forced sparse target-state persistence failure",
                ),
            );
        }
        let save_outcome = active_request.measure_operation_with_request(
            PerformanceOperation::PersistentPromptCacheStateExtraction,
            |active_request| -> Result<(), Qwen3_5ExecutionError> {
                // Extract state before constructing position metadata so the two
                // payloads describe one unchanged decoder boundary.
                let decoder_state_tensors = active_request
                    .request_decoder_state
                    .extract_speculative_prefill_target_state_tensors(target_model.runtime())?;
                let selected_target_token_positions = self
                    .persisted_speculative_prefill_target_position_tensor(
                        target_model,
                        active_request,
                    )?;
                let final_generation_kickoff_position =
                    active_request.input_token_ids.len().checked_sub(1).ok_or(
                        Qwen3_5ExecutionError::InvalidInput {
                            description: "generation prompt must not be empty",
                        },
                    )?;
                // Persist only the prefix already represented by decoder state.
                // The final token is forwarded exactly once by generation startup.
                let named_decoder_state_tensors = decoder_state_tensors
                    .iter()
                    .map(|(tensor_name, target_state_tensor)| {
                        (tensor_name.as_str(), target_state_tensor)
                    })
                    .collect::<Vec<_>>();
                target_persistent_prompt_cache.save_speculative_prefill_target_state(
                    target_model.runtime(),
                    &target_state_contract,
                    &active_request.input_token_ids[..final_generation_kickoff_position],
                    &active_request.ordered_image_sha256_digests,
                    &selected_target_token_positions,
                    &named_decoder_state_tensors,
                )?;
                Ok(())
            },
        );
        match save_outcome {
            Ok(()) => {
                active_request.performance_attribution.record_counter(
                    PerformanceCounter::SpeculativePrefillTargetPersistentStateWriteCount,
                    1,
                );
                Ok(())
            }
            Err(target_state_save_error) => Err(
                super::speculative_prefill_failure::configured_speculative_prefill_failure(
                    active_request.request_id,
                    "sparse target-state persistence",
                    target_state_save_error,
                ),
            ),
        }
    }

    /// Reconstructs the ordered absolute-position vector represented by current target state.
    ///
    /// Ordering mirrors execution history: previously restored rows, newly
    /// processed dense control rows, then newly selected sparse conversation rows.
    /// Concatenation stays on GPU for direct persistent-state serialization.
    fn persisted_speculative_prefill_target_position_tensor(
        &self,
        target_model: &crate::Qwen3_5Model,
        active_request: &Qwen3_5EngineRequest,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        // Dense control rows are absolute positions from zero up to the boundary.
        let dense_target_prefix_positions = (0..active_request
            .speculative_prefill_dense_target_prefix_token_count)
            .map(|dense_target_prefix_position| {
                u32::try_from(dense_target_prefix_position).map_err(|_| {
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "dense target prefix position exceeds u32",
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dense_target_prefix_position_tensor = target_model.runtime().array_from_u32(
            &dense_target_prefix_positions,
            &[
                i32::try_from(dense_target_prefix_positions.len()).map_err(|_| {
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "dense target prefix position count exceeds i32",
                    }
                })?,
            ],
        )?;
        let current_selected_positions = active_request
            .speculative_prefill_selected_token_positions
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|selected_prompt_position| {
                u32::try_from(*selected_prompt_position).map_err(|_| {
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "selected target prompt position exceeds u32",
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let current_selected_position_tensor = target_model.runtime().array_from_u32(
            &current_selected_positions,
            &[
                i32::try_from(current_selected_positions.len()).map_err(|_| {
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "selected target row count exceeds i32",
                    }
                })?,
            ],
        )?;
        let mut selected_position_tensors = Vec::with_capacity(2);
        // A restored sparse prefix already includes any historical dense rows;
        // append it first, then only work performed by this request.
        if let Some(restored_target_positions) = active_request
            .speculative_prefill_restored_target_token_positions
            .as_ref()
        {
            selected_position_tensors.push(restored_target_positions);
        }
        if !dense_target_prefix_positions.is_empty() {
            selected_position_tensors.push(&dense_target_prefix_position_tensor);
        }
        if !current_selected_positions.is_empty() {
            selected_position_tensors.push(&current_selected_position_tensor);
        }
        target_model
            .runtime()
            .concatenate_axis(&selected_position_tensors, 0)
            .map_err(Qwen3_5ExecutionError::from)
    }
}
