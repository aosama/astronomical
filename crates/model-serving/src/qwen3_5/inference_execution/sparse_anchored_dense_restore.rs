//! Restores the sparse-anchored dense tail chain on top of a restored SpecPrefill prefix.
//!
//! The ordinary dense restore in `decoder_state_reuse` assumes slab rows align with prompt
//! positions. A restored SpecPrefill sparse prefix is compact — its slab holds only the
//! selected rows — so the densely processed conversation tail that follows it must come
//! back from the anchor-relative chain published by `sparse_anchored_dense_capture`
//! (issue #659). This module owns that restore: chain lookup through the store's in-memory
//! index, one destination growth to the final row count, per-block absorption at slab
//! offsets, and the newest boundary snapshot.

use astronomical_ipc_protocol::RequestId;

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockCausalInput, PersistentPromptCacheBlockKey,
};

use super::super::RequestDecoderStateStack;
use super::sparse_anchored_dense_capture::SparseAnchoredDenseCaptureContext;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

/// What a request should do with the dense tail following its restored sparse prefix.
pub(in crate::qwen3_5) enum SparseAnchoredDenseTailOutcome {
    /// A published anchored chain restored: the cursor advances and the context carries
    /// the newest restored block key so later dense chunks can extend the same chain.
    Restored {
        restored_tail_token_count: usize,
        last_anchored_block_key: PersistentPromptCacheBlockKey,
        capture_context: SparseAnchoredDenseCaptureContext,
    },
    /// No published chain exists yet: the request plans to publish the first blocks.
    Planned(SparseAnchoredDenseCaptureContext),
    /// The request cannot capture an anchored tail (visual causal input or no store);
    /// the caller must keep dense capture disabled, as before issue #659.
    CaptureDisabled,
}

impl Qwen3_5EngineState {
    /// Plans or restores the sparse-anchored dense tail for one request (issue #659).
    ///
    /// Called right after the sparse target state restored. Text requests get a capture
    /// context so their dense tail publishes on the anchored chain, plus any tail that
    /// previous requests already published. Visual requests return `CaptureDisabled`
    /// because their causal-input plan cannot describe anchor-relative blocks.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::qwen3_5) fn restore_or_plan_sparse_anchored_dense_tail(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        persistent_prompt_cache_block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
        anchor_prompt_token_count: usize,
        anchor_compact_row_count: usize,
        total_context_tokens: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<SparseAnchoredDenseTailOutcome, crate::Qwen3_5ExecutionError> {
        // The sparse prefix is compact: its slab rows are selection-bound, so the dense
        // tail cannot extend the ordinary token-aligned chain. Visual requests keep
        // capture disabled because their causal-input plan is indexed from prompt
        // position zero and cannot describe anchored blocks.
        if anchor_prompt_token_count == 0
            || !persistent_prompt_cache_block_causal_inputs.is_empty()
            || self.persistent_prompt_cache.is_none()
        {
            return Ok(SparseAnchoredDenseTailOutcome::CaptureDisabled);
        }
        let Some(target_state_contract) = self.speculative_prefill_target_state_contract() else {
            return Ok(SparseAnchoredDenseTailOutcome::CaptureDisabled);
        };
        let sparse_target_state_identity = target_state_contract.target_state_identity_hash(
            &prompt_token_ids[..anchor_prompt_token_count],
            ordered_image_sha256_digests,
        );
        let mut capture_context = SparseAnchoredDenseCaptureContext::new(
            sparse_target_state_identity,
            anchor_prompt_token_count,
            anchor_compact_row_count,
        );
        let restored_tail = self
            .restore_sparse_anchored_dense_tail_prefix(
                request_id,
                prompt_token_ids,
                sparse_target_state_identity,
                anchor_prompt_token_count,
                anchor_compact_row_count,
                total_context_tokens,
                request_decoder_state,
                performance_attribution,
            )
            .map_err(qwen3_5_runtime_error)?;
        match restored_tail {
            Some((restored_tail_token_count, last_anchored_block_key)) => {
                capture_context.last_published_block_key = Some(last_anchored_block_key);
                Ok(SparseAnchoredDenseTailOutcome::Restored {
                    restored_tail_token_count,
                    last_anchored_block_key: capture_context
                        .last_published_block_key
                        .clone()
                        .ok_or_else(|| {
                            fatal_engine_error(
                                "sparse-anchored dense tail restore lost its newest block key",
                            )
                        })?,
                    capture_context,
                })
            }
            None => Ok(SparseAnchoredDenseTailOutcome::Planned(capture_context)),
        }
    }

    /// Restores the sparse-anchored dense tail chain on top of a restored SpecPrefill prefix.
    ///
    /// Returns the number of tail prompt tokens restored and the newest anchored block key.
    /// A miss is ordinary: an anchor without a complete published tail simply continues
    /// densely, and the first such request publishes the chain.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore_sparse_anchored_dense_tail_prefix(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: &[u32],
        sparse_target_state_identity: [u8; 32],
        anchor_prompt_token_count: usize,
        anchor_compact_row_count: usize,
        total_context_tokens: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<(usize, PersistentPromptCacheBlockKey)>, InferenceEngineError> {
        let Some(persistent_prompt_cache) = self.persistent_prompt_cache.as_ref() else {
            return Ok(None);
        };
        let persistent_prompt_cache_block_token_count =
            persistent_prompt_cache.model_contract.block_token_count();
        let empty_block_causal_input = PersistentPromptCacheBlockCausalInput::empty();
        // The anchored chain is text-only: image causal input is planned from prompt
        // position zero, which does not describe anchor-relative blocks.
        if !prompt_token_ids
            .len()
            .checked_sub(anchor_prompt_token_count)
            .is_some_and(|anchor_relative_token_count| {
                anchor_relative_token_count >= persistent_prompt_cache_block_token_count
            })
        {
            return Ok(None);
        }
        // Phase A runs on an immutable engine borrow: walk the anchored chain through the
        // store's in-memory index and collect the workspace sizes the loads will need. The
        // borrow ends here so memory admission can demote resident experts before any
        // block tensor is materialized.
        let (anchored_block_keys, persistent_prompt_cache_restore_temporary_workspace_bytes) = {
            let mut anchored_block_keys = Vec::new();
            let mut candidate_block_key =
                PersistentPromptCacheBlockKey::for_sparse_anchored_root_block_with_causal_input(
                    &persistent_prompt_cache.model_contract,
                    &sparse_target_state_identity,
                    &prompt_token_ids[anchor_prompt_token_count
                        ..anchor_prompt_token_count + persistent_prompt_cache_block_token_count],
                    &empty_block_causal_input,
                )
                .map_err(|block_key_error| {
                    fatal_engine_error(format!(
                        "sparse-anchored dense tail root identity failed: {block_key_error}"
                    ))
                })?;
            // Push every block whose identity exists, then build the next candidate only
            // while its prompt range still fits. Dropping a verified block here would
            // silently restore one block less than the store can supply.
            loop {
                if !persistent_prompt_cache.has_kv_block(&candidate_block_key.block_hash()) {
                    break;
                }
                anchored_block_keys.push(candidate_block_key.clone());
                let next_block_start = anchor_prompt_token_count
                    + anchored_block_keys.len() * persistent_prompt_cache_block_token_count;
                let next_block_end = next_block_start + persistent_prompt_cache_block_token_count;
                if next_block_end > prompt_token_ids.len() {
                    break;
                }
                candidate_block_key = candidate_block_key
                    .for_child_block(&prompt_token_ids[next_block_start..next_block_end])
                    .map_err(|block_key_error| {
                        fatal_engine_error(format!(
                            "sparse-anchored dense tail child identity failed: {block_key_error}"
                        ))
                    })?;
            }
            if anchored_block_keys.is_empty() {
                return Ok(None);
            }
            // Hybrid models restore only when the newest kept block carries a boundary
            // snapshot, mirroring the ordinary chain's strict restorability rule.
            let Some(newest_snapshot_block_index) = anchored_block_keys
                .iter()
                .enumerate()
                .rev()
                .find(|(_block_index, anchored_block_key)| {
                    persistent_prompt_cache.has_recurrent_snapshot(&anchored_block_key.block_hash())
                })
                .map(|(block_index, _anchored_block_key)| block_index)
            else {
                return Ok(None);
            };
            anchored_block_keys.truncate(newest_snapshot_block_index + 1);
            let persistent_prompt_cache_restore_temporary_workspace_bytes = anchored_block_keys
                .iter()
                .map(|anchored_block_key| {
                    persistent_prompt_cache
                        .sequence_state_block_file_size_bytes(&anchored_block_key.block_hash())
                        .map_or(0, |file_size_bytes| {
                            usize::try_from(file_size_bytes).unwrap_or(usize::MAX)
                        })
                })
                .fold(0_usize, usize::saturating_add);
            (
                anchored_block_keys,
                persistent_prompt_cache_restore_temporary_workspace_bytes,
            )
        };
        let restored_tail_token_count =
            anchored_block_keys.len() * persistent_prompt_cache_block_token_count;
        // Admission mirrors the ordinary restore: charge the temporary workspace for the
        // loaded blocks against the context that is still uncached after the anchor.
        let additional_maximum_expert_page_reservation_bytes =
            self.speculative_prefill_draft_maximum_expert_page_reservation_bytes();
        self.validate_context_memory_admission_with_resident_expert_demotion(
            total_context_tokens.saturating_sub(anchor_prompt_token_count),
            persistent_prompt_cache_restore_temporary_workspace_bytes,
            additional_maximum_expert_page_reservation_bytes,
            performance_attribution,
        )?;
        // Fresh borrows after admission: the earlier immutable borrows ended with the
        // walk phase, so expert demotion could freely mutate engine state.
        let persistent_prompt_cache = self.persistent_prompt_cache.as_ref().ok_or_else(|| {
            fatal_engine_error("Qwen3.5 engine lost its persistent prompt-cache owner")
        })?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let final_compact_row_count = anchor_compact_row_count + restored_tail_token_count;
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheStateReconstruction,
                |_performance_attribution| {
                    request_decoder_state.grow_persistent_prompt_cache_kv_restore_destination(
                        model.runtime(),
                        final_compact_row_count,
                    )
                },
            )
            .map_err(|persistent_prompt_cache_error| {
                fatal_engine_error(format!(
                    "failed to grow the sparse-anchored dense tail destination: \
                     {persistent_prompt_cache_error}"
                ))
            })?;
        for (block_index, anchored_block_key) in anchored_block_keys.iter().enumerate() {
            let mut loaded_kv_block_tensors = performance_attribution
                .measure_operation(
                    PerformanceOperation::PersistentPromptCacheKvBlockRead,
                    |performance_attribution| {
                        persistent_prompt_cache.load_kv_block(
                            model.runtime(),
                            anchored_block_key,
                            performance_attribution.positional_file_read_metrics(),
                        )
                    },
                )
                .map_err(|persistent_prompt_cache_error| {
                    fatal_engine_error(format!(
                        "failed to load sparse-anchored dense tail block {block_index}: \
                         {persistent_prompt_cache_error}"
                    ))
                })?
                .ok_or_else(|| {
                    fatal_engine_error(
                        "sparse-anchored dense tail block was reported as present \
                         but load returned None",
                    )
                })?;
            let sequence_start_tokens =
                anchor_compact_row_count + block_index * persistent_prompt_cache_block_token_count;
            performance_attribution
                .measure_operation(
                    PerformanceOperation::PersistentPromptCacheStateReconstruction,
                    |_performance_attribution| {
                        request_decoder_state.absorb_persistent_prompt_cache_kv_block(
                            model.runtime(),
                            &mut loaded_kv_block_tensors,
                            sequence_start_tokens,
                            final_compact_row_count,
                        )
                    },
                )
                .map_err(|persistent_prompt_cache_error| {
                    fatal_engine_error(format!(
                        "failed to absorb sparse-anchored dense tail block {block_index}: \
                         {persistent_prompt_cache_error}"
                    ))
                })?;
            drop(loaded_kv_block_tensors);
        }
        let newest_anchored_block_key = &anchored_block_keys[anchored_block_keys.len() - 1];
        let mut persistent_prompt_cache_recurrent_snapshot_tensors = performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheRecurrentSnapshotRead,
                |performance_attribution| {
                    persistent_prompt_cache.load_recurrent_snapshot(
                        model.runtime(),
                        newest_anchored_block_key,
                        performance_attribution.positional_file_read_metrics(),
                    )
                },
            )
            .map_err(|persistent_prompt_cache_error| {
                fatal_engine_error(format!(
                    "failed to load the sparse-anchored dense tail snapshot: \
                     {persistent_prompt_cache_error}"
                ))
            })?
            .ok_or_else(|| {
                fatal_engine_error(
                    "sparse-anchored dense tail snapshot was reported as present \
                     but load returned None",
                )
            })?;
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheStateReconstruction,
                |_performance_attribution| {
                    request_decoder_state.absorb_persistent_prompt_cache_recurrent_snapshot(
                        model.runtime(),
                        &mut persistent_prompt_cache_recurrent_snapshot_tensors,
                    )
                },
            )
            .map_err(|persistent_prompt_cache_error| {
                fatal_engine_error(format!(
                    "failed to absorb the sparse-anchored dense tail snapshot: \
                     {persistent_prompt_cache_error}"
                ))
            })?;
        drop(persistent_prompt_cache_recurrent_snapshot_tensors);
        performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheStateMaterializationSynchronizationWait,
                |_performance_attribution| {
                    request_decoder_state
                        .materialize_restored_persistent_prompt_cache_state(model.runtime())
                },
            )
            .map_err(|persistent_prompt_cache_error| {
                fatal_engine_error(format!(
                    "failed to materialize the sparse-anchored dense tail restore: \
                     {persistent_prompt_cache_error}"
                ))
            })?;
        performance_attribution
            .measure_operation(
                PerformanceOperation::MlxAllocatorCacheCleanup,
                |_performance_attribution| model.runtime().clear_allocator_cache(),
            )
            .map_err(|runtime_error| {
                fatal_engine_error(format!(
                    "failed to clear allocator memory after the sparse-anchored dense tail \
                     restore: {runtime_error}"
                ))
            })?;
        model.resume_expert_retention_after_request_memory_pressure();
        tracing::info!(
            request_id = request_id.value(),
            anchor_prompt_token_count,
            restored_tail_token_count,
            restored_tail_block_count = anchored_block_keys.len(),
            "sparse-anchored dense tail restored from disk"
        );
        self.persistent_prompt_cache_counters
            .record_cache_hit(restored_tail_token_count);
        Ok(Some((
            restored_tail_token_count,
            newest_anchored_block_key.clone(),
        )))
    }
}
