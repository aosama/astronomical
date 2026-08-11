//! Restores and publishes dense drafter decoder state.
//!
//! Drafter state is separate from sparse target state. The drafter always scores
//! a logically dense prompt, so its persistent cache uses ordinary fixed-size
//! key/value blocks plus one recurrent boundary snapshot. Block identities form
//! a parent-linked chain over exact token content and ordered image digests.
//!
//! Publication is synchronous and fail-closed. A completed block advances the
//! parent key only after storage reports `Published` or `AlreadyPublished`; this
//! prevents a later child from naming a parent that is not durable.

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore,
    PersistentPromptCachePrefixLookup, PersistentPromptCachePublicationOutcome,
    Qwen3_5ExecutionError,
};

use super::super::super::RequestDecoderStateStack;
use super::super::super::model::{
    Qwen3_5Model, Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlock,
};
use super::super::Qwen3_5EngineState;
use super::super::engine_request::{
    Qwen3_5EngineRequest, Qwen3_5SpeculativePrefillFailureStageForTests,
};
use super::speculative_prefill_failure::configured_speculative_prefill_failure;

/// Successful persistent-prefix restoration metadata needed by the scoring
/// phase to continue forwarding and to append new child blocks.
pub(crate) struct SpeculativePrefillDraftPersistentPrefixRestoreOutcome {
    /// Number of leading prompt tokens represented by restored decoder state.
    pub(crate) restored_token_count: usize,
    /// Identity of the chain tail; the next captured block must use it as parent.
    pub(crate) last_restored_persistent_prompt_cache_block_key: PersistentPromptCacheBlockKey,
}

impl Qwen3_5EngineState {
    /// Builds the callback used by model scoring whenever one complete cache
    /// block reaches a durable publication boundary.
    ///
    /// The closure owns no state arrays. It borrows the request's latest parent
    /// key and failure flag so each successful callback extends one linear chain
    /// and any callback failure can later be attributed to persistence rather
    /// than scoring.
    pub(crate) fn speculative_prefill_draft_block_persistence_consumer<'a>(
        &'a self,
        draft_model: &'a Qwen3_5Model,
        complete_prompt_token_ids: &'a [u32],
        ordered_image_sha256_digests: &'a [[u8; 32]],
        latest_persisted_draft_block_key: &'a mut Option<PersistentPromptCacheBlockKey>,
        draft_prompt_state_persistence_failed: &'a mut bool,
    ) -> impl FnMut(
        Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlock,
        &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError>
    + 'a {
        move |persistent_prompt_cache_block, performance_attribution| match self
            .save_speculative_prefill_draft_persistent_prompt_cache_block(
                draft_model,
                complete_prompt_token_ids,
                ordered_image_sha256_digests,
                latest_persisted_draft_block_key.as_ref(),
                persistent_prompt_cache_block,
                performance_attribution,
            ) {
            Ok(persisted_draft_block_key) => {
                // Advance only after durable success (including idempotent
                // AlreadyPublished); the following block must name this key.
                *latest_persisted_draft_block_key = Some(persisted_draft_block_key);
                Ok(())
            }
            Err(draft_prompt_state_persistence_error) => {
                // The scoring API returns one execution error type. Preserve a
                // side-band cause so the request error names persistence exactly.
                *draft_prompt_state_persistence_failed = true;
                Err(draft_prompt_state_persistence_error)
            }
        }
    }

    /// Decides whether scoring should emit persistent blocks for this request.
    ///
    /// A fresh score can begin a root chain. A disk-restored score can append to
    /// its known tail. An in-memory checkpoint has no durable parent identity, so
    /// it must not publish children that disk restoration could never traverse.
    pub(crate) fn prepare_speculative_prefill_draft_cache_capture(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
        restored_draft_prefix_token_count: usize,
        draft_persistent_prefix_block_key: Option<&PersistentPromptCacheBlockKey>,
    ) -> Result<bool, InferenceEngineError> {
        let should_capture_persistent_prompt_cache_blocks = self
            .speculative_prefill_draft_persistent_prompt_cache
            .is_some()
            && (restored_draft_prefix_token_count == 0
                || draft_persistent_prefix_block_key.is_some());
        // Failure injection is consumed only when real production conditions
        // would attempt persistence, keeping tests faithful to reachable paths.
        if should_capture_persistent_prompt_cache_blocks
            && active_request.take_forced_speculative_prefill_failure_for_tests(
                Qwen3_5SpeculativePrefillFailureStageForTests::DrafterPromptStatePersistence,
            )
        {
            return Err(configured_speculative_prefill_failure(
                active_request.request_id,
                "drafter prompt-state persistence",
                "forced drafter prompt-state persistence failure",
            ));
        }
        Ok(should_capture_persistent_prompt_cache_blocks)
    }

    /// Restores the drafter's longest dense SSD prefix independently of target state.
    ///
    /// The scorer forwards the remaining suffix at the returned position and
    /// then scores the complete restored-plus-new prompt key range.
    pub(crate) fn restore_speculative_prefill_draft_persistent_prefix(
        &self,
        draft_model: &Qwen3_5Model,
        prompt_token_ids: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        draft_request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<SpeculativePrefillDraftPersistentPrefixRestoreOutcome>, Qwen3_5ExecutionError>
    {
        // No store means caching is disabled. This is an ordinary miss rather
        // than an execution error and performs no filesystem work.
        let Some(draft_persistent_prompt_cache) = self
            .speculative_prefill_draft_persistent_prompt_cache
            .as_ref()
        else {
            return Ok(None);
        };
        let persistent_prompt_cache_prefix_lookup_result = performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCachePrefixLookup,
                |_performance_attribution| {
                    PersistentPromptCachePrefixLookup::for_prompt_with_image_digests(
                        draft_persistent_prompt_cache.model_contract_ref(),
                        prompt_token_ids,
                        ordered_image_sha256_digests,
                        |block_hash| draft_persistent_prompt_cache.has_kv_block(block_hash),
                        |block_hash| {
                            draft_persistent_prompt_cache.has_recurrent_snapshot(block_hash)
                        },
                    )
                },
            );
        // Lookup validates topology and finds only complete block boundaries. A
        // zero result requires neither key/value reads nor recurrent restoration.
        let restored_token_count =
            persistent_prompt_cache_prefix_lookup_result.restored_token_count();
        if restored_token_count == 0 {
            return Ok(None);
        }

        let complete_block_count = restored_token_count
            / draft_persistent_prompt_cache
                .model_contract_ref()
                .block_token_count();
        let mut persistent_prompt_cache_kv_block_tensors = Vec::with_capacity(complete_block_count);
        let mut last_restored_persistent_prompt_cache_block_key = None;
        // Reconstruct keys and load blocks in ancestry order. Each child key is
        // derived from the previous key, so reordered or skipped blocks cannot
        // accidentally produce a valid decoder-state chain.
        for block_index in 0..complete_block_count {
            let persistent_prompt_cache_block_key = draft_prompt_cache_block_key(
                draft_persistent_prompt_cache,
                prompt_token_ids,
                ordered_image_sha256_digests,
                block_index,
                last_restored_persistent_prompt_cache_block_key.as_ref(),
            )?;
            let loaded_kv_block_tensors = performance_attribution
                .measure_operation(
                    PerformanceOperation::PersistentPromptCacheKvBlockRead,
                    |performance_attribution| {
                        draft_persistent_prompt_cache.load_kv_block(
                            draft_model.runtime(),
                            &persistent_prompt_cache_block_key,
                            performance_attribution.positional_file_read_metrics(),
                        )
                    },
                )?
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    // Files can disappear after lookup (for example external
                    // cache deletion). Treat the race as an invalid restore,
                    // never as a partial prefix.
                    description: "speculative-prefill drafter KV block disappeared during restore",
                })?;
            persistent_prompt_cache_kv_block_tensors.push(loaded_kv_block_tensors);
            last_restored_persistent_prompt_cache_block_key =
                Some(persistent_prompt_cache_block_key);
        }

        let recurrent_snapshot_block_key = last_restored_persistent_prompt_cache_block_key
            .as_ref()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill drafter restore lost its boundary key",
            })?;
        let persistent_prompt_cache_recurrent_snapshot_tensors = performance_attribution
            .measure_operation(
                PerformanceOperation::PersistentPromptCacheRecurrentSnapshotRead,
                |performance_attribution| {
                    draft_persistent_prompt_cache.load_recurrent_snapshot(
                        draft_model.runtime(),
                        recurrent_snapshot_block_key,
                        performance_attribution.positional_file_read_metrics(),
                    )
                },
            )?
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description:
                    "speculative-prefill drafter recurrent snapshot disappeared during restore",
            })?;

        // Key/value blocks rebuild attention history; the final recurrent
        // snapshot rebuilds non-attention state at exactly the same boundary.
        draft_request_decoder_state.restore_from_persistent_prompt_cache_blocks(
            draft_model.runtime(),
            &persistent_prompt_cache_kv_block_tensors,
            &persistent_prompt_cache_recurrent_snapshot_tensors,
        )?;
        draft_request_decoder_state
            .materialize_restored_persistent_prompt_cache_state(draft_model.runtime())?;
        // Return the chain tail as well as token count because later synchronous
        // captures must append to this exact durable ancestry.
        let last_restored_persistent_prompt_cache_block_key =
            last_restored_persistent_prompt_cache_block_key.ok_or(
                Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill drafter restore lost its final block key",
                },
            )?;
        Ok(Some(
            SpeculativePrefillDraftPersistentPrefixRestoreOutcome {
                restored_token_count,
                last_restored_persistent_prompt_cache_block_key,
            },
        ))
    }

    /// Durably publishes one dense drafter state block completed during draft prompt scoring.
    pub(crate) fn save_speculative_prefill_draft_persistent_prompt_cache_block(
        &self,
        draft_model: &Qwen3_5Model,
        prompt_token_ids: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
        persistent_prompt_cache_block: Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlock,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PersistentPromptCacheBlockKey, Qwen3_5ExecutionError> {
        // Reaching this method without a store indicates an orchestration defect:
        // capture eligibility should have disabled the callback entirely.
        let Some(draft_persistent_prompt_cache) = self
            .speculative_prefill_draft_persistent_prompt_cache
            .as_ref()
        else {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill drafter cache is unavailable for block persistence",
            });
        };

        let mut previous_persistent_prompt_cache_block_key =
            parent_persistent_prompt_cache_block_key.cloned();
        // Validate model-emitted boundaries before slicing request tokens. Every
        // durable block must be complete, in range, and end exactly where its
        // declared fixed-size interval says it ends.
        let block_token_count = persistent_prompt_cache_block
            .block_end_tokens
            .saturating_sub(persistent_prompt_cache_block.block_start_tokens);
        let expected_block_token_count = draft_persistent_prompt_cache
            .model_contract_ref()
            .block_token_count();
        if block_token_count != expected_block_token_count
            || persistent_prompt_cache_block.block_end_tokens > prompt_token_ids.len()
            || persistent_prompt_cache_block.block_start_tokens
                != persistent_prompt_cache_block
                    .block_end_tokens
                    .saturating_sub(expected_block_token_count)
        {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill drafter cache block range is invalid",
            });
        }
        if persistent_prompt_cache_block.block_start_tokens > 0
            && previous_persistent_prompt_cache_block_key.is_none()
        {
            // A non-root block without a parent would create an unreachable
            // orphan and break exact ancestry validation during lookup.
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill drafter cache block has no parent",
            });
        }
        let persistent_prompt_cache_block_key =
            match previous_persistent_prompt_cache_block_key.as_ref() {
                // Child identities commit to parent ancestry and this block's tokens.
                Some(parent_persistent_prompt_cache_block_key) => {
                    parent_persistent_prompt_cache_block_key.for_child_block(
                        &prompt_token_ids[persistent_prompt_cache_block.block_start_tokens
                            ..persistent_prompt_cache_block.block_end_tokens],
                    )
                }
                // Root identities additionally commit to the model storage
                // contract and ordered images associated with the prompt.
                None => PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
                    draft_persistent_prompt_cache.model_contract_ref(),
                    &prompt_token_ids[persistent_prompt_cache_block.block_start_tokens
                        ..persistent_prompt_cache_block.block_end_tokens],
                    ordered_image_sha256_digests,
                ),
            };
        let persistent_prompt_cache_block_key = match persistent_prompt_cache_block_key {
            Ok(persistent_prompt_cache_block_key) => persistent_prompt_cache_block_key,
            Err(_block_key_error) => {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill drafter cache block identity failed",
                });
            }
        };
        let mut block_performance_attribution =
            std::mem::replace(performance_attribution, PerformanceAttribution::disabled());
        // Captured arrays are lazy MLX values. Synchronization materializes all
        // producer work before the disk writer reads bytes, and allocator cleanup
        // creates room for the direct-publication workspace.
        draft_model
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()?;
        let publication_outcome = draft_persistent_prompt_cache
            .publish_block_with_performance_attribution(
                draft_model.runtime(),
                &persistent_prompt_cache_block_key,
                previous_persistent_prompt_cache_block_key.as_ref(),
                &persistent_prompt_cache_block.kv_block_tensors,
                &persistent_prompt_cache_block.recurrent_snapshot_tensors,
                &mut block_performance_attribution,
            );
        *performance_attribution = block_performance_attribution;
        // Active-memory pressure is the only retryable publication failure. The
        // same immutable captured arrays are retried once after reclaiming
        // pageable experts from both models; storage or validation failures are
        // deterministic and return immediately below.
        let publication_outcome = match publication_outcome {
            Err(publication_error) if publication_error.active_memory_deficit_bytes().is_some() => {
                let active_memory_deficit_bytes =
                    publication_error.active_memory_deficit_bytes().unwrap_or(0);
                performance_attribution.measure_operation(
                    crate::PerformanceOperation::NativeExpertCacheReclamation,
                    |_performance_attribution| {
                        if let Some(target_model) = self.model.as_ref() {
                            // Target and drafter share the process-wide MLX
                            // ceiling, so either owner's retained expert pages
                            // may be the bytes preventing publication.
                            target_model.limit_expert_retention_for_request_memory_pressure(
                                active_memory_deficit_bytes,
                            )?;
                        }
                        draft_model.limit_expert_retention_for_request_memory_pressure(
                            active_memory_deficit_bytes,
                        )?;
                        draft_model
                            .runtime()
                            .synchronize_gpu_stream_and_clear_allocator_cache()
                            .map_err(Qwen3_5ExecutionError::from)
                    },
                )?;
                let mut retry_performance_attribution =
                    std::mem::replace(performance_attribution, PerformanceAttribution::disabled());
                let retry_outcome = draft_persistent_prompt_cache
                    .publish_block_with_performance_attribution(
                        draft_model.runtime(),
                        &persistent_prompt_cache_block_key,
                        previous_persistent_prompt_cache_block_key.as_ref(),
                        &persistent_prompt_cache_block.kv_block_tensors,
                        &persistent_prompt_cache_block.recurrent_snapshot_tensors,
                        &mut retry_performance_attribution,
                    );
                *performance_attribution = retry_performance_attribution;
                retry_outcome
            }
            publication_outcome => publication_outcome,
        };
        match publication_outcome {
            Ok(PersistentPromptCachePublicationOutcome::Published)
            | Ok(PersistentPromptCachePublicationOutcome::AlreadyPublished) => {
                // Idempotent pre-existence is as durable as a fresh write and is
                // therefore a valid parent for the next block.
                previous_persistent_prompt_cache_block_key =
                    Some(persistent_prompt_cache_block_key);
            }
            Err(write_error) => {
                tracing::error!(
                    error = %write_error,
                    "configured SpecPrefill drafter cache block write failed"
                );
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill drafter cache block write failed",
                });
            }
        }
        previous_persistent_prompt_cache_block_key.ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill drafter cache block persistence produced no identity",
        })
    }
}

fn draft_prompt_cache_block_key(
    draft_persistent_prompt_cache: &PersistentPromptCacheDiskStore,
    draft_prompt_prefix_token_ids: &[u32],
    ordered_image_sha256_digests: &[[u8; 32]],
    block_index: usize,
    parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
) -> Result<PersistentPromptCacheBlockKey, Qwen3_5ExecutionError> {
    // Derive byte-independent token boundaries with checked arithmetic before
    // constructing an identity. A malformed count must not wrap into a valid
    // but unrelated token slice.
    let persistent_prompt_cache_block_token_count = draft_persistent_prompt_cache
        .model_contract_ref()
        .block_token_count();
    let block_start_tokens = block_index
        .checked_mul(persistent_prompt_cache_block_token_count)
        .ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill drafter cache block start overflowed",
        })?;
    let block_end_tokens = block_start_tokens
        .checked_add(persistent_prompt_cache_block_token_count)
        .ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill drafter cache block end overflowed",
        })?;
    let block_tokens = draft_prompt_prefix_token_ids
        .get(block_start_tokens..block_end_tokens)
        .ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill drafter cache block exceeds the prefix",
        })?;
    match parent_persistent_prompt_cache_block_key {
        // Every block after index zero extends the previously validated chain.
        Some(parent_persistent_prompt_cache_block_key) => parent_persistent_prompt_cache_block_key
            .for_child_block(block_tokens)
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill drafter child cache block identity failed",
            }),
        // Only the first block has no parent; image order participates in its
        // identity so visually different prompts cannot share decoder state.
        None => PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
            draft_persistent_prompt_cache.model_contract_ref(),
            block_tokens,
            ordered_image_sha256_digests,
        )
        .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill drafter root cache block identity failed",
        }),
    }
}
