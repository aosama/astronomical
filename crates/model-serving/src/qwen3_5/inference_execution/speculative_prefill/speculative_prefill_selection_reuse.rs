//! Reuses an exact prior selection before loading the request-scoped drafter.
//!
//! Lookup order is intentionally cheapest-first: worker memory, then persistent
//! storage, then fresh scoring in the caller. A hit must still prepare the current
//! request's GPU prompt-token array because that array is not part of selection
//! storage and may not outlive the request that created the selection.

#[cfg(feature = "direct-mlx")]
use crate::{PerformanceCounter, PerformanceOperation};

#[cfg(feature = "direct-mlx")]
use super::super::Qwen3_5EngineState;
#[cfg(feature = "direct-mlx")]
use super::super::engine_request::Qwen3_5EngineRequest;
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_failure::configured_speculative_prefill_failure;
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_store::Qwen3_5SpeculativePrefillStoreKey;

#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    /// Installs an exact reusable selection and returns whether scoring can be skipped.
    ///
    /// Visual requests bypass both stores here. The in-memory key does not include
    /// image digests, and persisted selection contracts intentionally represent
    /// text selection only; reusing either could bind positions to different images.
    pub(in crate::qwen3_5) fn reuse_speculative_prefill_selection_if_available(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
        selection_store_token_key: &Qwen3_5SpeculativePrefillStoreKey,
        scoring_start_position_tokens: usize,
        scoring_start_position_tokens_u32: u32,
        is_visual_speculative_prefill_request: bool,
    ) -> Result<bool, crate::InferenceEngineError> {
        // Worker-memory lookup is allocation-light and avoids all disk/MLX
        // selection work. Clone the tiny position vector out of the RefCell before
        // mutating other request state.
        if !is_visual_speculative_prefill_request
            && let Some(selected_token_positions) = self
                .speculative_prefill_selection_store
                .borrow()
                .get(selection_store_token_key)
                .cloned()
        {
            if let Err(prompt_token_indices_error) =
                self.prepare_speculative_prefill_prompt_token_indices_on_gpu(active_request)
            {
                return Err(configured_speculative_prefill_failure(
                    active_request.request_id,
                    "reused sparse target input assembly",
                    prompt_token_indices_error,
                ));
            }
            active_request.speculative_prefill_selected_token_positions =
                Some(selected_token_positions);
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillSelectionStoreHitCount,
                1,
            );
            tracing::info!(
                request_id = active_request.request_id.value(),
                selected_token_count = active_request
                    .speculative_prefill_selected_token_positions
                    .as_ref()
                    .map_or(0, Vec::len),
                selection_source = "worker_memory",
                "reused speculative-prefill selection"
            );
            return Ok(true);
        }
        // A disk lookup requires every identity component to be available: store,
        // policy contract, target runtime, and exact prompt tokens.
        if !is_visual_speculative_prefill_request
            && let Some(draft_persistent_prompt_cache) = self
                .speculative_prefill_draft_persistent_prompt_cache
                .as_ref()
            && let Some(selection_contract) = self.speculative_prefill_selection_contract(
                scoring_start_position_tokens_u32,
                active_request.input_token_ids.len(),
            )
            && let Some(target_model) = self.model.as_ref()
        {
            let persistent_selection_load_result =
                active_request.performance_attribution.measure_operation(
                    PerformanceOperation::SpeculativePrefillSelectionDiskRead,
                    |performance_attribution| {
                        draft_persistent_prompt_cache.load_speculative_prefill_selection(
                            target_model.runtime(),
                            &selection_contract,
                            &active_request.input_token_ids,
                            performance_attribution.positional_file_read_metrics(),
                        )
                    },
                );
            if let Err(selection_load_error) = &persistent_selection_load_result {
                // Configured storage errors are not cache misses. Fail closed so
                // corruption or I/O failure is visible rather than masked by scoring.
                return Err(configured_speculative_prefill_failure(
                    active_request.request_id,
                    "selection restoration",
                    selection_load_error,
                ));
            }
            if let Ok(Some(selected_token_positions_on_gpu)) = persistent_selection_load_result {
                // Copy once, then validate before installing. Persisted bytes are
                // untrusted boundary data even though the store validated framing.
                let selected_token_positions = target_model
                    .runtime()
                    .copy_u32_values(&selected_token_positions_on_gpu)
                    .ok()
                    .and_then(|selected_token_positions| {
                        selected_token_positions
                            .into_iter()
                            .map(|selected_token_position| {
                                usize::try_from(selected_token_position).ok()
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .filter(|selected_token_positions| {
                        // Sparse execution assumes strictly increasing absolute
                        // positions inside the selectable range and before the
                        // final generation-kickoff token.
                        selected_token_positions
                            .windows(2)
                            .all(|position_pair| position_pair[0] < position_pair[1])
                            && selected_token_positions
                                .iter()
                                .all(|selected_token_position| {
                                    *selected_token_position >= scoring_start_position_tokens
                                        && *selected_token_position
                                            < active_request.input_token_ids.len().saturating_sub(1)
                                })
                    });
                if let Some(selected_token_positions) = selected_token_positions {
                    if let Err(prompt_token_indices_error) =
                        self.prepare_speculative_prefill_prompt_token_indices_on_gpu(active_request)
                    {
                        return Err(configured_speculative_prefill_failure(
                            active_request.request_id,
                            "restored sparse target input assembly",
                            prompt_token_indices_error,
                        ));
                    }
                    self.store_speculative_prefill_selection(
                        selection_store_token_key.clone(),
                        selected_token_positions.clone(),
                    );
                    // Promote a validated disk hit into the worker hot cache so a
                    // repeat in this process avoids another disk read.
                    active_request.speculative_prefill_selected_token_positions =
                        Some(selected_token_positions);
                    active_request.performance_attribution.record_counter(
                        PerformanceCounter::SpeculativePrefillSelectionStoreHitCount,
                        1,
                    );
                    active_request.performance_attribution.record_counter(
                        PerformanceCounter::SpeculativePrefillSelectionPersistentHitCount,
                        1,
                    );
                    tracing::info!(
                        request_id = active_request.request_id.value(),
                        selected_token_count = active_request
                            .speculative_prefill_selected_token_positions
                            .as_ref()
                            .map_or(0, Vec::len),
                        selection_source = "persistent_storage",
                        "reused speculative-prefill selection"
                    );
                    return Ok(true);
                }
                return Err(configured_speculative_prefill_failure(
                    active_request.request_id,
                    "selection validation",
                    "the restored selection is outside the current selectable conversation range",
                ));
            }
        }
        // No exact reusable selection exists; the caller may proceed to fresh
        // request-scoped drafter scoring.
        Ok(false)
    }
}
