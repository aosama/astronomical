//! Pre-forward decision logic for a single prompt-processing chunk.
//!
//! This module computes all pre-admission decisions: speculative-prefill mode,
//! cache eligibility, checkpoint boundary planning, workspace byte projection,
//! and the adaptive-RAM growth context. No side effects — only decisions that
//! feed into admission and the forward dispatch.

use crate::{
    AdaptiveRamGrowthContext, persistent_prompt_cache_boundary_completed_prefill_chunck_tokens,
};

use super::engine_request::Qwen3_5EngineRequest;
use super::prefill_execution_context::Qwen3_5PrefillExecutionContext;
use super::prompt_prefill_errors::PromptPrefillChunckAttemptError;
use super::{
    Qwen3_5EngineState, Qwen3_5SpeculativePrefillChunckMode, fatal_engine_error,
    qwen3_5_runtime_error, qwen3_5_selected_speculative_prefill_positions_for_range,
    qwen3_5_speculative_prefill_chunck_mode, qwen3_5_speculative_prefill_sparse_target_is_active,
};

/// Immutable decisions computed before memory admission and forward execution.
///
/// Every field is a pure function of the request state and the model config.
/// Nothing here triggers GPU work or mutable engine state.
pub(super) struct PrefillChunckPlan {
    /// Token count of this chunk (`prefill_end - prefill_start`).
    pub(super) prefill_token_count: usize,

    /// Speculative-prefill mode for this chunk (dense prefix, sparse target,
    /// or terminal history capture).
    pub(super) speculative_prefill_chunck_mode: Qwen3_5SpeculativePrefillChunckMode,

    /// Whether sparse speculative-prefill target execution is active for this
    /// chunk. When true, only selected positions are forwarded on the dense
    /// target decoder.
    pub(super) speculative_prefill_target_is_active: bool,

    /// How many sparse target positions fall inside this chunk.
    pub(super) speculative_prefill_target_token_count: usize,

    /// Absolute prompt positions selected for speculative-prefill sparse
    /// target within this chunk.
    pub(super) selected_speculative_prefill_positions_for_current_chunck: Vec<usize>,

    /// Completed chunk-end offsets (relative to chunk start) where a prompt
    /// cache boundary checkpoint should be captured after the forward.
    pub(super) all_completed_prefill_chunck_tokens: Vec<usize>,

    /// Like `all_completed_prefill_chunck_tokens` but with the final boundary
    /// removed — intermediate checkpoints that need a snapshot during the
    /// forward.
    pub(super) intermediate_completed_prefill_chunck_tokens: Vec<usize>,

    /// Block token count from the persistent prompt-cache contract, if
    /// applicable.
    pub(super) persistent_prompt_cache_block_token_count: Option<usize>,

    /// Workspace bytes needed for direct prompt-cache publication.
    pub(super) direct_publication_workspace_bytes: usize,

    /// Combined temporary workspace reservation.
    pub(super) exact_temporary_workspace_bytes: usize,

    /// Extra KV-state bytes needed for terminal history capture.
    pub(super) additional_persistent_state_growth_bytes: usize,

    /// Adaptive-RAM growth context derived from this plan's decisions.
    pub(super) adaptive_ram_growth_context: AdaptiveRamGrowthContext,
}

impl Qwen3_5EngineState {
    /// Compute all pre-admission decisions for a prompt-processing chunk.
    ///
    /// Returns a plan that feeds into `execute_prompt_prefill_chunck`'s
    /// admission and forward phases. This method performs no GPU work and
    /// no mutable engine state changes.
    pub(super) fn plan_prompt_prefill_chunck(
        &self,
        active_request: &Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<PrefillChunckPlan, PromptPrefillChunckAttemptError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let prefill_token_count = prefill_end - prefill_start;
        let final_prompt_index = active_request
            .input_token_ids
            .len()
            .checked_sub(1)
            .ok_or_else(|| fatal_engine_error("generation prompt must not be empty"))?;

        let speculative_prefill_chunck_mode = qwen3_5_speculative_prefill_chunck_mode(
            active_request.has_optional_prediction_session(),
            prefill_end,
            final_prompt_index,
        );

        let speculative_prefill_sparse_conversation_range_is_active =
            qwen3_5_speculative_prefill_sparse_target_is_active(
                active_request.should_use_speculative_prefill,
                prefill_start,
                active_request.ordinary_target_prefill_control_span_token_count,
            );

        // Dense persistent-cache capture and sparse target execution represent
        // different decoder-state contracts and may never share one checkpoint.
        let capture_is_eligible = self.persistent_prompt_cache.is_some()
            && active_request.can_use_persistent_prompt_cache
            && !active_request.has_optional_prediction_session()
            && !speculative_prefill_sparse_conversation_range_is_active;

        // Cache-disabled and request-ineligible paths stop here: they do not
        // plan checkpoint boundaries, derive a synthetic block length, or
        // reserve checkpoint/publication memory. Cache-enabled execution uses
        // the one contract owned by the open store.
        let (all_completed_prefill_chunck_tokens, persistent_prompt_cache_block_token_count) =
            if capture_is_eligible {
                let persistent_prompt_cache_block_token_count = self
                    .persistent_prompt_cache
                    .as_ref()
                    .ok_or_else(|| {
                        fatal_engine_error(
                            "eligible prompt-cache capture has no persistent cache owner",
                        )
                    })?
                    .model_contract_ref()
                    .block_token_count();
                (
                    persistent_prompt_cache_boundary_completed_prefill_chunck_tokens(
                        prefill_start,
                        prefill_end,
                        persistent_prompt_cache_block_token_count,
                    ),
                    Some(persistent_prompt_cache_block_token_count),
                )
            } else {
                (Vec::new(), None)
            };

        let mut intermediate_completed_prefill_chunck_tokens =
            all_completed_prefill_chunck_tokens.clone();
        if intermediate_completed_prefill_chunck_tokens.last().copied() == Some(prefill_token_count)
        {
            intermediate_completed_prefill_chunck_tokens.pop();
        }

        let boundary_checkpoint_workspace_bytes =
            if intermediate_completed_prefill_chunck_tokens.is_empty() {
                0
            } else {
                model
                    .decoder_cache_layout()
                    .boundary_snapshot_payload_byte_count()
                    .map_err(|error| {
                        fatal_engine_error(format!(
                            "failed to project boundary checkpoint workspace: {error}"
                        ))
                    })?
                    .checked_mul(intermediate_completed_prefill_chunck_tokens.len())
                    .ok_or_else(|| {
                        fatal_engine_error("boundary checkpoint workspace bytes overflowed")
                    })?
            };

        let direct_publication_workspace_bytes = if !all_completed_prefill_chunck_tokens.is_empty()
        {
            self.persistent_prompt_cache
                .as_ref()
                .map(|persistent_prompt_cache| {
                    persistent_prompt_cache
                        .model_contract_ref()
                        .direct_publication_workspace_bytes()
                })
                .unwrap_or(0)
        } else {
            0
        };

        let exact_temporary_workspace_bytes = boundary_checkpoint_workspace_bytes
            .checked_add(direct_publication_workspace_bytes)
            .ok_or_else(|| {
                fatal_engine_error("prompt-cache publication workspace bytes overflowed")
            })?;

        let selected_speculative_prefill_positions_for_current_chunck =
            if active_request.should_use_speculative_prefill {
                active_request
                    .speculative_prefill_selected_token_positions
                    .as_deref()
                    .map_or_else(Vec::new, |selected_token_positions| {
                        qwen3_5_selected_speculative_prefill_positions_for_range(
                            selected_token_positions,
                            prefill_start,
                            prefill_end,
                        )
                    })
            } else {
                Vec::new()
            };

        // Global selection positions are ascending absolute offsets. Slice them
        // to this logical chunk without changing their original prompt positions.
        let speculative_prefill_target_token_count =
            selected_speculative_prefill_positions_for_current_chunck.len();

        let speculative_prefill_target_is_active =
            speculative_prefill_sparse_conversation_range_is_active
                && !matches!(
                    speculative_prefill_chunck_mode,
                    Qwen3_5SpeculativePrefillChunckMode::TerminalAdditionalHistoryCapture
                );

        // Terminal optional-history capture needs dense target hidden rows, so
        // it deliberately overrides sparse execution for that one final chunk.
        let additional_persistent_state_growth_bytes = match (
            speculative_prefill_chunck_mode,
            active_request.optional_prediction_session(),
        ) {
            (
                Qwen3_5SpeculativePrefillChunckMode::TerminalAdditionalHistoryCapture,
                Some(optional_prediction_session),
            ) if active_request.visual_embeddings.is_none() => {
                let additional_full_attention_bytes_per_layer_token = model
                    .config()
                    .full_attention_key_value_state_bytes_per_layer_token()
                    .ok_or_else(|| {
                        fatal_engine_error(
                            "additional full-attention bytes per layer token overflowed",
                        )
                    })?;
                optional_prediction_session
                    .projected_full_attention_growth_bytes(
                        additional_full_attention_bytes_per_layer_token,
                        prefill_token_count,
                    )
                    .map_err(qwen3_5_runtime_error)?
            }
            _ => 0,
        };

        let adaptive_ram_growth_context = AdaptiveRamGrowthContext::prefill(
            speculative_prefill_target_token_count,
            Qwen3_5PrefillExecutionContext::new(
                active_request.visual_embeddings.is_some(),
                active_request.has_optional_prediction_session(),
                model.sparse_experts_are_paged(),
                self.persistent_prompt_cache.is_some()
                    && active_request.can_use_persistent_prompt_cache
                    && !active_request.has_optional_prediction_session(),
            )
            .with_target_only_prefix(matches!(
                speculative_prefill_chunck_mode,
                Qwen3_5SpeculativePrefillChunckMode::TargetOnlyPrefix
            ))
            .with_speculative_prefill_sparse_target(speculative_prefill_target_is_active)
            .context_identifier_flags(),
            active_request.visual_embeddings.is_some(),
            active_request.has_optional_prediction_session(),
            model.sparse_experts_are_paged(),
        );

        Ok(PrefillChunckPlan {
            prefill_token_count,
            speculative_prefill_chunck_mode,
            speculative_prefill_target_is_active,
            speculative_prefill_target_token_count,
            selected_speculative_prefill_positions_for_current_chunck,
            all_completed_prefill_chunck_tokens,
            intermediate_completed_prefill_chunck_tokens,
            persistent_prompt_cache_block_token_count,
            direct_publication_workspace_bytes,
            exact_temporary_workspace_bytes,
            additional_persistent_state_growth_bytes,
            adaptive_ram_growth_context,
        })
    }
}
