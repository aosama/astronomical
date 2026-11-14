//! Pre-forward decision logic for a single prompt-processing chunk.
//!
//! This module computes cache eligibility, checkpoint boundaries, workspace
//! projection, and the token count the forward is measured under, without
//! side effects.

use crate::persistent_prompt_cache_boundary_completed_prefill_chunk_tokens;

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::layer_major_span_plan::LayerMajorSpanPlan;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
/// Immutable decisions computed before a prompt forward runs.
///
/// Every field is a pure function of the request state and the model config.
/// Nothing here triggers GPU work or mutable engine state.
pub(super) struct PrefillChunkPlan {
    /// Token count of this chunk (`prefill_end - prefill_start`).
    pub(super) prefill_token_count: usize,

    /// Completed chunk-end offsets (relative to chunk start) where a prompt
    /// cache boundary checkpoint should be captured after the forward.
    pub(super) all_completed_prefill_chunk_tokens: Vec<usize>,

    /// Like `all_completed_prefill_chunk_tokens` but with the final boundary
    /// removed — intermediate checkpoints that need a snapshot during the
    /// forward.
    pub(super) intermediate_completed_prefill_chunk_tokens: Vec<usize>,

    /// Block token count from the persistent prompt-cache contract, if
    /// applicable.
    pub(super) persistent_prompt_cache_block_token_count: Option<usize>,

    /// The tokens the completed forward's memory is measured under: the
    /// chunk's, or one span's for a group read one layer at a time, whose
    /// peak is one span's working memory and never the whole group's.
    pub(super) measured_forward_token_count: usize,

    /// The spans a dense model runs layer by layer, when this forward
    /// covers several chunks at once.
    pub(super) layer_major_span_plan: Option<LayerMajorSpanPlan>,
}

impl Qwen3_5EngineState {
    /// The spans a layer-major prefill would run from `prefill_start`, for a
    /// text request on a model that reads layers from storage and has not
    /// had a layer-major prefill fail.
    pub(super) fn layer_major_span_plan_for(
        &self,
        active_request: &Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end_exclusive: usize,
    ) -> Option<LayerMajorSpanPlan> {
        let model = self.model.as_ref()?;
        if active_request.engine_extension.layer_major_prefill_declined
            || active_request.has_visual_inputs
            || !model.reads_prompts_one_layer_at_a_time()
        {
            return None;
        }
        let capture_block_token_count = self
            .persistent_prompt_cache
            .as_ref()
            .filter(|_| active_request.can_use_persistent_prompt_cache)
            .map(|persistent_prompt_cache| {
                persistent_prompt_cache
                    .model_contract_ref()
                    .block_token_count()
            });
        super::layer_major_span_plan::layer_major_span_plan(
            prefill_start,
            prefill_end_exclusive,
            self.prompt_processing_chunk_sizer
                .maximum_prompt_processing_chunk_size_tokens(),
            capture_block_token_count,
        )
    }

    /// Compute every decision a prompt-processing chunk needs before it runs.
    ///
    /// Returns a plan that feeds into `execute_prompt_prefill_chunk`. This
    /// method performs no GPU work and no mutable engine state changes.
    pub(super) fn plan_prompt_prefill_chunk(
        &self,
        active_request: &Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
        layer_major_span_plan: Option<LayerMajorSpanPlan>,
    ) -> Result<PrefillChunkPlan, PromptPrefillChunkAttemptError> {
        let prefill_token_count = prefill_end - prefill_start;
        let (all_completed_prefill_chunk_tokens, persistent_prompt_cache_block_token_count) =
            self.planned_prompt_cache_boundaries(active_request, prefill_start, prefill_end)?;

        let mut intermediate_completed_prefill_chunk_tokens =
            all_completed_prefill_chunk_tokens.clone();
        if intermediate_completed_prefill_chunk_tokens.last().copied() == Some(prefill_token_count)
        {
            intermediate_completed_prefill_chunk_tokens.pop();
        }

        let measured_forward_token_count = layer_major_span_plan.as_ref().map_or(
            prefill_token_count,
            LayerMajorSpanPlan::largest_span_token_count,
        );

        Ok(PrefillChunkPlan {
            prefill_token_count,
            all_completed_prefill_chunk_tokens,
            intermediate_completed_prefill_chunk_tokens,
            persistent_prompt_cache_block_token_count,
            measured_forward_token_count,
            layer_major_span_plan,
        })
    }

    /// The prompt-cache boundaries this forward captures, relative to its
    /// start, and the block size they fall on.
    ///
    /// Cache-disabled and request-ineligible paths plan none: they do not
    /// derive a synthetic block length or reserve checkpoint and publication
    /// memory. Cache-enabled execution uses the one contract the open store
    /// owns.
    fn planned_prompt_cache_boundaries(
        &self,
        active_request: &Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<(Vec<usize>, Option<usize>), PromptPrefillChunkAttemptError> {
        if self.persistent_prompt_cache.is_none() || !active_request.can_use_persistent_prompt_cache
        {
            return Ok((Vec::new(), None));
        }
        let persistent_prompt_cache_block_token_count = self
            .persistent_prompt_cache
            .as_ref()
            .ok_or_else(|| {
                super::fatal_engine_error(
                    "eligible prompt-cache capture has no persistent cache owner",
                )
            })?
            .model_contract_ref()
            .block_token_count();
        Ok((
            persistent_prompt_cache_boundary_completed_prefill_chunk_tokens(
                prefill_start,
                prefill_end,
                persistent_prompt_cache_block_token_count,
            ),
            Some(persistent_prompt_cache_block_token_count),
        ))
    }
}
