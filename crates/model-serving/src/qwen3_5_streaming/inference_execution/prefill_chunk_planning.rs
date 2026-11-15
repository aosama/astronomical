//! Decisions made before a stretch of a prompt is read.
//!
//! This module computes the spans, astro SSD astro SSD prompt cache eligibility, checkpoint
//! boundaries and the token count the forward is measured under, without
//! side effects.

use crate::astro_ssd_prompt_cache_boundary_completed_prefill_chunk_tokens;

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
    /// astro SSD astro SSD prompt cache boundary checkpoint should be captured after the forward.
    pub(super) all_completed_prefill_chunk_tokens: Vec<usize>,

    /// Block token count from the persistent astro-ssd-prompt-cache contract, if
    /// applicable.
    pub(super) astro_ssd_prompt_cache_block_token_count: Option<usize>,

    /// The tokens the completed forward's memory is measured under: the
    /// largest span's, whose MLX working buffers the peak is; never the whole
    /// stretch's.
    pub(super) measured_forward_token_count: usize,

    /// The spans the stretch is read in.
    pub(super) layer_major_span_plan: LayerMajorSpanPlan,
}

impl Qwen3_5EngineState {
    /// The spans the prompt is read in from `prefill_start`.
    pub(super) fn layer_major_span_plan_for(
        &self,
        active_request: &Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end_exclusive: usize,
    ) -> Option<LayerMajorSpanPlan> {
        let capture_block_token_count = self
            .astro_ssd_prompt_cache
            .as_ref()
            .filter(|_| active_request.can_use_astro_ssd_prompt_cache)
            .map(|astro_ssd_prompt_cache| {
                astro_ssd_prompt_cache
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
        layer_major_span_plan: LayerMajorSpanPlan,
    ) -> Result<PrefillChunkPlan, PromptPrefillChunkAttemptError> {
        let prefill_token_count = prefill_end - prefill_start;
        let (all_completed_prefill_chunk_tokens, astro_ssd_prompt_cache_block_token_count) = self
            .planned_astro_ssd_prompt_cache_boundaries(
            active_request,
            prefill_start,
            prefill_end,
        )?;

        let measured_forward_token_count = layer_major_span_plan.largest_span_token_count();

        Ok(PrefillChunkPlan {
            prefill_token_count,
            all_completed_prefill_chunk_tokens,
            astro_ssd_prompt_cache_block_token_count,
            measured_forward_token_count,
            layer_major_span_plan,
        })
    }

    /// The astro-ssd-prompt-cache boundaries this forward captures, relative to its
    /// start, and the block size they fall on.
    ///
    /// Cache-disabled and request-ineligible paths plan none: they do not
    /// derive a synthetic block length or reserve checkpoint and publication
    /// memory. Cache-enabled execution uses the one contract the open store
    /// owns.
    fn planned_astro_ssd_prompt_cache_boundaries(
        &self,
        active_request: &Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<(Vec<usize>, Option<usize>), PromptPrefillChunkAttemptError> {
        if self.astro_ssd_prompt_cache.is_none() || !active_request.can_use_astro_ssd_prompt_cache {
            return Ok((Vec::new(), None));
        }
        let astro_ssd_prompt_cache_block_token_count = self
            .astro_ssd_prompt_cache
            .as_ref()
            .ok_or_else(|| {
                super::fatal_engine_error(
                    "eligible astro-ssd-prompt-cache capture has no persistent cache owner",
                )
            })?
            .model_contract_ref()
            .block_token_count();
        Ok((
            astro_ssd_prompt_cache_boundary_completed_prefill_chunk_tokens(
                prefill_start,
                prefill_end,
                astro_ssd_prompt_cache_block_token_count,
            ),
            Some(astro_ssd_prompt_cache_block_token_count),
        ))
    }
}
