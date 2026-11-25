//! One folded-ternary request in the shared Qwen 3.5 request record.
//!
//! The request lives in [`TextEngineRequest`] like a resident or
//! streaming request; the extension holds what only the folded-ternary loop
//! reads. The forward helpers here are the folded-ternary engine's own: the
//! position advances by the forwarded token count after every forward,
//! saturating instead of failing, exactly as before the record was shared.

use ipc_protocol::{WorkerAstroSsdPromptCacheRequestDiagnostics, WorkerPromptWorkReuse};
use mlx_bridge::MlxArray;

use super::PrismHadamardQwen35ActiveGeneration;
use crate::prism_hadamard_qwen35::model::{PrismHadamardModelError, PrismHadamardTextModel};
use crate::{
    AstroSsdPromptCacheBlockKey, PerformanceAttribution, RequestDecoderStateStack,
    TextInferenceRequest, ThinkingBudgetState,
};
use text_generation::engine_request::TextEngineRequest;
use text_generation::generation_loop::astro_ssd_prompt_cache_restore::AstroSsdPromptCacheRestoreOutcome;

/// The per-request state only the folded-ternary loop reads: none since its
/// prompt reading moved onto the shared cursor (issue #2473, Prism step 9).
/// The type stays so the folded-ternary record keeps its own extension.
#[derive(Default)]
pub struct PrismRequestExtension;

/// What a new folded-ternary request starts from, gathered by
/// `start_generation` before the record is built.
pub(super) struct PrismRequestStart {
    pub(super) prompt_token_ids: Vec<u32>,
    pub(super) request_decoder_state: RequestDecoderStateStack,
    /// What the `astro_ssd_prompt_cache` restored into the decoder state;
    /// `None` when no store is open.
    pub(super) restore_outcome: Option<AstroSsdPromptCacheRestoreOutcome>,
    /// The sampler's random state; none for a greedy request.
    pub(super) random_state: Option<MlxArray>,
    pub(super) thinking_budget_state: ThinkingBudgetState,
    pub(super) performance_attribution: PerformanceAttribution,
    pub(super) can_use_astro_ssd_prompt_cache: bool,
}

/// Builds the shared request record. The restored prefix is already in the
/// decoder state, so the first forward continues right after it, and the
/// prompt work it saved is reported on every prefill progress event.
/// What a request starts from: the restored token count, the last restored
/// block key and the restore diagnostics, or nothing restored.
fn restored_prefix_facts(
    restore_outcome: Option<AstroSsdPromptCacheRestoreOutcome>,
) -> (
    u32,
    Option<AstroSsdPromptCacheBlockKey>,
    Option<WorkerAstroSsdPromptCacheRequestDiagnostics>,
) {
    match restore_outcome {
        Some(restore_outcome) => (
            restore_outcome.astro_ssd_prompt_cache_restored_token_count,
            restore_outcome.last_restored_astro_ssd_prompt_cache_block_key,
            Some(restore_outcome.astro_ssd_prompt_cache_diagnostics),
        ),
        None => (0, None, None),
    }
}

pub(super) fn prism_active_generation(
    inference_request: &mut TextInferenceRequest,
    request_start: PrismRequestStart,
) -> PrismHadamardQwen35ActiveGeneration {
    let (restored_token_count, last_restored_block_key, astro_ssd_prompt_cache_diagnostics) =
        restored_prefix_facts(request_start.restore_outcome);
    let prompt_work_reuse = WorkerPromptWorkReuse {
        target_eligible_token_count: request_start.prompt_token_ids.len() as u64,
        target_restored_token_count: u64::from(restored_token_count),
    };
    TextEngineRequest {
        request_decoder_state: request_start.request_decoder_state,
        generated_token_count: 0,
        input_token_ids: request_start.prompt_token_ids,
        last_restored_astro_ssd_prompt_cache_block_key: last_restored_block_key,
        can_use_astro_ssd_prompt_cache: request_start.can_use_astro_ssd_prompt_cache,
        maximum_output_tokens: inference_request.max_output_tokens(),
        astro_ssd_prompt_cache_block_causal_inputs: Vec::new(),
        next_position_tokens: restored_token_count,
        pending_generated_token: None,
        prefill_cursor: restored_token_count as usize,
        random_state: request_start.random_state,
        request_id: inference_request.request_id(),
        sampling_strategy: inference_request.sampling_strategy(),
        repetition_penalty_thousandths: inference_request.repetition_penalty_thousandths(),
        penalized_token_ids: inference_request.initial_penalized_token_ids(),
        visual_embeddings: None,
        consumed_visual_embedding_count: 0,
        has_visual_inputs: false,
        // A folded pack is text-only, so no prompt token is an image row.
        image_pad_token_id: 0,
        thinking_budget_state: request_start.thinking_budget_state,
        performance_attribution: request_start.performance_attribution,
        prompt_work_reuse,
        astro_ssd_prompt_cache_diagnostics,
        generation_residency_preparation_attempted: false,
        first_decode_forward_elapsed_millis: None,
        outgrown_attention_slab_reclaim: Default::default(),
        generation_preparation_announced: false,
        structured_generation: inference_request.take_structured_generation(),
        engine_extension: PrismRequestExtension,
    }
}

/// The forward the Prism engine runs on its request. A trait, because the
/// request type belongs to the text shell and takes no inherent methods here.
pub(super) trait PrismForwardingRequest {
    /// Runs one whole forward from the next position and returns its final
    /// logits, then advances the position by the forwarded token count.
    fn forward_pass(
        &mut self,
        model: &PrismHadamardTextModel,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, PrismHadamardModelError>;
}

impl PrismForwardingRequest for PrismHadamardQwen35ActiveGeneration {
    fn forward_pass(
        &mut self,
        model: &PrismHadamardTextModel,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let logits = model.forward_pass(
            token_indices,
            self.next_position_tokens,
            &mut self.request_decoder_state,
        )?;
        // The serving checks cap every request at the model's context
        // window, far inside u32, so the position never saturates in practice.
        let forwarded_token_count = token_indices.shape()[1];
        self.next_position_tokens = self
            .next_position_tokens
            .saturating_add_signed(forwarded_token_count);
        Ok(logits)
    }
}

#[cfg(test)]
#[path = "active_generation_tests.rs"]
mod active_generation_tests;
