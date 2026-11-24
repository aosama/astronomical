//! One folded-ternary request in the shared Qwen 3.5 request record.
//!
//! The request lives in [`Qwen3_5EngineRequest`] like a resident or
//! streaming request; the extension holds what only the folded-ternary loop
//! reads. The forward helpers here are the folded-ternary engine's own: the
//! position advances by the forwarded token count after every forward,
//! saturating instead of failing, exactly as before the record was shared.

use ipc_protocol::WorkerPromptWorkReuse;
use mlx_bridge::MlxArray;

use super::PrismHadamardQwen35ActiveGeneration;
use crate::prism_hadamard_qwen35::model::{PrismHadamardModelError, PrismHadamardTextModel};
use crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest;
use crate::{
    AstroSsdPromptCacheBlockKey, PerformanceAttribution, Qwen3_5InferenceRequest,
    Qwen3_5ThinkingBudgetState, RequestDecoderStateStack,
};

/// The per-request state only the folded-ternary loop reads.
pub struct PrismRequestExtension {
    /// Whether the reply reached its own end (an end-of-sequence token or
    /// the output budget). Every request ends through cancellation, so this
    /// is what tells a completed reply from one that was stopped.
    pub(in crate::prism_hadamard_qwen35::engine) has_reached_reply_end: bool,
    /// Output tokens the reply may still emit, counted down per emission.
    pub(in crate::prism_hadamard_qwen35::engine) remaining_output_tokens: u32,
    /// Prompt tokens no forward has read yet; prefill chunks drain it.
    pub(in crate::prism_hadamard_qwen35::engine) remaining_prompt_token_ids: Vec<u32>,
}

/// What a new folded-ternary request starts from, gathered by
/// `start_generation` before the record is built.
pub(super) struct PrismRequestStart {
    pub(super) prompt_token_ids: Vec<u32>,
    pub(super) request_decoder_state: RequestDecoderStateStack,
    pub(super) restored_token_count: u32,
    pub(super) last_restored_block_key: Option<AstroSsdPromptCacheBlockKey>,
    pub(super) random_state: MlxArray,
    pub(super) thinking_budget_state: Qwen3_5ThinkingBudgetState,
    pub(super) performance_attribution: PerformanceAttribution,
    pub(super) can_use_astro_ssd_prompt_cache: bool,
}

/// Builds the shared request record. The restored prefix is already in the
/// decoder state, so the first forward continues right after it, and the
/// prompt work it saved is reported on every prefill progress event.
pub(super) fn prism_active_generation(
    inference_request: &mut Qwen3_5InferenceRequest,
    request_start: PrismRequestStart,
) -> PrismHadamardQwen35ActiveGeneration {
    let restored_token_count = request_start.restored_token_count;
    let prompt_work_reuse = WorkerPromptWorkReuse {
        target_eligible_token_count: request_start.prompt_token_ids.len() as u64,
        target_restored_token_count: u64::from(restored_token_count),
    };
    let engine_extension = PrismRequestExtension {
        has_reached_reply_end: false,
        remaining_output_tokens: u32::from(inference_request.max_output_tokens()),
        remaining_prompt_token_ids: request_start.prompt_token_ids[restored_token_count as usize..]
            .to_vec(),
    };
    Qwen3_5EngineRequest {
        request_decoder_state: request_start.request_decoder_state,
        generated_token_count: 0,
        input_token_ids: request_start.prompt_token_ids,
        last_restored_astro_ssd_prompt_cache_block_key: request_start.last_restored_block_key,
        can_use_astro_ssd_prompt_cache: request_start.can_use_astro_ssd_prompt_cache,
        maximum_output_tokens: inference_request.max_output_tokens(),
        astro_ssd_prompt_cache_block_causal_inputs: Vec::new(),
        next_position_tokens: restored_token_count,
        pending_generated_token: None,
        prefill_cursor: restored_token_count as usize,
        random_state: Some(request_start.random_state),
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
        astro_ssd_prompt_cache_diagnostics: None,
        generation_residency_preparation_attempted: false,
        first_decode_forward_elapsed_millis: None,
        outgrown_attention_slab_reclaim: Default::default(),
        generation_preparation_announced: false,
        structured_generation: inference_request.take_structured_generation(),
        engine_extension,
    }
}

impl PrismHadamardQwen35ActiveGeneration {
    /// Runs the straight forward over `token_indices` (`[1, token_count]`)
    /// from the next position and returns its final hidden row, then
    /// advances the position by the forwarded token count.
    pub(super) fn forward_hidden_states(
        &mut self,
        model: &PrismHadamardTextModel,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let final_hidden_state_row = model.forward_hidden_states(
            token_indices,
            self.next_position_tokens,
            &mut self.request_decoder_state,
        )?;
        self.advance_position_after_forward(token_indices);
        Ok(final_hidden_state_row)
    }

    /// Runs one whole forward from the next position and returns its final
    /// logits, then advances the position by the forwarded token count.
    pub(super) fn forward_pass(
        &mut self,
        model: &PrismHadamardTextModel,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let logits = model.forward_pass(
            token_indices,
            self.next_position_tokens,
            &mut self.request_decoder_state,
        )?;
        self.advance_position_after_forward(token_indices);
        Ok(logits)
    }

    /// The serving checks cap every request at the model's context window,
    /// far inside u32, so the position never saturates in practice.
    fn advance_position_after_forward(&mut self, forwarded_token_indices: &MlxArray) {
        let forwarded_token_count = forwarded_token_indices.shape()[1];
        self.next_position_tokens = self
            .next_position_tokens
            .saturating_add_signed(forwarded_token_count);
    }
}

#[cfg(test)]
#[path = "active_generation_tests.rs"]
mod active_generation_tests;
