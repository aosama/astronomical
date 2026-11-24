//! Mutable state for one serial Qwen3.5 generation journey, shared by the
//! resident and streaming engines.
//!
//! This owner is the transaction boundary for sampling state and performance
//! attribution. State that can be rewound is deliberately kept here rather
//! than hidden in the model owner. State only one engine reads sits behind
//! the `RequestExtension` parameter, so the shared request code compiles once.

use ipc_protocol::{RequestId, WorkerAstroSsdPromptCacheRequestDiagnostics, WorkerPromptWorkReuse};
use mlx_bridge::MlxArray;

use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::decoder::reply_token_mlx_buffer_cache_reclaim::OutgrownAttentionSlabReclaim;
use crate::qwen3_5_core::engine_errors::{fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::text::sampler;
use crate::{
    AstroSsdPromptCacheBlockKey, InferenceEngineError, PerformanceAttribution,
    PerformanceOperation, Qwen3_5SamplingStrategy, Qwen3_5ThinkingBudgetState,
};
use engine_core::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;

/// One request's state in either Qwen3.5 engine. The fields both engines
/// read live here; `engine_extension` holds what only one engine needs
/// (speculative decoding in the resident engine, prefill memory recovery in
/// the streaming engine), so shared request code compiles once.
pub struct Qwen3_5EngineRequest<RequestExtension> {
    pub request_decoder_state: RequestDecoderStateStack,
    pub generated_token_count: u16,
    pub input_token_ids: Vec<u32>,
    pub last_restored_astro_ssd_prompt_cache_block_key: Option<AstroSsdPromptCacheBlockKey>,
    pub can_use_astro_ssd_prompt_cache: bool,
    pub maximum_output_tokens: u16,
    /// Qwen-owned visual identity aligned to ordinary target astro SSD prompt cache blocks.
    pub astro_ssd_prompt_cache_block_causal_inputs: Vec<AstroSsdPromptCacheBlockCausalInput>,
    pub next_position_tokens: u32,
    /// One-token-ahead successor submitted during the previous advancement.
    pub pending_generated_token: Option<MlxArray>,
    pub prefill_cursor: usize,
    pub random_state: Option<MlxArray>,
    pub request_id: RequestId,
    pub sampling_strategy: Qwen3_5SamplingStrategy,
    /// Resolved repetition penalty in thousandths; 1,000 is the neutral identity.
    pub repetition_penalty_thousandths: u16,
    /// Sorted unique ids the penalty dampens: prompt ids plus every generated
    /// token committed so far. Empty while the penalty is neutral so the
    /// sampling graph stays dispatch-free.
    pub penalized_token_ids: Vec<u32>,
    /// Pre-uploaded visual embeddings for image prompts; `None` for text-only.
    pub visual_embeddings: Option<MlxArray>,
    /// How many visual embeddings earlier prefill chunks have already consumed.
    pub consumed_visual_embedding_count: usize,
    /// Whether the request was constructed with an image source. Ordinary text may
    /// contain a vocabulary collision with the artifact's image-pad ID; only genuine
    /// image requests may demand visual embedding rows.
    pub has_visual_inputs: bool,
    /// Token ID used for image-pad placeholders in the input prompt.
    pub image_pad_token_id: u32,
    pub thinking_budget_state: Qwen3_5ThinkingBudgetState,
    pub performance_attribution: PerformanceAttribution,
    pub prompt_work_reuse: WorkerPromptWorkReuse,
    pub astro_ssd_prompt_cache_diagnostics: Option<WorkerAstroSsdPromptCacheRequestDiagnostics>,
    /// One-shot guard for no-I/O prefill-to-decode residency reconciliation.
    /// Mandatory decode reads populate any remaining elastic route ownership.
    pub generation_residency_preparation_attempted: bool,
    /// First decode-forward latency waiting to be emitted with its generated token.
    pub first_decode_forward_elapsed_millis: Option<u64>,
    /// Tracks attention slab growth so the reply path can release the slabs it outgrew.
    pub outgrown_attention_slab_reclaim: OutgrownAttentionSlabReclaim,
    /// Ensures the worker observes generation preparation before blocking handoff work.
    pub generation_preparation_announced: bool,
    pub structured_generation: Option<crate::structured_generation::StructuredTokenConstraint>,
    /// State only the owning engine reads.
    pub engine_extension: RequestExtension,
}

impl<RequestExtension> Qwen3_5EngineRequest<RequestExtension> {
    /// Releases attention slabs the last reply step outgrew; the rule and
    /// its measurements live with [`OutgrownAttentionSlabReclaim`].
    pub fn reclaim_outgrown_attention_slabs(
        &mut self,
        runtime: &runtime_integration::MlxRuntime,
    ) -> Result<(), InferenceEngineError> {
        self.outgrown_attention_slab_reclaim
            .reclaim_after_slab_growth(runtime, self.request_decoder_state.payload_byte_count())
            .map_err(qwen3_5_runtime_error)
    }

    /// Records one generated token in the penalty history, keeping it sorted and unique.
    ///
    /// Only requests with an active penalty call this, so the common path never
    /// pays the sorted-insert cost.
    pub fn observe_penalized_token_id(&mut self, generated_token_id: u32) {
        if let Err(insertion_index) = self.penalized_token_ids.binary_search(&generated_token_id) {
            self.penalized_token_ids
                .insert(insertion_index, generated_token_id);
        }
    }

    pub fn measure_operation_with_request<OperationOutput>(
        &mut self,
        performance_operation: PerformanceOperation,
        operation: impl FnOnce(&mut Self) -> OperationOutput,
    ) -> OperationOutput {
        if !self.performance_attribution.is_enabled() {
            return operation(self);
        }

        // Temporarily move attribution out of `self` so the measured closure
        // can borrow the complete mutable request. The disabled placeholder
        // prevents nested request code from recording the same outer span as a
        // leaf operation; the original accumulator is always restored.
        let mut request_performance_attribution = std::mem::replace(
            &mut self.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        let operation_output = request_performance_attribution
            .measure_operation(performance_operation, |_performance_attribution| {
                operation(self)
            });
        self.performance_attribution = request_performance_attribution;
        operation_output
    }

    pub fn build_generated_token(
        &mut self,
        model: &Qwen3_5ModelBase,
        logits: &MlxArray,
    ) -> Result<MlxArray, InferenceEngineError> {
        let sampling_strategy = self.sampling_strategy;
        let mut sampling_random_state = self.random_state.take();
        // The penalty transforms raw logits before the structured mask and the
        // sampling strategy, matching the HF reference ordering; a neutral
        // penalty returns None and keeps the sampling graph unchanged.
        let penalized_logits = self.performance_attribution.measure_operation(
            PerformanceOperation::RepetitionPenaltyLogitTransform,
            |_performance_attribution| {
                crate::gpu_token_sampling::apply_repetition_penalty(
                    &model.runtime,
                    logits,
                    &self.penalized_token_ids,
                    self.repetition_penalty_thousandths,
                )
            },
        )?;
        let logits = penalized_logits.as_ref().unwrap_or(logits);
        let outside_thinking = !self.is_inside_thinking() && !self.is_forcing_thinking_transition();
        let masked_logits = if outside_thinking
            && let Some(structured_generation) = self.structured_generation.as_mut()
        {
            let logit_bias_values = self.performance_attribution.measure_operation(
                PerformanceOperation::StructuredLogitMaskComputation,
                |_performance_attribution| structured_generation.logit_bias_values(),
            );
            Some(crate::gpu_token_sampling::add_token_logit_bias(
                &model.runtime,
                logits,
                &logit_bias_values,
            )?)
        } else {
            None
        };
        let logits = masked_logits.as_ref().unwrap_or(logits);
        let generated_token_outcome = self.performance_attribution.measure_operation(
            PerformanceOperation::TokenSamplingGraphConstruction,
            |_performance_attribution| {
                sample_generated_token(model, logits, sampling_strategy, &mut sampling_random_state)
            },
        );
        self.random_state = sampling_random_state;
        generated_token_outcome
    }

    pub fn advance_position(
        &mut self,
        forwarded_token_count: usize,
    ) -> Result<(), InferenceEngineError> {
        let forwarded_token_count = u32::try_from(forwarded_token_count)
            .map_err(|_| fatal_engine_error("forwarded token count exceeds the u32 range"))?;
        self.next_position_tokens = self
            .next_position_tokens
            .checked_add(forwarded_token_count)
            .ok_or_else(|| fatal_engine_error("model position counter overflowed"))?;
        Ok(())
    }

    pub fn is_inside_thinking(&self) -> bool {
        self.thinking_budget_state.is_inside_thinking()
    }

    pub fn next_forced_thinking_transition_token_id(
        &mut self,
    ) -> Result<Option<u32>, InferenceEngineError> {
        self.thinking_budget_state
            .next_forced_transition_token_id()
            .map_err(|source| {
                fatal_engine_error(format!("invalid Qwen3.5 thinking-budget state: {source}"))
            })
    }

    pub fn observe_committed_thinking_token(
        &mut self,
        committed_token_id: u32,
    ) -> Result<bool, InferenceEngineError> {
        self.thinking_budget_state
            .observe_committed_token(committed_token_id)
            .map_err(|source| {
                fatal_engine_error(format!("invalid Qwen3.5 thinking-budget state: {source}"))
            })
    }

    pub fn is_forcing_thinking_transition(&self) -> bool {
        self.thinking_budget_state.is_forcing_transition()
    }

    pub fn set_pending_generated_token(&mut self, pending_generated_token: MlxArray) {
        self.pending_generated_token = Some(pending_generated_token);
    }
}

/// Samples the next token under the request's sampling strategy.
///
/// Split from the measurement call so the two strategy arms read as named
/// dispatch; the caller owns the random state and restores it after
/// measurement so a lost state surfaces once, not per arm.
fn sample_generated_token(
    model: &Qwen3_5ModelBase,
    logits: &MlxArray,
    sampling_strategy: Qwen3_5SamplingStrategy,
    sampling_random_state: &mut Option<MlxArray>,
) -> Result<MlxArray, InferenceEngineError> {
    match sampling_strategy {
        Qwen3_5SamplingStrategy::HighestLogit => model
            .select_highest_logit_token(logits)
            .map_err(qwen3_5_runtime_error),
        Qwen3_5SamplingStrategy::TopKTopP {
            temperature_thousandths,
            top_k,
            top_p_thousandths,
            ..
        } => sampler::build_qwen3_5_sampled_token(
            model,
            logits,
            temperature_thousandths,
            top_p_thousandths,
            top_k,
            sampling_random_state
                .as_mut()
                .ok_or_else(|| fatal_engine_error("sampled request lost its random state"))?,
        ),
    }
}

#[cfg(test)]
#[path = "engine_request_tests.rs"]
mod engine_request_tests;
