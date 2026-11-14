//! Measures the working memory one forward of this engine needs.
//!
//! The memory governor sets working memory aside for a forward beside the
//! weights and the stored state (issue #2221). Rather than assume an
//! amount, or let the first request find out, the engine runs one prompt
//! chunk of the size it will serve and one reply token at load and reads
//! MLX's peak. Every later forward is measured too, and the figure only
//! grows.

use super::Qwen3_5ResidentModel;
use crate::PerformanceAttribution;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::model::probe_token_ids::vocabulary_walking_token_ids;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::forward_contract::forward_state_arrays;

/// What one prompt chunk and one reply token needed beyond the weights and
/// the stored state they left behind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProbedForwardWorkingMemory {
    pub(crate) prompt_chunk_token_count: usize,
    pub(crate) prompt_chunk_working_memory_bytes: u64,
    pub(crate) reply_token_working_memory_bytes: u64,
}

impl Qwen3_5ResidentModel {
    /// Runs one prompt chunk and one reply token on a state of its own and
    /// returns what each needed for a moment. The state is dropped and the
    /// allocator cache cleared before returning.
    pub(crate) fn measure_forward_working_memory(
        &self,
        prompt_chunk_token_count: usize,
        full_attention_kv_state_growth_tokens: i32,
    ) -> Result<ProbedForwardWorkingMemory, Qwen3_5ExecutionError> {
        let probe_token_ids = self.probe_token_ids(prompt_chunk_token_count)?;
        let mut probe_decoder_state =
            RequestDecoderStateStack::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                self.decoder_cache_layout(),
                full_attention_kv_state_growth_tokens,
            )?;
        self.runtime
            .synchronize_gpu_stream_and_clear_allocator_cache()?;
        let active_bytes_before_chunk = self.runtime.memory_snapshot()?.active_memory_bytes();
        self.runtime.reset_peak_memory()?;
        self.prefill_chunk_with_performance_attribution(
            &probe_token_ids,
            0,
            &mut probe_decoder_state,
            &mut PerformanceAttribution::disabled(),
        )?;
        let memory_after_chunk = self.runtime.memory_snapshot()?;
        let active_bytes_after_chunk = memory_after_chunk.active_memory_bytes();
        let prompt_chunk_working_memory_bytes = memory_after_chunk
            .peak_memory_bytes()
            .saturating_sub(active_bytes_before_chunk.max(active_bytes_after_chunk));

        self.runtime.reset_peak_memory()?;
        let reply_logits = self.build_forward_chunk_with_performance_attribution(
            &probe_token_ids[..1],
            u32::try_from(prompt_chunk_token_count).unwrap_or(u32::MAX),
            &mut probe_decoder_state,
            &mut PerformanceAttribution::disabled(),
        )?;
        self.runtime
            .evaluate_arrays(&forward_state_arrays(&reply_logits, &probe_decoder_state)?)?;
        let memory_after_reply_token = self.runtime.memory_snapshot()?;
        let reply_token_working_memory_bytes =
            memory_after_reply_token.peak_memory_bytes().saturating_sub(
                active_bytes_after_chunk.max(memory_after_reply_token.active_memory_bytes()),
            );

        drop(reply_logits);
        drop(probe_decoder_state);
        self.runtime
            .synchronize_gpu_stream_and_clear_allocator_cache()?;
        Ok(ProbedForwardWorkingMemory {
            prompt_chunk_token_count,
            prompt_chunk_working_memory_bytes: prompt_chunk_working_memory_bytes as u64,
            reply_token_working_memory_bytes: reply_token_working_memory_bytes as u64,
        })
    }

    fn probe_token_ids(
        &self,
        prompt_chunk_token_count: usize,
    ) -> Result<Vec<u32>, Qwen3_5ExecutionError> {
        let signed_token_count = i32::try_from(prompt_chunk_token_count).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "the working-memory probe token count exceeds the MLX range",
            }
        })?;
        if signed_token_count == 0 {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "the working-memory probe needs at least one token",
            });
        }
        let mut probe_token_ids = Vec::with_capacity(prompt_chunk_token_count);
        for signed_token_id in
            vocabulary_walking_token_ids(signed_token_count, self.config.vocabulary_size())
        {
            probe_token_ids.push(u32::try_from(signed_token_id).unwrap_or(0));
        }
        Ok(probe_token_ids)
    }
}
