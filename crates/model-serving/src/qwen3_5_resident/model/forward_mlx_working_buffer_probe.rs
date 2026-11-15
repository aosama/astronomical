//! Measures the MLX working buffers one forward of this engine needs.
//!
//! The memory governor sets MLX working buffers aside for a forward beside the
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
pub(crate) struct ProbedForwardMlxWorkingBuffers {
    pub(crate) prompt_chunk_token_count: usize,
    pub(crate) prompt_chunk_mlx_working_buffer_bytes: u64,
    pub(crate) reply_token_mlx_working_buffer_bytes: u64,
}

impl Qwen3_5ResidentModel {
    /// Runs one prompt chunk and one reply token on a state of its own and
    /// returns what each needed for a moment. The state is dropped and the
    /// MLX buffer cache cleared before returning.
    pub(crate) fn measure_forward_mlx_working_buffers(
        &self,
        prompt_chunk_token_count: usize,
        full_attention_kv_state_growth_tokens: i32,
    ) -> Result<ProbedForwardMlxWorkingBuffers, Qwen3_5ExecutionError> {
        let probe_token_ids = self.probe_token_ids(prompt_chunk_token_count)?;
        let mut probe_decoder_state =
            RequestDecoderStateStack::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                self.decoder_cache_layout(),
                full_attention_kv_state_growth_tokens,
            )?;
        let prompt_chunk_mlx_working_buffer_bytes =
            self.measure_probe_prompt_chunk(&probe_token_ids, &mut probe_decoder_state)?;
        let reply_token_mlx_working_buffer_bytes =
            self.measure_probe_reply_token(&probe_token_ids, &mut probe_decoder_state)?;
        drop(probe_decoder_state);
        self.runtime
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
        Ok(ProbedForwardMlxWorkingBuffers {
            prompt_chunk_token_count,
            prompt_chunk_mlx_working_buffer_bytes,
            reply_token_mlx_working_buffer_bytes,
        })
    }

    /// Runs the probe's prompt chunk and returns how far MLX's peak rose
    /// above what was held around it.
    fn measure_probe_prompt_chunk(
        &self,
        probe_token_ids: &[u32],
        probe_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        self.runtime
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
        let active_bytes_before_chunk = self.runtime.memory_snapshot()?.active_memory_bytes();
        self.runtime.reset_peak_memory()?;
        self.prefill_chunk_with_performance_attribution(
            probe_token_ids,
            0,
            probe_decoder_state,
            &mut PerformanceAttribution::disabled(),
        )?;
        self.peak_beyond_held_after_settling(active_bytes_before_chunk)
    }

    /// Runs one reply token after the probe's chunk and returns the same
    /// rise for it.
    fn measure_probe_reply_token(
        &self,
        probe_token_ids: &[u32],
        probe_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        let active_bytes_before_reply_token = self.runtime.memory_snapshot()?.active_memory_bytes();
        self.runtime.reset_peak_memory()?;
        let reply_logits = self.build_forward_chunk_with_performance_attribution(
            &probe_token_ids[..1],
            u32::try_from(probe_token_ids.len()).unwrap_or(u32::MAX),
            probe_decoder_state,
            &mut PerformanceAttribution::disabled(),
        )?;
        self.runtime
            .evaluate_arrays(&forward_state_arrays(&reply_logits, probe_decoder_state)?)?;
        drop(reply_logits);
        self.peak_beyond_held_after_settling(active_bytes_before_reply_token)
    }

    /// How far the peak rose above the larger of what was held before the
    /// step and what is held now.
    ///
    /// Arrays a step has finished with are released when the stream is
    /// synchronized. Reading what is held before that counts them as kept
    /// and under-reads the MLX working buffers: measured 0.53 GB against the
    /// 1.10 GB a real prompt step needed, and 1.09 GB once settled.
    fn peak_beyond_held_after_settling(
        &self,
        active_bytes_before_step: usize,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        self.runtime
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
        let memory_after_step = self.runtime.memory_snapshot()?;
        Ok(memory_after_step
            .peak_memory_bytes()
            .saturating_sub(active_bytes_before_step.max(memory_after_step.active_memory_bytes()))
            as u64)
    }

    fn probe_token_ids(
        &self,
        prompt_chunk_token_count: usize,
    ) -> Result<Vec<u32>, Qwen3_5ExecutionError> {
        let signed_token_count = i32::try_from(prompt_chunk_token_count).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "the MLX working-buffer probe token count exceeds the MLX range",
            }
        })?;
        if signed_token_count == 0 {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "the MLX working-buffer probe needs at least one token",
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
