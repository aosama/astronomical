//! Measures the MLX working buffers one expert layer's prompt step needs.
//!
//! The memory ledger sets MLX working buffers aside for one layer step, beside
//! the weights that step reads (issue #2221). Rather than assume an amount,
//! or wait for the first request to reveal it, the engine runs one
//! linear-attention and one full-attention layer at load over a chunk of the
//! size it will serve and reads MLX's peak. The dense path measures the same
//! way (`dense_layer_mlx_working_buffer_probe`).
//!
//! The probe's input is the embedding of a token sequence that walks the
//! vocabulary. An expert layer's cost depends on how many different experts
//! the chunk routes to: a constant input sends every token to the same few
//! experts and reads a page far smaller than a real prompt's.

use super::Qwen3_5Model;
use super::layer_probe_shape::LayerProbeShape;
use crate::PerformanceAttribution;
use crate::qwen3_5_core::model::probe_token_ids::vocabulary_walking_token_ids;
use crate::qwen3_5_streaming::model::paged_execution_mode::Qwen3_5MoEPagedExecutionMode;
use text_generation::RequestDecoderStateStack;
use text_generation::execution_error::TextExecutionError;

/// One measured expert-layer prompt step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbedPromptLayerMlxWorkingBuffers {
    /// Tokens of the chunk the layer step ran over.
    pub span_token_count: u32,
    /// The step's peak beyond what was held before it, the state it wrote
    /// and the expert weights it read.
    pub mlx_working_buffer_bytes: u64,
    /// The same for one token through one layer, which is what a reply step
    /// is made of. Without it the ledger would know nothing about the reply
    /// phase when the first reply begins, and would keep nothing for it.
    pub reply_token_mlx_working_buffer_bytes: u64,
}

impl Qwen3_5Model {
    /// Measures a sparse model's prompt-layer MLX working buffers and remembers
    /// it. A dense model measures its own way and is left alone.
    ///
    /// A probe the limit cannot hold does not fail the load (#1902): the
    /// MLX working buffers stay unmeasured and the first real layer step
    /// measures it instead.
    pub fn probe_expert_layer_mlx_working_buffers(&self, probe_shape: LayerProbeShape) {
        if self.expert_pager.is_none() {
            return;
        }
        match self.measure_expert_layer_mlx_working_buffers(probe_shape) {
            Ok(probed_mlx_working_buffers) => {
                tracing::info!(
                    span_token_count = probed_mlx_working_buffers.span_token_count,
                    mlx_working_buffer_bytes = probed_mlx_working_buffers.mlx_working_buffer_bytes,
                    reply_token_mlx_working_buffer_bytes =
                        probed_mlx_working_buffers.reply_token_mlx_working_buffer_bytes,
                    "measured one expert layer's prompt MLX working buffers at load"
                );
                self.probed_prompt_layer_mlx_working_buffers
                    .set(Some(probed_mlx_working_buffers));
            }
            Err(probe_error) => tracing::warn!(
                error = %probe_error,
                "could not measure an expert layer's prompt MLX working buffers at load"
            ),
        }
    }

    /// The largest MLX working buffers either layer kind's prompt step needs at
    /// the probe shape.
    fn measure_expert_layer_mlx_working_buffers(
        &self,
        probe_shape: LayerProbeShape,
    ) -> Result<ProbedPromptLayerMlxWorkingBuffers, TextExecutionError> {
        let reply_token_shape = LayerProbeShape {
            prefill_chunk_token_count: 1,
            ..probe_shape
        };
        let mut mlx_working_buffer_bytes = 0_u64;
        let mut reply_token_mlx_working_buffer_bytes = 0_u64;
        for probe_layer_index in self.probe_layer_indexes() {
            mlx_working_buffer_bytes = mlx_working_buffer_bytes
                .max(self.measure_one_expert_layer_step(probe_layer_index, probe_shape)?);
            reply_token_mlx_working_buffer_bytes = reply_token_mlx_working_buffer_bytes
                .max(self.measure_one_expert_layer_step(probe_layer_index, reply_token_shape)?);
        }
        self.runtime
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
        Ok(ProbedPromptLayerMlxWorkingBuffers {
            span_token_count: u32::try_from(probe_shape.prefill_chunk_token_count)
                .unwrap_or(u32::MAX),
            mlx_working_buffer_bytes,
            reply_token_mlx_working_buffer_bytes,
        })
    }

    fn measure_one_expert_layer_step(
        &self,
        layer_index: usize,
        probe_shape: LayerProbeShape,
    ) -> Result<u64, TextExecutionError> {
        let token_count = i32::try_from(probe_shape.prefill_chunk_token_count).map_err(|_| {
            TextExecutionError::InvalidInput {
                description: "the MLX working-buffer probe token count exceeds the MLX range",
            }
        })?;
        let layer_weights = self
            .weights
            .decoder_layer_weights
            .get(layer_index)
            .ok_or(TextExecutionError::MissingDecoderLayerWeights { layer_index })?;
        let mut request_decoder_state =
            RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
                &self.astro_decoder_state_cache_layout,
                probe_shape.full_attention_kv_state_growth_tokens,
            )?;
        self.runtime
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
        let active_bytes_before = self.runtime.memory_snapshot()?.active_memory_bytes() as u64;
        let expert_storage_read_bytes_before_step = self
            .request_expert_storage_read_tally
            .expert_storage_read_byte_count();
        self.runtime.reset_peak_memory()?;
        // The input is made after the baseline: a real step holds its
        // hidden states too.
        let probe_input = self.embedding_lookup(&self.runtime.array_from_i32(
            &vocabulary_walking_token_ids(token_count, self.config.vocabulary_size()),
            &[1, token_count],
        )?)?;
        let layer_model_state = request_decoder_state.layer_mut(layer_index).ok_or(
            TextExecutionError::InvalidRequestDecoderState {
                layer_index,
                description: "the probe decoder state lacks the probed layer",
            },
        )?;
        let probe_output = self.forward_decoder_layer(
            &probe_input,
            token_count,
            0,
            layer_index,
            layer_weights,
            layer_model_state,
            None,
            None,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            &mut PerformanceAttribution::disabled(),
        )?;
        self.runtime.evaluate_arrays(&[&probe_output])?;
        let peak_bytes = self.runtime.memory_snapshot()?.peak_memory_bytes() as u64;
        // The expert weights the step read are what the ledger's slots hold,
        // and the state it wrote is context; neither is MLX working buffers.
        let streamed_expert_bytes = self
            .request_expert_storage_read_tally
            .expert_storage_read_byte_count()
            .saturating_sub(expert_storage_read_bytes_before_step);
        Ok(peak_bytes
            .saturating_sub(active_bytes_before)
            .saturating_sub(request_decoder_state.payload_byte_count())
            .saturating_sub(streamed_expert_bytes))
    }
}
