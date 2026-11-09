//! Measures the working memory one streamed dense layer's forward needs.
//!
//! The dense layer plan reserves working memory beside the kept layers and
//! the two layer slots (issue #1985). Rather than assume an amount, the
//! engine runs one linear-attention and one full-attention layer at load, at
//! the prefill chunk size it will serve, and reads MLX's peak: the transient
//! beyond the layer's own weights and the decoder state it writes.

use mlx_c_rust::MlxArray;

use super::Qwen3_5Model;
use super::decoder_cache_dtype_flow::embedded_hidden_state_dtype;
use crate::PerformanceAttribution;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::model::paged_execution_mode::Qwen3_5MoEPagedExecutionMode;

/// The forward shape the engine will serve, which the probe reproduces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseLayerProbeShape {
    /// The largest prefill chunk the engine's chunk sizer will submit.
    pub prefill_chunk_token_count: usize,
    /// The attention slab granularity requests allocate state with.
    pub full_attention_kv_state_growth_tokens: i32,
}

impl Qwen3_5Model {
    /// The largest transient either layer kind's forward adds at the probe
    /// shape, beyond the layer's weights and the state it writes.
    ///
    /// A zero input is enough: the two kinds allocate differently, but how
    /// much does not depend on the values.
    pub(crate) fn measure_dense_layer_working_memory(
        &self,
        probe_shape: DenseLayerProbeShape,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        let mut working_memory_bytes = 0_u64;
        for probe_layer_index in self.probe_layer_indexes() {
            working_memory_bytes = working_memory_bytes
                .max(self.measure_one_layer_working_memory(probe_layer_index, probe_shape)?);
        }
        self.runtime
            .synchronize_gpu_stream_and_clear_allocator_cache()?;
        Ok(working_memory_bytes)
    }

    /// The first linear-attention layer and the first full-attention layer.
    fn probe_layer_indexes(&self) -> Vec<usize> {
        let layer_count = self.config.layer_count() as usize;
        let mut probe_layer_indexes = Vec::with_capacity(2);
        for wants_full_attention in [false, true] {
            for layer_index in 0..layer_count {
                if self.config.decoder_layer_is_full_attention(layer_index) == wants_full_attention
                {
                    probe_layer_indexes.push(layer_index);
                    break;
                }
            }
        }
        probe_layer_indexes
    }

    fn measure_one_layer_working_memory(
        &self,
        layer_index: usize,
        probe_shape: DenseLayerProbeShape,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return Ok(0);
        };
        let layer_weights = dense_layer_streamer.read_layer(
            &self.runtime,
            &self.config,
            self.weights.raw_hugging_face_norm_offsets_apply(),
            layer_index,
        )?;
        let token_count = i32::try_from(probe_shape.prefill_chunk_token_count).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "the working-memory probe token count exceeds the MLX range",
            }
        })?;
        let mut request_decoder_state =
            RequestDecoderStateStack::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                &self.decoder_cache_layout,
                probe_shape.full_attention_kv_state_growth_tokens,
            )?;
        let mut layer_arrays = Vec::new();
        layer_weights.append_array_references(&mut layer_arrays);
        self.runtime.evaluate_arrays(&layer_arrays)?;
        self.runtime
            .synchronize_gpu_stream_and_clear_allocator_cache()?;
        let active_bytes_before = self.runtime.memory_snapshot()?.active_memory_bytes() as u64;
        self.runtime.reset_peak_memory()?;
        // The input is made after the baseline: a real forward holds its
        // hidden states too, about 34 MB at 4,096 tokens on the 9B model.
        let probe_input = self.runtime.full(
            &[1, token_count, self.config.hidden_size() as i32],
            0.0,
            embedded_hidden_state_dtype(&self.weights)?,
        )?;
        let probe_output = self.probe_forward(
            &probe_input,
            token_count,
            layer_index,
            &layer_weights,
            &mut request_decoder_state,
        )?;
        self.runtime.evaluate_arrays(&[&probe_output])?;
        let peak_bytes = self.runtime.memory_snapshot()?.peak_memory_bytes() as u64;
        // The layer's own state persists past the forward and admission
        // charges it as context, so it is not working memory. The peak
        // includes the input, which the forward's hidden states stand for.
        Ok(peak_bytes
            .saturating_sub(active_bytes_before)
            .saturating_sub(request_decoder_state.payload_byte_count()))
    }

    fn probe_forward(
        &self,
        probe_input: &MlxArray,
        token_count: i32,
        layer_index: usize,
        layer_weights: &Qwen3_5DecoderLayerWeights,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let layer_model_state = request_decoder_state.layer_mut(layer_index).ok_or(
            Qwen3_5ExecutionError::InvalidRequestDecoderState {
                layer_index,
                description: "the probe decoder state lacks the probed layer",
            },
        )?;
        self.forward_decoder_layer(
            probe_input,
            token_count,
            0,
            layer_index,
            layer_weights,
            layer_model_state,
            None,
            None,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            &mut PerformanceAttribution::disabled(),
        )
    }
}
