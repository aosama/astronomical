//! Measures the MLX working buffers one streamed dense layer's forward needs.
//!
//! The dense layer plan reserves MLX working buffers beside the kept layers and
//! the two layer slots (issue #1985). Rather than assume an amount, the
//! engine runs one linear-attention and one full-attention layer at load, at
//! the prefill chunk size it will serve, and reads MLX's peak: the transient
//! beyond the layer's own weights and the decoder state it writes.

use mlx_bridge::MlxArray;

use super::Qwen3_5Model;
use super::layer_probe_shape::LayerProbeShape;
use crate::PerformanceAttribution;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::model::bound_weight_dtype_flow::embedded_hidden_state_dtype;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::model::paged_execution_mode::Qwen3_5MoEPagedExecutionMode;

impl Qwen3_5Model {
    /// The largest transient either layer kind's forward adds at the probe
    /// shape, beyond the layer's weights and the state it writes.
    ///
    /// A zero input is enough: the two kinds allocate differently, but how
    /// much does not depend on the values.
    pub(crate) fn measure_dense_layer_mlx_working_buffers(
        &self,
        probe_shape: LayerProbeShape,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        let mut mlx_working_buffer_bytes = 0_u64;
        for probe_layer_index in self.probe_layer_indexes() {
            mlx_working_buffer_bytes = mlx_working_buffer_bytes
                .max(self.measure_one_layer_mlx_working_buffers(probe_layer_index, probe_shape)?);
        }
        self.runtime
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
        Ok(mlx_working_buffer_bytes)
    }

    fn measure_one_layer_mlx_working_buffers(
        &self,
        layer_index: usize,
        probe_shape: LayerProbeShape,
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
                description: "the MLX working-buffer probe token count exceeds the MLX range",
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
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
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
        // charges it as context, so it is not MLX working buffers. The peak
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
