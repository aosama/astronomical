//! Entry points the dense streaming speed ladder times (issue #1985).
//!
//! Performance work starts at the bottom of the call stack: how fast one
//! decoder layer comes off the SSD through the bounded-range reader, then
//! what one streamed layer costs read and run, before a whole prompt. These
//! entry points do exactly that work and nothing else; the measurement
//! harness owns the clock and the cold file cache.

use mlx_c_rust::MlxArray;

use super::Qwen3_5Model;
use crate::PerformanceAttribution;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::model::bound_weight_dtype_flow::embedded_hidden_state_dtype;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::model::paged_execution_mode::Qwen3_5MoEPagedExecutionMode;

impl Qwen3_5Model {
    /// Reads one decoder layer through the streamer's bounded-range reader
    /// and waits until its bytes are in memory, returning their count.
    #[doc(hidden)]
    pub fn read_dense_layer_for_measurement(
        &self,
        layer_index: usize,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        let dense_layer_streamer = self.dense_layer_streamer()?;
        let layer_weights = dense_layer_streamer.read_layer(
            &self.runtime,
            &self.config,
            self.weights.raw_hugging_face_norm_offsets_apply(),
            layer_index,
        )?;
        let mut layer_arrays = Vec::new();
        layer_weights.append_array_references(&mut layer_arrays);
        self.runtime.evaluate_arrays(&layer_arrays)?;
        Ok(dense_layer_streamer
            .layer_payload_bytes()
            .get(layer_index)
            .copied()
            .unwrap_or(0))
    }

    /// Reads one decoder layer and runs it over `token_count` tokens from
    /// position zero with fresh decoder state, waiting for the result: one
    /// streamed layer's cost in a prompt (`token_count` above one) or in a
    /// decode step (one).
    #[doc(hidden)]
    pub fn run_streamed_dense_layer_for_measurement(
        &self,
        layer_index: usize,
        token_count: usize,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let layer_weights = self.dense_layer_streamer()?.read_layer(
            &self.runtime,
            &self.config,
            self.weights.raw_hugging_face_norm_offsets_apply(),
            layer_index,
        )?;
        let signed_token_count =
            i32::try_from(token_count).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "the measured token count exceeds the MLX range",
            })?;
        let mut request_decoder_state =
            RequestDecoderStateStack::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                &self.decoder_cache_layout,
                signed_token_count.max(1),
            )?;
        let layer_input: MlxArray = self.runtime.full(
            &[1, signed_token_count, self.config.hidden_size() as i32],
            0.0,
            embedded_hidden_state_dtype(&self.weights)?,
        )?;
        let layer_model_state = request_decoder_state.layer_mut(layer_index).ok_or(
            Qwen3_5ExecutionError::InvalidRequestDecoderState {
                layer_index,
                description: "the measured decoder state lacks the layer",
            },
        )?;
        let layer_output = self.forward_decoder_layer(
            &layer_input,
            signed_token_count,
            0,
            layer_index,
            &layer_weights,
            layer_model_state,
            None,
            None,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            &mut PerformanceAttribution::disabled(),
        )?;
        self.runtime.evaluate_arrays(&[&layer_output])?;
        Ok(())
    }

    fn dense_layer_streamer(
        &self,
    ) -> Result<&super::dense_layer_streamer::Qwen3_5DenseLayerStreamer, Qwen3_5ExecutionError>
    {
        self.dense_layer_streamer
            .as_ref()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "only a dense model on the streaming engine reads its layers",
            })
    }
}
