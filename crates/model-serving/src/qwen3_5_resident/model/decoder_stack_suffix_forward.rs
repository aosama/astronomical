//! Forwards over a suffix of the decoder stack, for a caller that replays
//! one prompt many times with a single layer's weights changed between
//! replays (the quantization sensitivity sweep).
//!
//! A decoder layer's output depends only on the rows entering it and on the
//! weights from that layer onward. A replay that changed layer `L` therefore
//! reproduces the unchanged forward exactly up to layer `L`, and recomputing
//! those layers is pure waste: on average half of the stack per probe, and
//! the sweep runs two probes per module per calibration sequence. The first
//! forward records the rows entering chosen layers; later forwards resume
//! from the nearest record at or before the changed layer. Both entries run
//! the same per-layer forward as the ordinary graph, so a resumed forward is
//! the full forward's arithmetic on the full forward's own intermediate
//! values, not an approximation of it.
//!
//! Only a fresh, single-chunk forward from position zero is supported: a
//! resumed forward starts with an empty decoder state for the layers it
//! runs, which is what those layers held at that point of a fresh forward.

use crate::performance_attribution::PerformanceAttribution;
use crate::qwen3_5_core::decoder::request_state::RequestDecoderStateStack;
use mlx_bridge::MlxArray;

use super::{Qwen3_5ExecutionError, Qwen3_5ResidentModel};

/// The rows entering one decoder layer during a recorded forward. A record
/// whose index equals the layer count holds the rows leaving the last layer,
/// before the final normalization.
pub(crate) struct RecordedDecoderLayerInput {
    pub(crate) decoder_layer_index: usize,
    pub(crate) hidden_states: MlxArray,
}

/// Which layer inputs a recording forward keeps, and where they go.
struct LayerInputRecording<'records> {
    /// Keep the input of every layer whose index is a multiple of this.
    layer_stride: usize,
    recorded_layer_inputs: &'records mut Vec<RecordedDecoderLayerInput>,
}

impl Qwen3_5ResidentModel {
    /// Runs the whole decoder stack over `token_ids` from position zero and
    /// returns the recorded layer inputs, evaluated: one for every layer
    /// whose index is a multiple of `recording_layer_stride` (so always
    /// layer zero, the embeddings), then the rows leaving the last layer.
    pub(crate) fn forward_decoder_stack_recording_layer_inputs(
        &self,
        token_ids: &[u32],
        recording_layer_stride: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<Vec<RecordedDecoderLayerInput>, Qwen3_5ExecutionError> {
        let (token_indices, token_count) =
            self.validated_token_indices(token_ids, 0, None, request_decoder_state)?;
        let embedded_tokens = self.embedding_lookup(&token_indices)?;
        let mut recorded_layer_inputs = Vec::new();
        let final_hidden_states = self.run_decoder_layers_from(
            embedded_tokens,
            0,
            token_count,
            request_decoder_state,
            Some(LayerInputRecording {
                layer_stride: recording_layer_stride.max(1),
                recorded_layer_inputs: &mut recorded_layer_inputs,
            }),
        )?;
        recorded_layer_inputs.push(RecordedDecoderLayerInput {
            decoder_layer_index: self.config.layer_count() as usize,
            hidden_states: final_hidden_states,
        });
        let mut evaluation_arrays = Vec::with_capacity(recorded_layer_inputs.len());
        for recorded_layer_input in &recorded_layer_inputs {
            evaluation_arrays.push(&recorded_layer_input.hidden_states);
        }
        self.runtime.evaluate_arrays(&evaluation_arrays)?;
        Ok(recorded_layer_inputs)
    }

    /// Runs the decoder layers from `recorded_layer_input`'s layer to the
    /// end and returns the rows leaving the last layer, evaluated. A record
    /// of the last layer's output is returned as it is.
    pub(crate) fn forward_decoder_stack_from_recorded_layer_input(
        &self,
        recorded_layer_input: &RecordedDecoderLayerInput,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let hidden_state_shape = recorded_layer_input.hidden_states.shape();
        let [_, token_count, _] = hidden_state_shape[..] else {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "a recorded decoder layer input must have three axes",
            });
        };
        let final_hidden_states = self.run_decoder_layers_from(
            recorded_layer_input.hidden_states.retain()?,
            recorded_layer_input.decoder_layer_index,
            token_count,
            request_decoder_state,
            None,
        )?;
        self.runtime.evaluate_arrays(&[&final_hidden_states])?;
        Ok(final_hidden_states)
    }

    /// The ordinary graph's layer loop over `first_layer_index..`, without
    /// the vocabulary head and without intermediate graph submission: the
    /// caller evaluates once, and no other graphics work is waiting on a
    /// conversion.
    fn run_decoder_layers_from(
        &self,
        mut hidden_states: MlxArray,
        first_layer_index: usize,
        token_count: i32,
        request_decoder_state: &mut RequestDecoderStateStack,
        mut layer_input_recording: Option<LayerInputRecording<'_>>,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        let decoder_layer_count = self.config.layer_count() as usize;
        for layer_index in first_layer_index..decoder_layer_count {
            if let Some(recording) = &mut layer_input_recording
                && layer_index.is_multiple_of(recording.layer_stride)
            {
                recording
                    .recorded_layer_inputs
                    .push(RecordedDecoderLayerInput {
                        decoder_layer_index: layer_index,
                        hidden_states: hidden_states.retain()?,
                    });
            }
            let decoder_layer_weights = self
                .weights
                .decoder_layer_weights
                .get(layer_index)
                .ok_or(Qwen3_5ExecutionError::MissingDecoderLayerWeights { layer_index })?;
            let layer_model_state = request_decoder_state.layer_mut(layer_index).ok_or(
                Qwen3_5ExecutionError::InvalidRequestDecoderState {
                    layer_index,
                    description: "decoder layer entry is missing",
                },
            )?;
            hidden_states = self.forward_decoder_layer(
                &hidden_states,
                token_count,
                0,
                layer_index,
                decoder_layer_weights,
                layer_model_state,
                None,
                None,
                &mut disabled_performance_attribution,
            )?;
        }
        Ok(hidden_states)
    }
}
