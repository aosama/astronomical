//! The sweep's forward: the reference pass records the rows entering the
//! decoder layers, and every probe resumes from the record nearest the
//! module it swapped instead of recomputing the unchanged layers before it.
//!
//! This is not in OptiQ, whose sweep reruns the whole model for every probe.
//! It changes no measured value: a resumed forward is the full forward's own
//! arithmetic from the full forward's own intermediate rows, and the engine's
//! forward is bit-reproducible. It removes about half of the decoder work of
//! a sweep: measured on Qwen3.5 0.8B, where the decoder is a little over half
//! of a probe's time, a 24-sequence sweep went from 1,178 to 811 seconds with
//! a byte-identical artifact.

use astronomical_quantization::error::QuantizationError;
use mlx_bridge::MlxArray;

use super::super::layer_input_reuse_plan::{
    first_affected_decoder_layer, recording_layer_stride, resume_record_position,
};
use super::super::simulated_quantization::mlx_step;
use super::{Qwen3_5SensitivityProbe, is_tied_head_key, probe_step};

/// How far up the decoder stack the currently swapped modules reach.
enum SwappedExtent {
    /// Nothing is swapped: this forward is a reference forward.
    Nothing,
    /// Every swapped module sits at or after this decoder layer.
    FromDecoderLayer(usize),
    /// A swap changes the rows entering the stack, or cannot be placed.
    WholeStack,
}

impl Qwen3_5SensitivityProbe {
    /// Logits at every position for one calibration sequence, reusing the
    /// reference forward's layer inputs when a swap leaves them valid. A
    /// sequence longer than one prompt chunk takes the chunked full forward:
    /// the records describe a single-chunk forward only.
    pub(super) fn sweep_forward_logits(
        &mut self,
        token_ids: &[u32],
    ) -> Result<MlxArray, QuantizationError> {
        if token_ids.len() > self.prefill_chunk_tokens.max(1) {
            return self.forward_logits(token_ids);
        }
        let final_hidden_states = match self.swapped_extent() {
            SwappedExtent::Nothing => self.record_reference_forward(token_ids)?,
            SwappedExtent::WholeStack => return self.forward_logits(token_ids),
            SwappedExtent::FromDecoderLayer(first_affected_layer) => {
                match self.resume_from_record(token_ids, first_affected_layer)? {
                    Some(final_hidden_states) => final_hidden_states,
                    None => return self.forward_logits(token_ids),
                }
            }
        };
        self.logits_from_final_hidden_states(&final_hidden_states)
    }

    /// The earliest decoder layer any current swap changes.
    fn swapped_extent(&self) -> SwappedExtent {
        let decoder_layer_count = self.model.config.layer_count() as usize;
        let mut earliest_affected_layer = None;
        for swapped_key in self.swapped_out_weights.keys() {
            // A tied head rides along with its embedding swap, which is
            // already a whole-stack swap under its own key.
            if is_tied_head_key(swapped_key) {
                continue;
            }
            let Some(location) = self.module_locations.get(swapped_key) else {
                return SwappedExtent::WholeStack;
            };
            let Some(affected_layer) = first_affected_decoder_layer(*location, decoder_layer_count)
            else {
                return SwappedExtent::WholeStack;
            };
            earliest_affected_layer = match earliest_affected_layer {
                Some(earliest_layer) if earliest_layer <= affected_layer => Some(earliest_layer),
                _ => Some(affected_layer),
            };
        }
        match earliest_affected_layer {
            Some(first_affected_layer) => SwappedExtent::FromDecoderLayer(first_affected_layer),
            None => SwappedExtent::Nothing,
        }
    }

    /// Runs the whole stack, keeps the layer inputs for later probes of the
    /// same sequence, and returns the rows leaving the last layer.
    fn record_reference_forward(
        &mut self,
        token_ids: &[u32],
    ) -> Result<MlxArray, QuantizationError> {
        let layer_stride = recording_layer_stride(
            self.model.config.layer_count() as usize,
            self.model.config.hidden_size() as usize,
            self.model.config.vocabulary_size() as usize,
        );
        let mut decoder_state = self.empty_probe_decoder_state()?;
        let recorded_layer_inputs =
            probe_step(self.model.forward_decoder_stack_recording_layer_inputs(
                token_ids,
                layer_stride,
                &mut decoder_state,
            ))?;
        let Some(final_record) = recorded_layer_inputs.last() else {
            return Err(QuantizationError::ProbeModelFailure {
                description: "the reference forward recorded nothing".to_owned(),
            });
        };
        let final_hidden_states = mlx_step(
            final_record.hidden_states.retain(),
            "retain the reference hidden states",
        )?;
        self.reference_layer_inputs
            .insert(token_ids.to_vec(), recorded_layer_inputs);
        Ok(final_hidden_states)
    }

    /// The rows leaving the last layer, recomputed from the record nearest
    /// the swapped layer; `None` when this sequence has no usable record.
    fn resume_from_record(
        &self,
        token_ids: &[u32],
        first_affected_layer: usize,
    ) -> Result<Option<MlxArray>, QuantizationError> {
        let Some(recorded_layer_inputs) = self.reference_layer_inputs.get(token_ids) else {
            return Ok(None);
        };
        let mut recorded_layer_indices = Vec::with_capacity(recorded_layer_inputs.len());
        for recorded_layer_input in recorded_layer_inputs {
            recorded_layer_indices.push(recorded_layer_input.decoder_layer_index);
        }
        let Some(resume_position) =
            resume_record_position(&recorded_layer_indices, first_affected_layer)
        else {
            return Ok(None);
        };
        let mut decoder_state = self.empty_probe_decoder_state()?;
        let final_hidden_states =
            probe_step(self.model.forward_decoder_stack_from_recorded_layer_input(
                &recorded_layer_inputs[resume_position],
                &mut decoder_state,
            ))?;
        Ok(Some(final_hidden_states))
    }
}
