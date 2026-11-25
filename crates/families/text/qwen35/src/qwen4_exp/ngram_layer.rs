//! The n-gram layer's forward: what Flash-Next adds to its residual lanes on
//! one layer, before that layer's attention.
//!
//! Each token's sixteen gathered rows become a key per lane and one value.
//! Every lane scores its own key against its normalized state, the score's
//! square root (sign kept) gates the value into that lane, and a short
//! dilated convolution over the gated values adds a little of the recent
//! past. The reference module is `Qwen4ExpPLELayer`.

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::configuration::Qwen4ExpArchitecture;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;

use super::centered_norm::grouped_rms_norm;
use super::layer_weights::NgramLayerWeights;
use super::request_state::Qwen4ExpRequestState;

/// Scores below this magnitude gate as if they were it, so the square root
/// never sees zero; the reference's floor.
const MINIMUM_GATE_MAGNITUDE: f32 = 1e-6;

impl Qwen3_5ModelBase {
    /// `lanes` plus the n-gram layer's contribution for the forward's tokens.
    pub fn add_ngram_contribution(
        &self,
        lanes: &MlxArray,
        ngram_layer: &NgramLayerWeights,
        request_state: &mut Qwen4ExpRequestState,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let token_count = lanes.shape()[1];
        let embeddings = self.gathered_ngram_rows(request_state, token_count)?;
        let gated_values =
            self.gated_ngram_values(lanes, &embeddings, ngram_layer, architecture)?;
        let convolved =
            self.ngram_convolution(&gated_values, ngram_layer, request_state, architecture)?;
        let contribution = self.runtime.add(&gated_values, &convolved)?;
        Ok(self.runtime.add(lanes, &contribution)?)
    }

    /// The forward's table rows, hashed from its tokens and the history
    /// before them; advances the history.
    fn gathered_ngram_rows(
        &self,
        request_state: &mut Qwen4ExpRequestState,
        token_count: i32,
    ) -> Result<MlxArray, TextExecutionError> {
        let ngram_table = self
            .ngram_table
            .as_ref()
            .ok_or(TextExecutionError::MissingTensor {
                tensor_name: "ple.ple_embedding.ngram_embedding".to_owned(),
            })?;
        if request_state.forward_token_ids.len() != token_count as usize {
            return Err(TextExecutionError::InvalidInput {
                description: "the n-gram layer's token ids do not match the forward",
            });
        }
        let hash_tables = ngram_table.hash_tables();
        let row_ids = hash_tables.row_ids(
            &request_state.ngram_context,
            &request_state.forward_token_ids,
        );
        request_state.ngram_context = hash_tables.next_context(
            &request_state.ngram_context,
            &request_state.forward_token_ids,
        );
        ngram_table.gather_rows(&self.runtime, &row_ids, token_count)
    }

    /// Each lane's gate times the value, as `[1, tokens, lanes * hidden]`.
    fn gated_ngram_values(
        &self,
        lanes: &MlxArray,
        embeddings: &MlxArray,
        ngram_layer: &NgramLayerWeights,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let hidden_size = self.config.hidden_size() as i32;
        let epsilon = f32::from_bits(self.config.rms_norm_epsilon_bits());
        let lane_shape = [
            1,
            lanes.shape()[1],
            architecture.residual_lane_count as i32,
            hidden_size,
        ];
        let keys = grouped_rms_norm(
            runtime,
            &self.quantized_linear(embeddings, &ngram_layer.key_projection)?,
            &ngram_layer.key_norm_scale,
            hidden_size,
            epsilon,
        )?;
        let queries = grouped_rms_norm(
            runtime,
            lanes,
            &ngram_layer.query_norm_scale,
            hidden_size,
            epsilon,
        )?;
        let gates = self.ngram_lane_gates(&keys, &queries, &lane_shape)?;
        let values = runtime.expand_dims(
            &self.quantized_linear(embeddings, &ngram_layer.value_projection)?,
            -2,
        )?;
        let gated = runtime.multiply(&gates, &values)?;
        Ok(runtime.reshape(&gated, &lanes.shape())?)
    }

    /// Each lane's gate: its key's agreement with its state, as
    /// `sigmoid(sign(s) * sqrt(max(|s|, floor)))` with `s = k·q / sqrt(d)`.
    fn ngram_lane_gates(
        &self,
        keys: &MlxArray,
        queries: &MlxArray,
        lane_shape: &[i32],
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let hidden_size = lane_shape[3];
        let scores = runtime.sum_axis(
            &runtime.multiply(
                &runtime.reshape(keys, lane_shape)?,
                &runtime.reshape(queries, lane_shape)?,
            )?,
            -1,
            true,
        )?;
        let scores = runtime.multiply_scalar(&scores, (hidden_size as f32).sqrt().recip())?;
        let floor = runtime.full(&[], MINIMUM_GATE_MAGNITUDE, scores.dtype())?;
        let root_magnitude = runtime.sqrt(&runtime.maximum(&runtime.abs(&scores)?, &floor)?)?;
        Ok(runtime.sigmoid(&runtime.multiply(&runtime.sign(&scores)?, &root_magnitude)?)?)
    }

    /// `silu` of the depthwise dilated convolution over the normalized gated
    /// values, continuing from the window the previous forward left.
    fn ngram_convolution(
        &self,
        gated_values: &MlxArray,
        ngram_layer: &NgramLayerWeights,
        request_state: &mut Qwen4ExpRequestState,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let lane_width = gated_values.shape()[2];
        // The kernel's taps sit `ngram_size` tokens apart.
        let dilation = architecture.ngram_size as i32;
        let window_length = (architecture.ngram_convolution_kernel_size as i32 - 1) * dilation;
        let normalized = grouped_rms_norm(
            runtime,
            gated_values,
            &ngram_layer.convolution_norm_scale,
            self.config.hidden_size() as i32,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let window = match request_state.ngram_convolution_window.take() {
            Some(window) => window,
            None => runtime.zeros(&[1, window_length, lane_width], normalized.dtype())?,
        };
        let convolution_input = runtime.concatenate_axis(&[&window, &normalized], 1)?;
        let input_length = convolution_input.shape()[1];
        request_state.ngram_convolution_window = Some(runtime.slice(
            &convolution_input,
            &[0, input_length - window_length, 0],
            &[1, input_length, lane_width],
            &[1, 1, 1],
        )?);
        let convolution_weight =
            runtime.astype(&ngram_layer.convolution_weight, normalized.dtype())?;
        let convolved = runtime.conv1d(
            &convolution_input,
            &convolution_weight,
            1,
            0,
            dilation,
            lane_width,
        )?;
        Ok(runtime.silu(&convolved)?)
    }
}
