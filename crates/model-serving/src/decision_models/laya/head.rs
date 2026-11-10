//! The Laya decision head: hidden states in, one logit per option out.
//!
//! The head adds a learned vector for the question's type to every position,
//! runs a few pre-norm transformer layers over the whole sequence, and scores
//! the hidden state at each option's marker with a small feed-forward network.
//! It mirrors `torch.nn.TransformerEncoderLayer(norm_first=True)` with its
//! default ReLU feed-forward, which is what the checkpoints were trained with.
//!
//! The checkpoints also carry an "act head" that was meant to say whether to
//! act on an answer or escalate it. Upstream reports it carries no usable
//! signal, so it is neither loaded nor reported.

use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};
use mlx_bridge::{MlxArray, MlxDtype};

use crate::decision_models::laya::configuration::LayaConfiguration;
use crate::modernbert::attention_mask::key_padding_mask_values;
use crate::modernbert::token_batch::PaddedTokenBatch;
use crate::modernbert::trunk::{ModernBertTrunkError, slice_stacked_projection};

/// LayerNorm epsilon of every norm in the head: PyTorch's default, which the
/// checkpoints were built with.
const HEAD_LAYER_NORM_EPSILON: f32 = 1e-5;

/// One loaded head: the runtime, the artifact's tensors and its settings.
pub(crate) struct LayaDecisionHead<'model> {
    pub(crate) runtime: &'model MlxRuntime,
    pub(crate) tensors: &'model MlxSafetensors,
    pub(crate) configuration: &'model LayaConfiguration,
}

/// What the head needs to know about each row of one forward pass.
pub(crate) struct HeadPassRows<'pass> {
    pub(crate) token_batch: &'pass PaddedTokenBatch,
    /// Each row's question type, as the row index of the type embedding.
    pub(crate) question_type_indices: &'pass [u32],
    /// Each row's option marker positions, in canonical option order.
    pub(crate) option_marker_positions: &'pass [&'pass [usize]],
}

impl LayaDecisionHead<'_> {
    /// Scores every option of every row and returns the logits per row.
    pub(crate) fn option_logits(
        &self,
        trunk_hidden_states: &MlxArray,
        pass_rows: &HeadPassRows<'_>,
    ) -> Result<Vec<Vec<f32>>, ModernBertTrunkError> {
        let mut head_states = self.add_question_type(trunk_hidden_states, pass_rows)?;
        let key_padding_mask = self.key_padding_mask(pass_rows.token_batch)?;
        for head_layer_index in 0..self.configuration.head_layer_count {
            head_states =
                self.encode_head_layer(&head_states, head_layer_index, key_padding_mask.as_ref())?;
        }
        let marker_states = self.gather_marker_states(&head_states, pass_rows)?;
        let padded_logits = self.score_markers(&marker_states)?;
        let flat_logits = self
            .runtime
            .astype(&padded_logits, MlxDtype::Float32)?
            .to_vec_f32()?;
        let widest_option_count = widest_option_count(pass_rows);
        let mut row_logits = Vec::with_capacity(pass_rows.option_marker_positions.len());
        for (row_index, marker_positions) in pass_rows.option_marker_positions.iter().enumerate() {
            let row_start = row_index * widest_option_count;
            row_logits.push(flat_logits[row_start..row_start + marker_positions.len()].to_vec());
        }
        Ok(row_logits)
    }

    /// Adds each row's question-type vector to every position of that row.
    fn add_question_type(
        &self,
        trunk_hidden_states: &MlxArray,
        pass_rows: &HeadPassRows<'_>,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let row_count = pass_rows.question_type_indices.len() as i32;
        let mut type_indices = Vec::with_capacity(pass_rows.question_type_indices.len());
        for question_type_index in pass_rows.question_type_indices {
            type_indices.push(*question_type_index as i32);
        }
        let type_index_array = self.runtime.array_from_i32(&type_indices, &[row_count])?;
        let type_table = self.float32_tensor("type_emb.weight")?;
        let type_vectors = self.runtime.take_axis(&type_table, &type_index_array, 0)?;
        let broadcast_type_vectors = self.runtime.reshape(
            &type_vectors,
            &[row_count, 1, self.configuration.encoder.hidden_size as i32],
        )?;
        Ok(self
            .runtime
            .add(trunk_hidden_states, &broadcast_type_vectors)?)
    }

    /// The additive mask that hides padding from the head's attention, or
    /// `None` when no row is padded.
    fn key_padding_mask(
        &self,
        token_batch: &PaddedTokenBatch,
    ) -> Result<Option<MlxArray>, ModernBertTrunkError> {
        if !token_batch.has_padding() {
            return Ok(None);
        }
        // Both counts fit `i32`: `PaddedTokenBatch` refuses a batch that does not.
        Ok(Some(self.runtime.array_from_f32(
            &key_padding_mask_values(token_batch),
            &[
                token_batch.sequence_count() as i32,
                1,
                1,
                token_batch.padded_length() as i32,
            ],
        )?))
    }

    /// One pre-norm layer: `x + attention(norm1(x))`, then `x + feed_forward(norm2(x))`.
    fn encode_head_layer(
        &self,
        head_states: &MlxArray,
        head_layer_index: u32,
        key_padding_mask: Option<&MlxArray>,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let layer_prefix = format!("head.layers.{head_layer_index}");
        let attention_input = self.layer_norm(&format!("{layer_prefix}.norm1"), head_states)?;
        let attention_output =
            self.self_attention(&attention_input, &layer_prefix, key_padding_mask)?;
        let attended_states = self.runtime.add(head_states, &attention_output)?;
        let feed_forward_input =
            self.layer_norm(&format!("{layer_prefix}.norm2"), &attended_states)?;
        let expanded =
            self.quantized_linear(&format!("{layer_prefix}.linear1"), &feed_forward_input)?;
        let zero = self.runtime.zeros(&[1], MlxDtype::Float32)?;
        let rectified = self.runtime.maximum(&expanded, &zero)?;
        let feed_forward_output =
            self.quantized_linear(&format!("{layer_prefix}.linear2"), &rectified)?;
        Ok(self.runtime.add(&attended_states, &feed_forward_output)?)
    }

    /// `torch.nn.MultiheadAttention` with one packed projection producing
    /// query, key and value, in that order along the last axis.
    fn self_attention(
        &self,
        attention_input: &MlxArray,
        layer_prefix: &str,
        key_padding_mask: Option<&MlxArray>,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let input_shape = attention_input.shape();
        let (row_count, padded_length) = (input_shape[0], input_shape[1]);
        let hidden_size = self.configuration.encoder.hidden_size as i32;
        let head_count = self.configuration.head_attention_head_count() as i32;
        let head_width = hidden_size / head_count;
        let packed_projection = self.quantized_linear(
            &format!("{layer_prefix}.self_attn.in_proj"),
            attention_input,
        )?;
        let split_projection = self.runtime.reshape(
            &packed_projection,
            &[row_count, padded_length, 3, head_count, head_width],
        )?;
        let reordered_projection = self
            .runtime
            .transpose_axes(&split_projection, &[0, 3, 2, 1, 4])?;
        let queries = slice_stacked_projection(self.runtime, &reordered_projection, 0)?;
        let keys = slice_stacked_projection(self.runtime, &reordered_projection, 1)?;
        let values = slice_stacked_projection(self.runtime, &reordered_projection, 2)?;
        let attention_scale = (head_width as f32).powf(-0.5);
        let attended = match key_padding_mask {
            Some(attention_mask) => self.runtime.masked_scaled_dot_product_attention(
                &queries,
                &keys,
                &values,
                attention_scale,
                attention_mask,
            )?,
            None => self.runtime.scaled_dot_product_attention(
                &queries,
                &keys,
                &values,
                attention_scale,
            )?,
        };
        let sequence_ordered = self.runtime.transpose_axes(&attended, &[0, 2, 1, 3])?;
        let merged_heads = self
            .runtime
            .reshape(&sequence_ordered, &[row_count, padded_length, hidden_size])?;
        self.quantized_linear(&format!("{layer_prefix}.self_attn.out_proj"), &merged_heads)
    }

    /// The hidden state at each option marker: `[rows, widest option count, hidden]`.
    ///
    /// Rows with fewer options repeat position zero in their unused slots;
    /// those slots are dropped when the logits are read back.
    fn gather_marker_states(
        &self,
        head_states: &MlxArray,
        pass_rows: &HeadPassRows<'_>,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let widest_option_count = widest_option_count(pass_rows);
        let row_count = pass_rows.option_marker_positions.len();
        let mut marker_indices = vec![0_i32; row_count * widest_option_count];
        for (row_index, marker_positions) in pass_rows.option_marker_positions.iter().enumerate() {
            for (option_index, marker_position) in marker_positions.iter().enumerate() {
                marker_indices[row_index * widest_option_count + option_index] =
                    *marker_position as i32;
            }
        }
        let marker_index_array = self.runtime.array_from_i32(
            &marker_indices,
            &[row_count as i32, widest_option_count as i32, 1],
        )?;
        Ok(self
            .runtime
            .take_along_axis(head_states, &marker_index_array, 1)?)
    }

    /// `LayerNorm → Linear → GELU → Linear(1)`, giving `[rows, widest option count]`.
    fn score_markers(&self, marker_states: &MlxArray) -> Result<MlxArray, ModernBertTrunkError> {
        let normalized = self.layer_norm("scorer.layers.0", marker_states)?;
        let hidden = self.dense_linear("scorer.layers.1", &normalized)?;
        let activated = self.runtime.gelu(&hidden)?;
        let single_logit = self.dense_linear("scorer.layers.3", &activated)?;
        let logit_shape = single_logit.shape();
        Ok(self
            .runtime
            .reshape(&single_logit, &[logit_shape[0], logit_shape[1]])?)
    }

    fn layer_norm(
        &self,
        norm_prefix: &str,
        input_states: &MlxArray,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let weight = self.float32_tensor(&format!("{norm_prefix}.weight"))?;
        let bias = self.float32_tensor(&format!("{norm_prefix}.bias"))?;
        Ok(self
            .runtime
            .layer_norm(input_states, &weight, &bias, HEAD_LAYER_NORM_EPSILON)?)
    }

    /// An affine-quantized linear layer with a dense bias.
    fn quantized_linear(
        &self,
        linear_prefix: &str,
        activations: &MlxArray,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let quantized_weights = self.tensor(&format!("{linear_prefix}.weight"))?;
        let scales = self.tensor(&format!("{linear_prefix}.scales"))?;
        let biases = self.tensor(&format!("{linear_prefix}.biases"))?;
        let projected = self.runtime.quantized_matmul_affine(
            activations,
            &quantized_weights,
            &scales,
            &biases,
            true,
            self.configuration.encoder.quantization_group_size as i32,
            self.configuration.encoder.quantization_bits as i32,
        )?;
        let bias = self.float32_tensor(&format!("{linear_prefix}.bias"))?;
        Ok(self.runtime.add(&projected, &bias)?)
    }

    /// A linear layer whose `[outputs, inputs]` weight is stored dense.
    fn dense_linear(
        &self,
        linear_prefix: &str,
        activations: &MlxArray,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let weight = self.float32_tensor(&format!("{linear_prefix}.weight"))?;
        let transposed_weight = self.runtime.transpose_axes(&weight, &[1, 0])?;
        let projected = self.runtime.matmul(activations, &transposed_weight)?;
        let bias = self.float32_tensor(&format!("{linear_prefix}.bias"))?;
        Ok(self.runtime.add(&projected, &bias)?)
    }

    /// A tensor widened to single precision, the precision the head runs in.
    fn float32_tensor(&self, tensor_name: &str) -> Result<MlxArray, ModernBertTrunkError> {
        let stored_tensor = self.tensor(tensor_name)?;
        Ok(self.runtime.astype(&stored_tensor, MlxDtype::Float32)?)
    }

    fn tensor(&self, tensor_name: &str) -> Result<MlxArray, ModernBertTrunkError> {
        match self.tensors.tensor(tensor_name) {
            Ok(tensor) => Ok(tensor),
            Err(tensor_error) => Err(ModernBertTrunkError::UnavailableTensor {
                tensor_name: tensor_name.to_owned(),
                reason: tensor_error.to_string(),
            }),
        }
    }
}

fn widest_option_count(pass_rows: &HeadPassRows<'_>) -> usize {
    let mut widest_option_count = 1;
    for marker_positions in pass_rows.option_marker_positions {
        widest_option_count = widest_option_count.max(marker_positions.len());
    }
    widest_option_count
}

#[cfg(test)]
#[path = "head_tests.rs"]
mod tests;
