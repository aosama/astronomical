//! The ModernBERT trunk: token identifiers in, final hidden states out.
//!
//! One call builds one lazy bidirectional-encoder graph for a batch of padded
//! sequences and returns `[sequence_count, padded_length, hidden_size]` hidden
//! states after the final LayerNorm. What a consumer does with them is its own
//! concern: the embedding engine pools them into one vector, and a decision
//! model reads them at its option markers.
//!
//! Layout contract for an MLX ModernBERT artifact: fused `Wqkv` produces
//! query, key, then value; global layers (index divisible by
//! `global_attn_every_n_layers`) attend across the whole sequence; local
//! layers use a symmetric window of `local_attention // 2` positions on each
//! side; every layer ends with a gated MLP whose activation is exact GELU.
//!
//! A batch whose rows all have the same length builds exactly the graph a
//! single sequence always built, so an unpadded forward pass is unchanged by
//! batching. Only a batch that carries padding adds the padding masks.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError, MlxSafetensors};
use mlx_c_rust::{MlxArray, MlxCError, MlxDtype};
use thiserror::Error;

use crate::modernbert::attention_mask::{
    HIDDEN_POSITION_MASK_VALUE, key_padding_mask_values, local_window_hides_any_pair,
    local_window_mask_values,
};
use crate::modernbert::configuration::ModernBertConfiguration;
use crate::modernbert::token_batch::PaddedTokenBatch;
use crate::modernbert::weight_layout::{ModernBertWeightLayout, TokenEmbeddingStorage};

const QUANTIZED_WEIGHTS_TAIL: &str = ".weight";
const QUANTIZED_SCALES_TAIL: &str = ".scales";
const QUANTIZED_BIASES_TAIL: &str = ".biases";

/// Why a trunk forward pass could not be built.
#[derive(Debug, Error)]
pub(crate) enum ModernBertTrunkError {
    #[error("{0}")]
    Runtime(MlxRuntimeError),
    #[error("tensor {tensor_name} is unavailable: {reason}")]
    UnavailableTensor { tensor_name: String, reason: String },
}

impl From<MlxCError> for ModernBertTrunkError {
    fn from(binding_error: MlxCError) -> Self {
        Self::Runtime(MlxRuntimeError::from(binding_error))
    }
}

/// One loaded trunk: the runtime, the artifact's tensors and how to read them.
pub(crate) struct ModernBertTrunk<'model> {
    pub(crate) runtime: &'model MlxRuntime,
    pub(crate) tensors: &'model MlxSafetensors,
    pub(crate) configuration: &'model ModernBertConfiguration,
    pub(crate) weight_layout: &'model ModernBertWeightLayout,
}

/// The additive masks one batch needs, built once and shared by every layer.
struct TrunkAttentionMasks {
    /// For global layers: hides padding. `None` when the batch has none.
    global_layer_mask: Option<MlxArray>,
    /// For local layers: hides padding and everything outside the window.
    /// `None` when neither hides anything.
    local_layer_mask: Option<MlxArray>,
}

/// One layer's attention inputs, `[batch, heads, length, head width]` each,
/// with the rotary embedding already applied to queries and keys.
struct RotatedAttentionProjections {
    queries: MlxArray,
    keys: MlxArray,
    values: MlxArray,
}

impl ModernBertTrunk<'_> {
    /// Builds the lazy graph for `token_batch` and returns its final hidden states.
    pub(crate) fn encode_hidden_states(
        &self,
        token_batch: &PaddedTokenBatch,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        // Both counts fit `i32`: `PaddedTokenBatch` refuses a batch that does not.
        let batch_shape = [
            token_batch.sequence_count() as i32,
            token_batch.padded_length() as i32,
        ];
        let token_id_array = self
            .runtime
            .array_from_i32(token_batch.token_ids(), &batch_shape)?;
        let attention_masks = self.build_attention_masks(token_batch)?;
        let mut layer_states = self.embed_tokens(&token_id_array)?;
        for layer_index in 0..self.configuration.layer_count {
            layer_states =
                self.encode_one_layer(layer_states, layer_index, batch_shape, &attention_masks)?;
        }
        self.layer_norm("final_norm.weight", &layer_states)
    }

    fn build_attention_masks(
        &self,
        token_batch: &PaddedTokenBatch,
    ) -> Result<TrunkAttentionMasks, ModernBertTrunkError> {
        let window_half = (self.configuration.local_attention_window / 2) as usize;
        let window_hides_pairs =
            local_window_hides_any_pair(token_batch.padded_length(), window_half);
        if token_batch.has_padding() {
            return self.build_padded_batch_masks(token_batch, window_half, window_hides_pairs);
        }
        // For sequences longer than the window the mask is a dense
        // [1,1,length,length] additive tensor. At the ModernBERT maximum
        // (8192 tokens) this is about 256 MiB of f32, accepted for
        // correctness; a banded representation would reduce it.
        let local_layer_mask = if window_hides_pairs {
            Some(self.upload_window_mask(
                token_batch.padded_length(),
                window_half,
                f32::NEG_INFINITY,
            )?)
        } else {
            None
        };
        Ok(TrunkAttentionMasks {
            global_layer_mask: None,
            local_layer_mask,
        })
    }

    /// The masks of a batch that carries padding: both layer kinds hide the
    /// padding, and local layers also hide everything outside the window.
    fn build_padded_batch_masks(
        &self,
        token_batch: &PaddedTokenBatch,
        window_half: usize,
        window_hides_pairs: bool,
    ) -> Result<TrunkAttentionMasks, ModernBertTrunkError> {
        let key_padding_values = key_padding_mask_values(token_batch);
        // Both counts fit `i32`: `PaddedTokenBatch` refuses a batch that does not.
        let key_padding_shape = [
            token_batch.sequence_count() as i32,
            1,
            1,
            token_batch.padded_length() as i32,
        ];
        let key_padding_mask = self
            .runtime
            .array_from_f32(&key_padding_values, &key_padding_shape)?;
        let local_layer_mask = if window_hides_pairs {
            let window_mask = self.upload_window_mask(
                token_batch.padded_length(),
                window_half,
                HIDDEN_POSITION_MASK_VALUE,
            )?;
            // The elementwise minimum hides a pair that either mask hides.
            // Adding the two would overflow to negative infinity where both
            // hide it, which is the value the finite mask exists to avoid.
            self.runtime.minimum(&window_mask, &key_padding_mask)?
        } else {
            // Every pair is inside the window, so local layers need the
            // padding mask alone; it is uploaded a second time because the
            // two layer kinds each own their mask.
            self.runtime
                .array_from_f32(&key_padding_values, &key_padding_shape)?
        };
        Ok(TrunkAttentionMasks {
            global_layer_mask: Some(key_padding_mask),
            local_layer_mask: Some(local_layer_mask),
        })
    }

    fn upload_window_mask(
        &self,
        padded_length: usize,
        window_half: usize,
        hidden_value: f32,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let padded_length_i32 = padded_length as i32;
        Ok(self.runtime.array_from_f32(
            &local_window_mask_values(padded_length, window_half, hidden_value),
            &[1, 1, padded_length_i32, padded_length_i32],
        )?)
    }

    fn embed_tokens(&self, token_id_array: &MlxArray) -> Result<MlxArray, ModernBertTrunkError> {
        let embedding_prefix = self.weight_layout.tensor_name("embeddings.tok_embeddings");
        let embedded_tokens = match self.weight_layout.token_embedding_storage() {
            TokenEmbeddingStorage::AffineQuantized => {
                self.quantized_gather(&embedding_prefix, token_id_array)?
            }
            TokenEmbeddingStorage::Dense => {
                let embedding_table =
                    self.tensor(&format!("{embedding_prefix}{QUANTIZED_WEIGHTS_TAIL}"))?;
                let looked_up_rows = self
                    .runtime
                    .take_axis(&embedding_table, token_id_array, 0)?;
                // The table is stored in half precision. The trunk runs in
                // single precision, as the reference does, so the rows are
                // widened before the first normalization.
                self.runtime.astype(&looked_up_rows, MlxDtype::Float32)?
            }
        };
        self.layer_norm("embeddings.norm.weight", &embedded_tokens)
    }

    fn encode_one_layer(
        &self,
        layer_states: MlxArray,
        layer_index: u32,
        batch_shape: [i32; 2],
        attention_masks: &TrunkAttentionMasks,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let normalized_states;
        let attention_input = if layer_index == 0 {
            // ModernBERT applies no attention norm on layer zero (Identity).
            &layer_states
        } else {
            normalized_states = self.layer_norm(
                &format!("layers.{layer_index}.attn_norm.weight"),
                &layer_states,
            )?;
            &normalized_states
        };
        let attention_output =
            self.encode_attention(attention_input, layer_index, batch_shape, attention_masks)?;
        let residual_states = self.runtime.add(&layer_states, &attention_output)?;
        let mlp_input = self.layer_norm(
            &format!("layers.{layer_index}.mlp_norm.weight"),
            &residual_states,
        )?;
        let mlp_output = self.encode_mlp(&mlp_input, layer_index)?;
        Ok(self.runtime.add(&residual_states, &mlp_output)?)
    }

    fn encode_attention(
        &self,
        attention_input: &MlxArray,
        layer_index: u32,
        batch_shape: [i32; 2],
        attention_masks: &TrunkAttentionMasks,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let [sequence_count, padded_length] = batch_shape;
        let is_global_layer = self.is_global_layer(layer_index);
        let rotated_projections =
            self.project_and_rotate(attention_input, layer_index, batch_shape, is_global_layer)?;
        let head_dimension = self.configuration.head_dimension();
        let attention_scale = f32::from(head_dimension as u16).powf(-0.5);
        let layer_mask = if is_global_layer {
            attention_masks.global_layer_mask.as_ref()
        } else {
            attention_masks.local_layer_mask.as_ref()
        };
        let attended = match layer_mask {
            Some(attention_mask) => self.runtime.masked_scaled_dot_product_attention(
                &rotated_projections.queries,
                &rotated_projections.keys,
                &rotated_projections.values,
                attention_scale,
                attention_mask,
            )?,
            None => self.runtime.scaled_dot_product_attention(
                &rotated_projections.queries,
                &rotated_projections.keys,
                &rotated_projections.values,
                attention_scale,
            )?,
        };
        let sequence_ordered = self.runtime.transpose_axes(&attended, &[0, 2, 1, 3])?;
        let flat_attention = self.runtime.reshape(
            &sequence_ordered,
            &[
                sequence_count,
                padded_length,
                self.configuration.hidden_size as i32,
            ],
        )?;
        self.quantized_matmul(&format!("layers.{layer_index}.attn.Wo"), &flat_attention)
    }

    /// Splits the fused `Wqkv` projection into per-head queries, keys and
    /// values and applies the layer kind's rotary embedding to queries and keys.
    fn project_and_rotate(
        &self,
        attention_input: &MlxArray,
        layer_index: u32,
        batch_shape: [i32; 2],
        is_global_layer: bool,
    ) -> Result<RotatedAttentionProjections, ModernBertTrunkError> {
        let [sequence_count, padded_length] = batch_shape;
        let head_dimension = self.configuration.head_dimension();
        let fused_qkv =
            self.quantized_matmul(&format!("layers.{layer_index}.attn.Wqkv"), attention_input)?;
        let split_qkv = self.runtime.reshape(
            &fused_qkv,
            &[
                sequence_count,
                padded_length,
                3,
                self.configuration.attention_head_count as i32,
                head_dimension as i32,
            ],
        )?;
        let reordered_qkv = self.runtime.transpose_axes(&split_qkv, &[0, 3, 2, 1, 4])?;
        let queries = self.slice_projection(&reordered_qkv, 0)?;
        let keys = self.slice_projection(&reordered_qkv, 1)?;
        let values = self.slice_projection(&reordered_qkv, 2)?;
        let rope_theta = if is_global_layer {
            self.configuration.global_rope_theta
        } else {
            self.configuration.local_rope_theta
        };
        let rope_dimension = i32::try_from(head_dimension).unwrap_or(i32::MAX);
        Ok(RotatedAttentionProjections {
            queries: self.runtime.rope(&queries, rope_dimension, rope_theta, 0)?,
            keys: self.runtime.rope(&keys, rope_dimension, rope_theta, 0)?,
            values,
        })
    }

    fn encode_mlp(
        &self,
        mlp_input: &MlxArray,
        layer_index: u32,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let fused_input_projection =
            self.quantized_matmul(&format!("layers.{layer_index}.mlp.Wi"), mlp_input)?;
        let intermediate_width = fused_input_projection.shape().last().copied().unwrap_or(0) / 2;
        // HuggingFace ModernBERT applies the activation to the FIRST Wi chunk and
        // multiplies it by the SECOND chunk (act(input) * gate); GELU is not
        // symmetric, so the chunk order changes every MLP output.
        let activation_input =
            self.slice_last_axis(&fused_input_projection, 0, intermediate_width)?;
        let gate_states = self.slice_last_axis(
            &fused_input_projection,
            intermediate_width,
            intermediate_width * 2,
        )?;
        let activated = self.runtime.gelu(&activation_input)?;
        let gated_input = self.runtime.multiply(&activated, &gate_states)?;
        self.quantized_matmul(&format!("layers.{layer_index}.mlp.Wo"), &gated_input)
    }

    fn quantized_gather(
        &self,
        full_tensor_prefix: &str,
        token_id_array: &MlxArray,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let quantized_weights =
            self.tensor(&format!("{full_tensor_prefix}{QUANTIZED_WEIGHTS_TAIL}"))?;
        let scales = self.tensor(&format!("{full_tensor_prefix}{QUANTIZED_SCALES_TAIL}"))?;
        let biases = self.tensor(&format!("{full_tensor_prefix}{QUANTIZED_BIASES_TAIL}"))?;
        let selected_weights = self
            .runtime
            .take_axis(&quantized_weights, token_id_array, 0)?;
        let selected_scales = self.runtime.take_axis(&scales, token_id_array, 0)?;
        let selected_biases = self.runtime.take_axis(&biases, token_id_array, 0)?;
        Ok(self.runtime.dequantize_affine(
            &selected_weights,
            &selected_scales,
            &selected_biases,
            self.configuration.quantization_group_size as i32,
            self.configuration.quantization_bits as i32,
        )?)
    }

    fn quantized_matmul(
        &self,
        tensor_suffix: &str,
        activations: &MlxArray,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let full_tensor_prefix = self.weight_layout.tensor_name(tensor_suffix);
        let quantized_weights =
            self.tensor(&format!("{full_tensor_prefix}{QUANTIZED_WEIGHTS_TAIL}"))?;
        let scales = self.tensor(&format!("{full_tensor_prefix}{QUANTIZED_SCALES_TAIL}"))?;
        let biases = self.tensor(&format!("{full_tensor_prefix}{QUANTIZED_BIASES_TAIL}"))?;
        Ok(self.runtime.quantized_matmul_affine(
            activations,
            &quantized_weights,
            &scales,
            &biases,
            true,
            self.configuration.quantization_group_size as i32,
            self.configuration.quantization_bits as i32,
        )?)
    }

    /// LayerNorm with the artifact's weight and no bias, as ModernBERT defines it.
    fn layer_norm(
        &self,
        weight_tensor_suffix: &str,
        input_states: &MlxArray,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let stored_weight = self.tensor(&self.weight_layout.tensor_name(weight_tensor_suffix))?;
        let weight = self.runtime.astype(&stored_weight, MlxDtype::Float32)?;
        let zero_bias = self
            .runtime
            .zeros(&[self.configuration.hidden_size as i32], MlxDtype::Float32)?;
        Ok(self.runtime.layer_norm(
            input_states,
            &weight,
            &zero_bias,
            self.configuration.layer_norm_epsilon,
        )?)
    }

    /// One of the query, key or value projections out of `[batch, heads, 3, length, width]`.
    fn slice_projection(
        &self,
        reordered_qkv: &MlxArray,
        projection_index: i32,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let qkv_shape = reordered_qkv.shape();
        let one_projection = self.runtime.slice(
            reordered_qkv,
            &[0, 0, projection_index, 0, 0],
            &[
                qkv_shape[0],
                qkv_shape[1],
                projection_index + 1,
                qkv_shape[3],
                qkv_shape[4],
            ],
            &[1, 1, 1, 1, 1],
        )?;
        Ok(self.runtime.squeeze_axis(&one_projection, 2)?)
    }

    /// The `[start, stop)` range of the last axis of a `[batch, length, width]` array.
    fn slice_last_axis(
        &self,
        fused_states: &MlxArray,
        start: i32,
        stop: i32,
    ) -> Result<MlxArray, ModernBertTrunkError> {
        let fused_shape = fused_states.shape();
        let width = fused_shape.last().copied().unwrap_or(0);
        Ok(self.runtime.slice(
            fused_states,
            &[0, 0, start],
            &[fused_shape[0], fused_shape[1], stop.min(width)],
            &[1, 1, 1],
        )?)
    }

    fn is_global_layer(&self, layer_index: u32) -> bool {
        layer_index.is_multiple_of(self.configuration.global_attn_every_n_layers)
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

#[cfg(test)]
#[path = "trunk_tests.rs"]
mod tests;
