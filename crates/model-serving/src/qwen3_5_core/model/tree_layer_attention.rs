//! One decoder layer's attention over the rows of a draft tree, and the
//! commit of the chain a round keeps.
//!
//! A tree pass must give each row what its own chain alone would give. Full
//! attention appends every row's key and value, then masks each row to the
//! context and its ancestors, with each row rotated to its own position
//! (siblings share one). A DeltaNet layer convolves each row's own window
//! and runs the recurrence from each row's ancestor chain (`gated_delta_tree`);
//! it never advances the layer's state, because the kept chain is not known
//! until every row has been verified. The commit then advances each layer
//! over exactly the kept rows: the convolution window and a replay of the
//! recurrence over those rows for DeltaNet, and the kept keys and values written
//! back in order for attention.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxMetalKernel};

use super::Qwen3_5DecoderLayerAttentionOutput;
use super::base::Qwen3_5ModelBase;
use super::full_attention::RotatedAttentionInputs;
use super::gated_delta::normalize_linear_attention_convolution_weight;
use crate::astro_decoder_state_cache::{
    ConvolutionState, FullAttentionKeyValueState, GatedDeltaRecurrentState,
};
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AttentionWeights, Qwen3_5DecoderLayerWeights, Qwen3_5FullAttentionWeights,
    Qwen3_5LinearAttentionWeights,
};
use crate::qwen3_5_core::model_math::error::{
    Qwen3_5ExecutionError, invalid_request_decoder_state,
};
use crate::qwen3_5_core::model_math::gated_delta_tree::{
    UploadedTreeRowChains, qwen3_5_gated_delta_replay, qwen3_5_gated_delta_tree,
};
use crate::{AstroDecoderStateCacheState, PerformanceAttribution};

/// The rows of one draft tree on the GPU, shared by every layer of a pass.
pub(crate) struct TreeRowInputs {
    pub(crate) row_count: i32,
    /// int32 [rows]: each row's absolute position.
    pub(crate) positions: MlxArray,
    /// [rows, context + rows] additive mask in the activations' dtype.
    pub(crate) attention_mask: MlxArray,
    /// int32 [rows x window]: each row's convolution inputs, as indices into
    /// the convolution state's rows followed by the tree's rows.
    pub(crate) convolution_windows: MlxArray,
    pub(crate) row_chains: UploadedTreeRowChains,
}

/// What one layer's tree pass leaves for the round's commit.
pub(crate) enum TreeLayerRows {
    LinearAttention {
        /// [1, rows, conv channels]: each row's convolution input.
        mixed_queries_keys_values: MlxArray,
        queries: MlxArray,
        keys: MlxArray,
        values: MlxArray,
        decays: MlxArray,
        update_rates: MlxArray,
    },
    FullAttention {
        /// [1, key-value heads, rows, features], already in the KV state
        /// after the context; the commit rewrites the kept rows in order.
        rotated_keys: MlxArray,
        transposed_values: MlxArray,
    },
}

impl Qwen3_5ModelBase {
    /// Declares the tree recurrence kernel, so this model can verify draft
    /// trees. MLX compiles it on first use.
    pub fn enable_draft_tree_verification(&mut self) -> Result<(), Qwen3_5ExecutionError> {
        self.optional_accelerators.gated_delta_tree_kernel = Some(
            crate::qwen3_5_core::model_math::gated_delta_tree::qwen3_5_gated_delta_tree_kernel()?,
        );
        self.optional_accelerators.gated_delta_replay_kernel = Some(
            crate::qwen3_5_core::model_math::gated_delta_tree::qwen3_5_gated_delta_replay_kernel()?,
        );
        Ok(())
    }

    /// The layer's attention half over the tree rows: input norm, attention,
    /// residual, post-attention norm, as `forward_decoder_layer_attention`.
    pub(crate) fn forward_tree_layer_attention(
        &self,
        hidden_states: &MlxArray,
        tree: &TreeRowInputs,
        layer_index: usize,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
        layer_model_state: &mut AstroDecoderStateCacheState,
    ) -> Result<(Qwen3_5DecoderLayerAttentionOutput, TreeLayerRows), Qwen3_5ExecutionError> {
        let epsilon = f32::from_bits(self.config.rms_norm_epsilon_bits());
        let normalized_input = self.runtime.rms_norm(
            hidden_states,
            &decoder_layer_weights.input_normalization_weight,
            epsilon,
        )?;
        let (attention_output, layer_rows) =
            match (&decoder_layer_weights.attention_weights, layer_model_state) {
                (
                    Qwen3_5AttentionWeights::Linear(linear_weights),
                    AstroDecoderStateCacheState::Composite {
                        convolution,
                        recurrent,
                    },
                ) => self.tree_linear_attention(
                    &normalized_input,
                    tree,
                    layer_index,
                    linear_weights,
                    (convolution, recurrent),
                )?,
                (
                    Qwen3_5AttentionWeights::Full(full_weights),
                    AstroDecoderStateCacheState::AppendOnlyAttention { attention },
                ) => self.tree_full_attention(&normalized_input, tree, full_weights, attention)?,
                _ => {
                    return Err(invalid_request_decoder_state(
                        layer_index,
                        "decoder state attention family does not match the bound layer weights",
                    ));
                }
            };
        let attention_residual = self.runtime.add(hidden_states, &attention_output)?;
        let normalized_attention = self.runtime.rms_norm(
            &attention_residual,
            &decoder_layer_weights.post_attention_normalization_weight,
            epsilon,
        )?;
        Ok((
            Qwen3_5DecoderLayerAttentionOutput {
                attention_residual,
                normalized_attention,
            },
            layer_rows,
        ))
    }

    fn tree_linear_attention(
        &self,
        normalized_input: &MlxArray,
        tree: &TreeRowInputs,
        layer_index: usize,
        weights: &Qwen3_5LinearAttentionWeights,
        (convolution_state, recurrent_state): (
            &mut ConvolutionState,
            &mut GatedDeltaRecurrentState,
        ),
    ) -> Result<(MlxArray, TreeLayerRows), Qwen3_5ExecutionError> {
        let mut attribution = PerformanceAttribution::disabled();
        let (mixed_queries_keys_values, output_gate, update_logits, decay_inputs) = self
            .project_linear_attention_inputs(
                normalized_input,
                weights,
                tree.row_count,
                layer_index,
                &mut attribution,
            )?;
        let convolved =
            self.tree_convolution(&mixed_queries_keys_values, tree, weights, convolution_state)?;
        let (queries, keys, values) =
            self.normalized_heads_from_convolution(&convolved, tree.row_count, &mut attribution)?;
        let update_rates = self.runtime.sigmoid(&update_logits)?;
        let decays = self.runtime.apply_compiled_gated_delta_decay(
            &self.compiled_elementwise_graphs,
            &weights.decay_rate_logarithm,
            &decay_inputs,
            &weights.decay_interval_bias,
        )?;
        let recurrent_output = qwen3_5_gated_delta_tree(
            &self.runtime,
            self.tree_kernel()?,
            &queries,
            &keys,
            &values,
            &decays,
            &update_rates,
            &recurrent_state.current_or_zero(&self.runtime)?,
            &tree.row_chains,
        )?;
        let projected_output = self.gated_linear_attention_output(
            &recurrent_output,
            &output_gate,
            tree.row_count,
            weights,
        )?;
        Ok((
            projected_output,
            TreeLayerRows::LinearAttention {
                mixed_queries_keys_values,
                queries,
                keys,
                values,
                decays,
                update_rates,
            },
        ))
    }

    /// Each row's depthwise convolution over its own window: the windows are
    /// gathered into a batch of `window`-long sequences, so the convolution is
    /// the same `conv1d` the chain path runs.
    fn tree_convolution(
        &self,
        mixed_queries_keys_values: &MlxArray,
        tree: &TreeRowInputs,
        weights: &Qwen3_5LinearAttentionWeights,
        convolution_state: &ConvolutionState,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let channel_count = mixed_queries_keys_values.shape()[2];
        let window = self.config.linear_convolution_kernel_dimension() as i32;
        let convolution_weight = normalize_linear_attention_convolution_weight(
            &self.runtime,
            &weights.convolution_weight,
            channel_count,
            window,
        )?;
        let state_rows =
            convolution_state.current_or_zero(&self.runtime, mixed_queries_keys_values.dtype())?;
        let sources = self
            .runtime
            .concatenate_axis(&[&state_rows, mixed_queries_keys_values], 1)?;
        let windows = self.runtime.reshape(
            &self
                .runtime
                .take_axis(&sources, &tree.convolution_windows, 1)?,
            &[tree.row_count, window, channel_count],
        )?;
        let convolved =
            self.runtime
                .conv1d(&windows, &convolution_weight, 1, 0, 1, channel_count)?;
        Ok(self
            .runtime
            .reshape(&convolved, &[1, tree.row_count, channel_count])?)
    }

    fn tree_full_attention(
        &self,
        normalized_input: &MlxArray,
        tree: &TreeRowInputs,
        weights: &Qwen3_5FullAttentionWeights,
        kv_state: &mut FullAttentionKeyValueState,
    ) -> Result<(MlxArray, TreeLayerRows), Qwen3_5ExecutionError> {
        let RotatedAttentionInputs {
            rotated_queries,
            rotated_keys,
            transposed_values,
            output_gate,
        } = self.rotated_attention_inputs(
            normalized_input,
            tree.row_count,
            0,
            weights,
            Some(&tree.positions),
        )?;
        let context_offset = kv_state.offset_tokens();
        let (active_keys, active_values) = kv_state.update_and_fetch(
            &self.runtime,
            &rotated_keys,
            &transposed_values,
            context_offset,
        )?;
        let scale = (self.config.head_dimension() as f32).sqrt().recip();
        let attended = self.runtime.masked_scaled_dot_product_attention(
            &rotated_queries,
            &active_keys,
            &active_values,
            scale,
            &tree.attention_mask,
        )?;
        let merged_heads = self.runtime.reshape(
            &self.runtime.transpose_axes(&attended, &[0, 2, 1, 3])?,
            &[1, tree.row_count, output_gate.shape()[2]],
        )?;
        let gated = self.runtime.apply_compiled_attention_output_gate(
            &self.compiled_elementwise_graphs,
            &merged_heads,
            &output_gate,
        )?;
        let projected = self.quantized_linear(&gated, &weights.output_projection)?;
        Ok((
            projected,
            TreeLayerRows::FullAttention {
                rotated_keys,
                transposed_values,
            },
        ))
    }

    fn tree_kernel(&self) -> Result<&MlxMetalKernel, Qwen3_5ExecutionError> {
        self.optional_accelerators
            .gated_delta_tree_kernel
            .as_ref()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "draft-tree verification needs `enable_draft_tree_verification` first",
            })
    }

    /// Advances one layer over the kept rows (`kept_rows` int32 indices into
    /// the tree's rows, anchor first) from where the tree pass left it.
    pub(crate) fn commit_tree_layer(
        &self,
        layer_index: usize,
        layer_model_state: &mut AstroDecoderStateCacheState,
        layer_rows: TreeLayerRows,
        kept_rows: &MlxArray,
        context_offset: i32,
    ) -> Result<(), Qwen3_5ExecutionError> {
        match (layer_rows, layer_model_state) {
            (
                linear_rows @ TreeLayerRows::LinearAttention { .. },
                AstroDecoderStateCacheState::Composite {
                    convolution,
                    recurrent,
                },
            ) => commit_linear_attention(
                &self.runtime,
                self.optional_accelerators
                    .gated_delta_replay_kernel
                    .as_ref()
                    .ok_or(Qwen3_5ExecutionError::InvalidInput {
                        description: "a tree commit needs `enable_draft_tree_verification` first",
                    })?,
                linear_rows,
                kept_rows,
                (convolution, recurrent),
            ),
            (
                TreeLayerRows::FullAttention {
                    rotated_keys,
                    transposed_values,
                },
                AstroDecoderStateCacheState::AppendOnlyAttention { attention },
            ) => {
                attention.truncate_to_offset(context_offset)?;
                attention.update_and_fetch(
                    &self.runtime,
                    &self.runtime.take_axis(&rotated_keys, kept_rows, 2)?,
                    &self.runtime.take_axis(&transposed_values, kept_rows, 2)?,
                    context_offset,
                )?;
                Ok(())
            }
            _ => Err(invalid_request_decoder_state(
                layer_index,
                "a tree commit's rows do not match the layer's state",
            )),
        }
    }
}

/// The DeltaNet layer after the kept rows: the convolution window slides
/// over their inputs, and the replay kernel steps the recurrence over them
/// from the round's starting state, exactly as feeding them would.
fn commit_linear_attention(
    runtime: &MlxRuntime,
    replay_kernel: &MlxMetalKernel,
    linear_rows: TreeLayerRows,
    kept_rows: &MlxArray,
    (convolution_state, recurrent_state): (&mut ConvolutionState, &mut GatedDeltaRecurrentState),
) -> Result<(), Qwen3_5ExecutionError> {
    let TreeLayerRows::LinearAttention {
        mixed_queries_keys_values,
        queries,
        keys,
        values,
        decays,
        update_rates,
    } = linear_rows
    else {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "a DeltaNet commit needs DeltaNet rows",
        });
    };
    let kept_count = kept_rows.shape()[0];
    convolution_state.update(
        runtime,
        &runtime.take_axis(&mixed_queries_keys_values, kept_rows, 1)?,
        kept_count,
    )?;
    let starting_state = recurrent_state.current_or_zero(runtime)?;
    recurrent_state.set_next(qwen3_5_gated_delta_replay(
        runtime,
        replay_kernel,
        &queries,
        &keys,
        &values,
        &decays,
        &update_rates,
        &starting_state,
        kept_rows,
    )?);
    Ok(())
}
