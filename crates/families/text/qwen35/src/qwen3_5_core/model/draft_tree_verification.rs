//! Verifying a whole draft tree in one target pass, then keeping one chain.
//!
//! `verify_draft_tree` runs every row of the tree (the anchor first, each
//! node after its parent) through every layer, each row computing what its
//! own chain alone would compute (`tree_layer_attention`). DeltaNet layers
//! are left where the round started and attention layers hold every row
//! after the context; `commit_draft_tree` then advances every layer over the
//! kept chain only. Nothing is captured per row and nothing runs twice.
//!
//! The pass lives in the shared core so every Mac verifies the same way,
//! whether it keeps the model in memory or streams it. Only where a layer
//! finds its weights (all in memory, or read through the
//! `astro_expert_cache`) and how the pass's graph is handed to the GPU differ,
//! so those steps alone are the `DraftTreeTarget` trait.

use mlx_bridge::{MlxArray, MlxDtype};

use super::Qwen3_5ModelBase;
use super::tree_layer_attention::{TreeLayerRows, TreeRowInputs};
use crate::RequestDecoderStateStack;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::gated_delta_tree::TreeRowChains;
use engine_core::speculative_decoding::tree_rows::TreeRows;
use text_generation::execution_error::TextExecutionError;

/// Layers built between asynchronous starts. Building the pass's graph
/// takes about 4 ms of CPU for the 27B's 64 layers (some 40 nodes a layer),
/// during which the GPU would idle; starting it on each few layers as they
/// are built lets the two overlap.
const LAYERS_PER_ASYNCHRONOUS_START: usize = 4;

/// One verified tree, held until the round decides which chain it keeps.
pub struct Qwen3_5DraftTreeVerification {
    /// [1, rows, vocabulary] float32: the target's logits after every row.
    pub per_row_logits: MlxArray,
    /// One tap per requested layer, each [1, rows, hidden].
    pub hidden_state_taps: Vec<MlxArray>,
    layer_rows: Vec<TreeLayerRows>,
    /// Tokens in the state before the tree's anchor.
    context_offset: i32,
}

/// What one layer's step over the tree's rows is given.
pub struct TreeLayerStep<'pass> {
    /// `[1, rows, hidden]`: the rows as the layer before left them.
    pub hidden_states: &'pass MlxArray,
    pub tree: &'pass TreeRowInputs,
    pub layer_index: usize,
    pub request_decoder_state: &'pass mut RequestDecoderStateStack,
}

/// One layer's weights as a pass holds them: a reference to weights bound
/// in memory, or the owner of a layer read for this pass alone.
pub trait TreeLayerWeights {
    fn decoder_layer_weights(&self) -> &Qwen3_5DecoderLayerWeights;
}

impl TreeLayerWeights for &Qwen3_5DecoderLayerWeights {
    fn decoder_layer_weights(&self) -> &Qwen3_5DecoderLayerWeights {
        self
    }
}

/// What a layer's feed-forward over the tree's rows is given.
pub struct TreeLayerFeedForward<'pass> {
    pub attention_output: &'pass super::Qwen3_5DecoderLayerAttentionOutput,
    pub row_count: i32,
    pub layer_index: usize,
    pub decoder_layer_weights: &'pass Qwen3_5DecoderLayerWeights,
}

/// A model a draft tree can be verified against: the shared base, plus the
/// steps that depend on where the weights live.
pub trait DraftTreeTarget {
    /// The shared model the tree's embedding, attention and commit run on.
    fn draft_tree_model_base(&self) -> &Qwen3_5ModelBase;

    /// Where this engine finds one layer's weights for the pass: bound in
    /// memory, or read from storage for this pass alone.
    fn tree_layer_weights(
        &self,
        layer_index: usize,
    ) -> Result<Box<dyn TreeLayerWeights + '_>, TextExecutionError>;

    /// The feed-forward half of one layer over every tree row, from that
    /// layer's attention output.
    fn tree_layer_feed_forward(
        &self,
        feed_forward: TreeLayerFeedForward<'_>,
    ) -> Result<MlxArray, TextExecutionError>;

    /// One layer over every tree row: its attention half from the base,
    /// then this engine's feed-forward.
    fn forward_tree_layer(
        &self,
        layer_step: TreeLayerStep<'_>,
    ) -> Result<(MlxArray, TreeLayerRows), TextExecutionError> {
        let layer_index = layer_step.layer_index;
        let row_count = layer_step.tree.row_count;
        let layer_weights = self.tree_layer_weights(layer_index)?;
        let decoder_layer_weights = layer_weights.decoder_layer_weights();
        let (attention_output, rows) = self
            .draft_tree_model_base()
            .tree_layer_attention_in_request(layer_step, decoder_layer_weights)?;
        let layer_output = self.tree_layer_feed_forward(TreeLayerFeedForward {
            attention_output: &attention_output,
            row_count,
            layer_index,
            decoder_layer_weights,
        })?;
        Ok((layer_output, rows))
    }

    /// `[1, rows, vocabulary]` float32: the target's logits after every row
    /// of the final hidden states.
    fn tree_per_row_logits(&self, hidden_states: &MlxArray)
    -> Result<MlxArray, TextExecutionError>;

    /// Whether the pass hands the GPU each few layers while the next are
    /// built. An engine whose layers run host steps inside the graph hands
    /// it over once, in its own evaluation.
    fn starts_tree_layers_asynchronously(&self) -> bool;

    /// Verifies `tree_rows`, whose anchor sits at `anchor_position` (every token
    /// before it is already in `request_decoder_state`).
    fn verify_draft_tree(
        &self,
        tree_rows: &TreeRows,
        anchor_position: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        target_hidden_state_tap_layer_ids: &[i32],
    ) -> Result<Qwen3_5DraftTreeVerification, TextExecutionError> {
        let base = self.draft_tree_model_base();
        let context_offset =
            i32::try_from(anchor_position).map_err(|_| TextExecutionError::InvalidInput {
                description: "the anchor position exceeds the MLX int32 range",
            })?;
        let mut hidden_states = base.tree_row_embeddings(tree_rows)?;
        let tree = base.tree_row_inputs(tree_rows, context_offset, hidden_states.dtype())?;
        let mut tap_slots: Vec<Option<MlxArray>> = Vec::new();
        tap_slots.resize_with(target_hidden_state_tap_layer_ids.len(), Option::default);
        let layer_count = base.config.layer_count() as usize;
        let mut layer_rows = Vec::with_capacity(layer_count);
        for layer_index in 0..layer_count {
            let (next_hidden_states, rows) = self.forward_tree_layer(TreeLayerStep {
                hidden_states: &hidden_states,
                tree: &tree,
                layer_index,
                request_decoder_state,
            })?;
            hidden_states = next_hidden_states;
            layer_rows.push(rows);
            if self.starts_tree_layers_asynchronously()
                && (layer_index + 1) % LAYERS_PER_ASYNCHRONOUS_START == 0
            {
                base.runtime.async_eval_arrays(&[&hidden_states])?;
            }
            for (tap_order, tap_layer_id) in target_hidden_state_tap_layer_ids.iter().enumerate() {
                if *tap_layer_id == layer_index as i32 {
                    tap_slots[tap_order] = Some(hidden_states.retain()?);
                }
            }
        }
        Ok(Qwen3_5DraftTreeVerification {
            per_row_logits: self.tree_per_row_logits(&hidden_states)?,
            hidden_state_taps: collected_taps(tap_slots)?,
            layer_rows,
            context_offset,
        })
    }

    /// Advances every layer over `kept_rows` (tree rows, anchor first), as if
    /// only that chain had been fed after the context.
    fn commit_draft_tree(
        &self,
        request_decoder_state: &mut RequestDecoderStateStack,
        verification: Qwen3_5DraftTreeVerification,
        kept_rows: &[usize],
    ) -> Result<(), TextExecutionError> {
        let base = self.draft_tree_model_base();
        let mut kept_row_ids = Vec::with_capacity(kept_rows.len());
        for kept_row in kept_rows {
            kept_row_ids.push(*kept_row as i32);
        }
        let kept_row_array = base
            .runtime
            .array_from_i32(&kept_row_ids, &[kept_rows.len() as i32])?;
        for (layer_index, layer_rows) in verification.layer_rows.into_iter().enumerate() {
            let layer_model_state = request_layer_state(request_decoder_state, layer_index)?;
            base.commit_tree_layer(
                layer_index,
                layer_model_state,
                layer_rows,
                &kept_row_array,
                verification.context_offset,
            )?;
        }
        Ok(())
    }
}

/// The state of layer `layer_index` in the request's stack, or the error
/// every forward reports when the stack has no such layer.
pub fn request_layer_state(
    request_decoder_state: &mut RequestDecoderStateStack,
    layer_index: usize,
) -> Result<&mut crate::AstroDecoderStateCacheState, TextExecutionError> {
    request_decoder_state.layer_mut(layer_index).ok_or(
        TextExecutionError::InvalidRequestDecoderState {
            layer_index,
            description: "decoder layer entry is missing",
        },
    )
}

impl Qwen3_5ModelBase {
    /// A layer's attention half over the tree rows, on that layer's state
    /// in the request: what every engine's layer step begins with.
    pub fn tree_layer_attention_in_request(
        &self,
        layer_step: TreeLayerStep<'_>,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
    ) -> Result<(super::Qwen3_5DecoderLayerAttentionOutput, TreeLayerRows), TextExecutionError>
    {
        self.forward_tree_layer_attention(
            layer_step.hidden_states,
            layer_step.tree,
            layer_step.layer_index,
            decoder_layer_weights,
            request_layer_state(layer_step.request_decoder_state, layer_step.layer_index)?,
        )
    }

    /// `[1, rows, vocabulary]` float32 logits after every row of Qwen 3.5's
    /// final hidden states: the final norm, then the head, widened so two
    /// rows' picks compare the same way whatever the head's precision.
    pub fn normalized_vocabulary_logits_for_every_row(
        &self,
        hidden_states: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        let normalized_states = self.final_normalized_states(hidden_states)?;
        let head_logits = self.quantized_linear(
            &normalized_states,
            &self.weights.language_model_head_weights,
        )?;
        Ok(self.runtime.astype(&head_logits, MlxDtype::Float32)?)
    }

    fn tree_row_embeddings(&self, tree_rows: &TreeRows) -> Result<MlxArray, TextExecutionError> {
        let mut token_ids = Vec::with_capacity(tree_rows.row_count());
        for token_id in &tree_rows.token_ids {
            token_ids.push(*token_id as i32);
        }
        let token_indices = self
            .runtime
            .array_from_i32(&token_ids, &[1, token_ids.len() as i32])?;
        self.context_free_token_meanings(&token_indices)
    }

    /// The tree's positions, attention mask, convolution windows and row
    /// chains on the GPU, built once for every layer of the pass.
    fn tree_row_inputs(
        &self,
        tree_rows: &TreeRows,
        context_offset: i32,
        activation_dtype: MlxDtype,
    ) -> Result<TreeRowInputs, TextExecutionError> {
        let row_count = tree_rows.row_count() as i32;
        let window = self.config.linear_convolution_kernel_dimension() as usize;
        let window_sources = tree_rows.convolution_window_sources(window);
        Ok(TreeRowInputs {
            row_count,
            positions: self
                .runtime
                .array_from_i32(&tree_rows.positions(context_offset), &[row_count])?,
            attention_mask: self.tree_attention_mask(
                tree_rows,
                context_offset,
                activation_dtype,
            )?,
            convolution_windows: self
                .runtime
                .array_from_i32(&window_sources, &[window_sources.len() as i32])?,
            row_chains: TreeRowChains::from_parent_rows(&tree_rows.parent_rows)
                .upload(&self.runtime)?,
        })
    }

    /// `[rows, context + rows]`: 0 where a row may attend (all of the
    /// context, its ancestors and itself), negative infinity elsewhere, in
    /// the activations' dtype so the fused attention keeps its path.
    fn tree_attention_mask(
        &self,
        tree_rows: &TreeRows,
        context_offset: i32,
        activation_dtype: MlxDtype,
    ) -> Result<MlxArray, TextExecutionError> {
        let row_count = tree_rows.row_count();
        let context = context_offset as usize;
        let visibility = tree_rows.ancestor_visibility();
        let mut additive = vec![0.0_f32; row_count * (context + row_count)];
        for row in 0..row_count {
            for column in 0..row_count {
                if !visibility[row * row_count + column] {
                    additive[row * (context + row_count) + context + column] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = self
            .runtime
            .array_from_f32(&additive, &[row_count as i32, (context + row_count) as i32])?;
        Ok(self.runtime.astype(&mask, activation_dtype)?)
    }
}

fn collected_taps(tap_slots: Vec<Option<MlxArray>>) -> Result<Vec<MlxArray>, TextExecutionError> {
    let mut taps = Vec::with_capacity(tap_slots.len());
    for tap_slot in tap_slots {
        taps.push(tap_slot.ok_or(TextExecutionError::InvalidInput {
            description: "a requested DFlash target hidden-state tap was never captured",
        })?);
    }
    Ok(taps)
}
