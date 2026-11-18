//! Verifying a whole draft tree in one target pass, then keeping one chain.
//!
//! `verify_draft_tree` runs every row of the tree (the anchor first, each
//! node after its parent) through every layer, each row computing what its
//! own chain alone would compute (`tree_layer_attention`). DeltaNet layers
//! are left where the round started and attention layers hold every row
//! after the context; `commit_draft_tree` then advances every layer over the
//! kept chain only. Nothing is captured per row and nothing runs twice.
//!
//! The pass lives in the shared core so the resident engine and the
//! streaming engine verify the same way: a speed-up that only one engine can
//! run leaves every Mac that streams a model without it. The engines differ
//! in one step, where a layer's feed-forward finds its weights (all in
//! memory, or paged through the `astro_expert_cache`), so that step alone is
//! the `DraftTreeTarget` trait.

use mlx_bridge::{MlxArray, MlxDtype};

use super::Qwen3_5DecoderLayerAttentionOutput;
use super::Qwen3_5ModelBase;
use super::tree_layer_attention::{TreeLayerRows, TreeRowInputs};
use crate::RequestDecoderStateStack;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::gated_delta_tree::TreeRowChains;
use crate::speculative_decoding::tree_rows::TreeRows;

/// Layers built between asynchronous starts. Building the pass's graph
/// takes about 4 ms of CPU for the 27B's 64 layers (some 40 nodes a layer),
/// during which the GPU would idle; starting it on each few layers as they
/// are built lets the two overlap.
const LAYERS_PER_ASYNCHRONOUS_START: usize = 4;

/// Whether a tree pass's results can be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TreePassOutcome {
    /// Every row was computed from the weights it routed to.
    Valid,
    /// The pass assumed weights were in memory that were not: its results
    /// are wrong and it must be run again without that assumption.
    MustRunAgain,
}

/// A model a draft tree can be verified against: the shared base plus the
/// steps each engine does its own way.
pub(crate) trait DraftTreeTarget {
    /// The shared model the tree's attention, embedding and head run on.
    fn draft_tree_model_base(&self) -> &Qwen3_5ModelBase;

    /// The feed-forward half of one layer over every tree row, from that
    /// layer's attention output.
    fn forward_tree_rows_feed_forward(
        &self,
        attention_output: &Qwen3_5DecoderLayerAttentionOutput,
        row_count: i32,
        layer_index: usize,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
    ) -> Result<MlxArray, Qwen3_5ExecutionError>;

    /// Whether the next pass may assume every routed weight is in memory,
    /// to be checked when the pass is evaluated. A model with every weight
    /// in memory has nothing to assume.
    fn starts_tree_pass_optimistically(&self) -> bool {
        false
    }

    /// Tells the layers built from here on whether they run under that
    /// assumption.
    fn set_tree_pass_optimism(&self, _is_optimistic: bool) {}

    /// Evaluates the pass through `evaluation_roots` and reports whether its
    /// results can be used.
    fn evaluate_tree_pass(
        &self,
        evaluation_roots: &[&MlxArray],
    ) -> Result<TreePassOutcome, Qwen3_5ExecutionError> {
        self.draft_tree_model_base()
            .runtime
            .evaluate_arrays(evaluation_roots)?;
        Ok(TreePassOutcome::Valid)
    }
}

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

/// Verifies `tree_rows`, whose anchor sits at `anchor_position` (every token
/// before it is already in `request_decoder_state`).
pub(crate) fn verify_draft_tree(
    target: &impl DraftTreeTarget,
    tree_rows: &TreeRows,
    anchor_position: u32,
    request_decoder_state: &mut RequestDecoderStateStack,
    target_hidden_state_tap_layer_ids: &[i32],
) -> Result<Qwen3_5DraftTreeVerification, Qwen3_5ExecutionError> {
    let model_base = target.draft_tree_model_base();
    let context_offset =
        i32::try_from(anchor_position).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
            description: "the anchor position exceeds the MLX int32 range",
        })?;
    let mut hidden_states = model_base.tree_row_embeddings(tree_rows)?;
    let tree = model_base.tree_row_inputs(tree_rows, context_offset, hidden_states.dtype())?;
    let mut tap_slots: Vec<Option<MlxArray>> = Vec::new();
    tap_slots.resize_with(target_hidden_state_tap_layer_ids.len(), Option::default);
    let layer_count = model_base.config.layer_count() as usize;
    let mut layer_rows = Vec::with_capacity(layer_count);
    for layer_index in 0..layer_count {
        let (next_hidden_states, rows) = forward_tree_layer(
            target,
            &hidden_states,
            &tree,
            layer_index,
            request_decoder_state,
        )?;
        hidden_states = next_hidden_states;
        layer_rows.push(rows);
        if (layer_index + 1) % LAYERS_PER_ASYNCHRONOUS_START == 0 {
            model_base.runtime.async_eval_arrays(&[&hidden_states])?;
        }
        for (tap_order, tap_layer_id) in target_hidden_state_tap_layer_ids.iter().enumerate() {
            if *tap_layer_id == layer_index as i32 {
                tap_slots[tap_order] = Some(hidden_states.retain()?);
            }
        }
    }
    Ok(Qwen3_5DraftTreeVerification {
        per_row_logits: model_base.tree_per_row_logits(&hidden_states)?,
        hidden_state_taps: collected_taps(tap_slots)?,
        layer_rows,
        context_offset,
    })
}

/// Advances every layer over `kept_rows` (tree rows, anchor first), as if
/// only that chain had been fed after the context. Only attention state
/// moves, so the commit is the same on every engine.
pub(crate) fn commit_draft_tree(
    model_base: &Qwen3_5ModelBase,
    request_decoder_state: &mut RequestDecoderStateStack,
    verification: Qwen3_5DraftTreeVerification,
    kept_rows: &[usize],
) -> Result<(), Qwen3_5ExecutionError> {
    let mut kept_row_ids = Vec::with_capacity(kept_rows.len());
    for kept_row in kept_rows {
        kept_row_ids.push(*kept_row as i32);
    }
    let kept_row_array = model_base
        .runtime
        .array_from_i32(&kept_row_ids, &[kept_rows.len() as i32])?;
    for (layer_index, layer_rows) in verification.layer_rows.into_iter().enumerate() {
        let layer_model_state = request_decoder_state.layer_mut(layer_index).ok_or(
            Qwen3_5ExecutionError::InvalidRequestDecoderState {
                layer_index,
                description: "decoder layer entry is missing",
            },
        )?;
        model_base.commit_tree_layer(
            layer_index,
            layer_model_state,
            layer_rows,
            &kept_row_array,
            verification.context_offset,
        )?;
    }
    Ok(())
}

fn forward_tree_layer(
    target: &impl DraftTreeTarget,
    hidden_states: &MlxArray,
    tree: &TreeRowInputs,
    layer_index: usize,
    request_decoder_state: &mut RequestDecoderStateStack,
) -> Result<(MlxArray, TreeLayerRows), Qwen3_5ExecutionError> {
    let model_base = target.draft_tree_model_base();
    let decoder_layer_weights = model_base
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
    let (attention_output, rows) = model_base.forward_tree_layer_attention(
        hidden_states,
        tree,
        layer_index,
        decoder_layer_weights,
        layer_model_state,
    )?;
    let layer_output = target.forward_tree_rows_feed_forward(
        &attention_output,
        tree.row_count,
        layer_index,
        decoder_layer_weights,
    )?;
    Ok((layer_output, rows))
}

impl Qwen3_5ModelBase {
    fn tree_row_embeddings(&self, tree_rows: &TreeRows) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let mut token_ids = Vec::with_capacity(tree_rows.row_count());
        for token_id in &tree_rows.token_ids {
            token_ids.push(*token_id as i32);
        }
        let token_indices = self
            .runtime
            .array_from_i32(&token_ids, &[1, token_ids.len() as i32])?;
        self.embedding_lookup(&token_indices)
    }

    /// `[1, rows, vocabulary]` float32: the final norm and the head over
    /// every row, widened so picks compare deterministically.
    fn tree_per_row_logits(
        &self,
        hidden_states: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let normalized_states = self.runtime.rms_norm(
            hidden_states,
            &self.weights.final_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let head_logits = self.quantized_linear(
            &normalized_states,
            &self.weights.language_model_head_weights,
        )?;
        Ok(self.runtime.astype(&head_logits, MlxDtype::Float32)?)
    }

    /// The tree's positions, attention mask, convolution windows and row
    /// chains on the GPU, built once for every layer of the pass.
    fn tree_row_inputs(
        &self,
        tree_rows: &TreeRows,
        context_offset: i32,
        activation_dtype: MlxDtype,
    ) -> Result<TreeRowInputs, Qwen3_5ExecutionError> {
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
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
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

fn collected_taps(
    tap_slots: Vec<Option<MlxArray>>,
) -> Result<Vec<MlxArray>, Qwen3_5ExecutionError> {
    let mut taps = Vec::with_capacity(tap_slots.len());
    for tap_slot in tap_slots {
        taps.push(tap_slot.ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "a requested DFlash target hidden-state tap was never captured",
        })?);
    }
    Ok(taps)
}
