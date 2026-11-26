//! A dense target's one-token step attended through the row-exact kernel,
//! so that a draft tree's rows, which the same kernel verifies in one pass,
//! get the bits their one-token steps give (issue #2599).
//!
//! MLX's attention changes kernels with the number of query rows, so a tree
//! verified in one MLX call does not match the one-token step. Scoring the
//! tree a few rows a call does match, and reads the whole context once a
//! call: over the dense 27B's 16 attention layers a 16-row tree cost 34.6 ms
//! that way at 2,048 keys and 61.7 ms at 10,240, against 19.2 and 33.9 ms
//! for one inexact call. The row-exact kernel costs 11.0 and 18.1 ms for the
//! tree and is the same arithmetic for one row, so it serves both. Its one
//! row costs more than MLX's (8.1 against 6.4 ms at 2,048 keys, 7.9 against
//! 5.5 at 10,240; `cargo measure-row-exact-tree-attention`), which a dense
//! target that hosts a drafter pays on the tokens it does not draft.
//!
//! A mixture-of-experts target keeps MLX's attention for its one-token
//! step: its expert pass rounds a tree's rows differently already, so its
//! drafted reply is held to rounding either way.

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use super::Qwen3_5ModelBase;
use crate::astro_decoder_state_cache::FullAttentionKeyValueState;
use crate::qwen3_5_core::configuration::Qwen3_5FeedForwardArchitecture;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5FullAttentionWeights;
use crate::qwen3_5_core::model_math::gated_delta_tree::TreeRowChains;
use crate::qwen3_5_core::model_math::row_exact_tree_attention::{
    RowExactTreeAttentionInputs, row_exact_tree_attention, row_exact_tree_attention_accepts,
};

impl Qwen3_5ModelBase {
    /// Whether this target attends row-exactly at every context length, for
    /// its one-token steps and its trees alike: a dense target whose kernels
    /// were enabled when its drafter bound. Both must take the same kernel
    /// at the same positions, so a dense target has no context threshold.
    pub(super) fn attends_row_exactly_at_every_position(&self) -> bool {
        self.optional_accelerators
            .row_exact_tree_attention_kernels
            .is_some()
            && self.config.feed_forward_architecture() == Qwen3_5FeedForwardArchitecture::Dense
    }

    /// A forward's attention half once its keys and values are stored: a
    /// one-token step of a target that attends row-exactly through the
    /// kernel, and everything else through MLX's attention.
    pub(super) fn attended_gated_and_projected(
        &self,
        (rotated_queries, output_gate): (&MlxArray, &MlxArray),
        (active_keys, active_values): (&MlxArray, &MlxArray),
        kv_state: &FullAttentionKeyValueState,
        full_attention_weights: &Qwen3_5FullAttentionWeights,
    ) -> Result<MlxArray, TextExecutionError> {
        if let Some(row_exact_output) = self.one_token_step_attended_row_exactly(
            rotated_queries,
            kv_state,
            output_gate,
            full_attention_weights,
        )? {
            return Ok(row_exact_output);
        }
        self.run_full_attention_computation(
            rotated_queries,
            active_keys,
            active_values,
            output_gate,
            (self.config.head_dimension() as f32).sqrt().recip(),
            &full_attention_weights.output_projection,
        )
    }

    /// A one-token step's whole attention half after its key and value are
    /// stored: the row-exact kernel's attention, Qwen's output gate and the
    /// output projection. `None` when this target's one-token step takes
    /// MLX's attention.
    fn one_token_step_attended_row_exactly(
        &self,
        rotated_queries: &MlxArray,
        kv_state: &FullAttentionKeyValueState,
        output_gate: &MlxArray,
        full_attention_weights: &Qwen3_5FullAttentionWeights,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        let scale = (self.config.head_dimension() as f32).sqrt().recip();
        let Some(attended) =
            self.one_row_attention_through_the_row_exact_kernel(rotated_queries, kv_state, scale)?
        else {
            return Ok(None);
        };
        // Tokens before heads, then every head's features side by side, and
        // the decode step's own gate: sigmoid, then multiply.
        let merged_heads = self.runtime.reshape(
            &self.runtime.transpose_axes(&attended, &[0, 2, 1, 3])?,
            &[1, 1, output_gate.shape()[2]],
        )?;
        let gated = self
            .runtime
            .multiply(&merged_heads, &self.runtime.sigmoid(output_gate)?)?;
        Ok(Some(self.quantized_linear(
            &gated,
            &full_attention_weights.output_projection,
        )?))
    }

    /// One token's attention over the stored keys, the token's own included,
    /// through the row-exact kernel, with heads still apart:
    /// `[1, query heads, 1, head size]`. `None` when this target's one-token
    /// step takes MLX's attention, or the kernel does not take these inputs.
    fn one_row_attention_through_the_row_exact_kernel(
        &self,
        rotated_queries: &MlxArray,
        kv_state: &FullAttentionKeyValueState,
        scale: f32,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        if rotated_queries.shape()[2] != 1 || !self.attends_row_exactly_at_every_position() {
            return Ok(None);
        }
        let (Some(kernels), Some(key_slab), Some(value_slab)) = (
            self.optional_accelerators
                .row_exact_tree_attention_kernels
                .as_ref(),
            kv_state.keys_state(),
            kv_state.values_state(),
        ) else {
            return Ok(None);
        };
        // The token's own key and value are already stored: it is a tree of
        // one row whose context is everything before it.
        let single_row_chain = TreeRowChains::from_parent_rows(&[None]).upload(&self.runtime)?;
        let kernel_inputs = RowExactTreeAttentionInputs {
            queries: rotated_queries,
            key_slab,
            value_slab,
            context_length: kv_state.offset_tokens() - 1,
            row_chains: &single_row_chain,
            scale,
        };
        if !row_exact_tree_attention_accepts(&kernel_inputs) {
            return Ok(None);
        }
        Ok(Some(row_exact_tree_attention(
            &self.runtime,
            kernels,
            &kernel_inputs,
        )?))
    }
}
