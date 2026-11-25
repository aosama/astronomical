//! Qwen Sparse Attention: which past tokens a full-attention query reads
//! once the context outgrows the indexer's budget.
//!
//! The indexer keeps one small key per token. Every `compress_ratio` keys
//! are averaged into a block key; each query scores every complete block
//! (`sum over heads of relu(q·k) / sqrt(d)`) and keeps the best
//! `budget / compress_ratio` blocks, plus the tokens of its own unfinished
//! block. Until a query has more complete blocks than it may keep, it reads
//! every past token, so below the budget this is ordinary causal attention
//! and the Qwen 3.5 attention path serves it unchanged. The reference is
//! `Qwen4ExpQSAIndexer` (`select_from_projected` and `build_mask`).
//!
//! The selection becomes a token mask over the layer's whole key range and
//! the attention runs through MLX's masked attention: correct first, with
//! no custom kernel until a measurement shows the masked path is slow.

use mlx_bridge::{MlxArray, MlxDtype};
use text_generation::execution_error::TextExecutionError;

use crate::astro_decoder_state_cache::FullAttentionKeyValueState;
use crate::qwen3_5_core::configuration::Qwen4ExpArchitecture;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5FullAttentionWeights;

use super::centered_norm::grouped_rms_norm;
use super::growing_sequence::GrowingSequence;
use super::layer_weights::SparseAttentionIndexerWeights;
use super::request_state::IndexerLayerState;

/// The forward's queries and their positions, as the indexer needs them.
struct IndexerQueries {
    /// `[1, query heads, tokens, head dimension]`, normalized and rotated.
    rotated: MlxArray,
    /// Position of each query's token, `past + i`.
    past_token_count: i32,
    token_count: i32,
}

impl Qwen3_5ModelBase {
    /// One Flash-Next full-attention layer: the Qwen 3.5 attention, with the
    /// indexer's token mask once any query's context exceeds the budget.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_sparse_full_attention(
        &self,
        block_input: &MlxArray,
        token_count: i32,
        rope_offset_tokens: i32,
        attention_weights: &Qwen3_5FullAttentionWeights,
        indexer: &SparseAttentionIndexerWeights,
        key_value_state: &mut FullAttentionKeyValueState,
        indexer_state: &mut Option<IndexerLayerState>,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let token_mask = self.sparse_attention_token_mask(
            block_input,
            indexer,
            indexer_state,
            rope_offset_tokens,
            architecture,
        )?;
        let Some(token_mask) = token_mask else {
            return self.forward_full_attention(
                block_input,
                token_count,
                rope_offset_tokens,
                attention_weights,
                key_value_state,
                None,
            );
        };
        self.masked_full_attention(
            block_input,
            token_count,
            rope_offset_tokens,
            attention_weights,
            key_value_state,
            &token_mask,
        )
    }

    /// The indexer's `[1, 1, tokens, keys]` token mask, or `None` while every
    /// query still reads its whole context. Appends this forward's indexer
    /// keys either way.
    fn sparse_attention_token_mask(
        &self,
        block_input: &MlxArray,
        indexer: &SparseAttentionIndexerWeights,
        indexer_state: &mut Option<IndexerLayerState>,
        past_token_count: i32,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        let token_count = block_input.shape()[1];
        let (queries, new_keys) = self.indexer_projections(block_input, indexer, architecture)?;
        let layer_state = match indexer_state.take() {
            Some(mut layer_state) => {
                layer_state.keys.append(&self.runtime, &new_keys)?;
                layer_state
            }
            None => IndexerLayerState {
                keys: GrowingSequence::starting_with(&self.runtime, &new_keys, 1)?,
                block_keys: None,
            },
        };
        let layer_state = indexer_state.insert(layer_state);
        let key_count = layer_state.keys.used_rows();
        if key_count != past_token_count + token_count {
            return Err(TextExecutionError::InvalidInput {
                description: "the indexer's keys do not cover the attention's context",
            });
        }
        let complete_block_count = key_count / architecture.indexer_compress_ratio as i32;
        if complete_block_count <= architecture.indexer_selected_block_count() as i32 {
            return Ok(None);
        }
        let indexer_queries = IndexerQueries {
            rotated: self.runtime.rope(
                &queries,
                self.config.rotary_dimension() as i32,
                f32::from_bits(self.config.rope_theta_bits()),
                past_token_count,
            )?,
            past_token_count,
            token_count,
        };
        let block_keys =
            self.updated_block_keys(layer_state, indexer, complete_block_count, architecture)?;
        self.selected_token_mask(&indexer_queries, &block_keys, key_count, architecture)
            .map(Some)
    }

    /// The block keys of every complete block, computing only the blocks
    /// completed since the last forward, as the reference's cached
    /// `index_block_keys` does.
    fn updated_block_keys(
        &self,
        layer_state: &mut IndexerLayerState,
        indexer: &SparseAttentionIndexerWeights,
        complete_block_count: i32,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let first_new_block = layer_state
            .block_keys
            .as_ref()
            .map_or(0, GrowingSequence::used_rows);
        if first_new_block < complete_block_count {
            let all_keys = layer_state.keys.used(&self.runtime)?;
            let new_block_keys = self.rotated_block_keys(
                &all_keys,
                indexer,
                first_new_block,
                complete_block_count,
                architecture,
            )?;
            match layer_state.block_keys.as_mut() {
                Some(block_keys) => block_keys.append(&self.runtime, &new_block_keys)?,
                None => {
                    layer_state.block_keys = Some(GrowingSequence::starting_with(
                        &self.runtime,
                        &new_block_keys,
                        2,
                    )?);
                }
            }
        }
        let block_keys =
            layer_state
                .block_keys
                .as_ref()
                .ok_or(TextExecutionError::InvalidInput {
                    description: "the indexer has no block keys past its budget",
                })?;
        Ok(block_keys.used(&self.runtime)?)
    }

    /// Normalized queries `[1, heads, tokens, d]` and raw keys `[1, tokens, d]`.
    fn indexer_projections(
        &self,
        block_input: &MlxArray,
        indexer: &SparseAttentionIndexerWeights,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<(MlxArray, MlxArray), TextExecutionError> {
        let runtime = &self.runtime;
        let token_count = block_input.shape()[1];
        let head_count = architecture.indexer_query_head_count as i32;
        let head_dimension = architecture.indexer_head_dimension as i32;
        let projected = runtime.reshape(
            &self.quantized_linear(block_input, &indexer.query_key_projection)?,
            &[1, token_count, head_count + 1, head_dimension],
        )?;
        let raw_queries = runtime.slice(
            &projected,
            &[0, 0, 0, 0],
            &[1, token_count, head_count, head_dimension],
            &[1, 1, 1, 1],
        )?;
        let raw_keys = runtime.slice(
            &projected,
            &[0, 0, head_count, 0],
            &[1, token_count, head_count + 1, head_dimension],
            &[1, 1, 1, 1],
        )?;
        let queries = grouped_rms_norm(
            runtime,
            &raw_queries,
            &indexer.query_norm_scale,
            head_dimension,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        Ok((
            runtime.transpose_axes(&queries, &[0, 2, 1, 3])?,
            runtime.reshape(&raw_keys, &[1, token_count, head_dimension])?,
        ))
    }

    /// Blocks `first_block..end_block`'s mean keys, normalized and rotated
    /// to each block's first token: `[1, 1, blocks, d]`.
    fn rotated_block_keys(
        &self,
        all_keys: &MlxArray,
        indexer: &SparseAttentionIndexerWeights,
        first_block: i32,
        end_block: i32,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let compress_ratio = architecture.indexer_compress_ratio as i32;
        let head_dimension = architecture.indexer_head_dimension as i32;
        let block_count = end_block - first_block;
        let block_tokens = runtime.slice(
            all_keys,
            &[0, first_block * compress_ratio, 0],
            &[1, end_block * compress_ratio, head_dimension],
            &[1, 1, 1],
        )?;
        let grouped = runtime.reshape(
            &runtime.astype(&block_tokens, MlxDtype::Float32)?,
            &[1, block_count, compress_ratio, head_dimension],
        )?;
        let block_means =
            runtime.astype(&runtime.mean_axis(&grouped, 2, false)?, all_keys.dtype())?;
        let normalized = grouped_rms_norm(
            runtime,
            &block_means,
            &indexer.key_norm_scale,
            head_dimension,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let block_positions = runtime.multiply_scalar(
            &runtime.arange_i32(first_block, end_block)?,
            compress_ratio as f32,
        )?;
        let block_positions = runtime.astype(&block_positions, MlxDtype::Int32)?;
        Ok(runtime.rope_with_token_position_offsets(
            &runtime.expand_dims(&normalized, 1)?,
            &block_positions,
            self.config.rotary_dimension() as i32,
            f32::from_bits(self.config.rope_theta_bits()),
        )?)
    }

    /// Scores the blocks, keeps the best per query, and spreads the choice
    /// onto the token axis together with each query's unfinished block.
    fn selected_token_mask(
        &self,
        queries: &IndexerQueries,
        block_keys: &MlxArray,
        key_count: i32,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let compress_ratio = architecture.indexer_compress_ratio as i32;
        let kept_block_count = architecture.indexer_selected_block_count() as i32;
        let query_ends = runtime.astype(
            &runtime.add(
                &runtime.arange_i32(0, queries.token_count)?,
                &runtime.array_from_i32(&[(queries.past_token_count + 1)], &[1])?,
            )?,
            MlxDtype::Int32,
        )?;
        let complete_counts = runtime.floor_divide(
            &query_ends,
            &runtime.array_from_i32(&[compress_ratio], &[1])?,
        )?;
        let kept_blocks = self.highest_scoring_blocks(
            queries,
            block_keys,
            &complete_counts,
            kept_block_count,
            architecture,
        )?;
        let selected_tokens = self.tokens_of_kept_blocks(
            &kept_blocks,
            &complete_counts,
            queries.token_count,
            key_count,
            compress_ratio,
        )?;
        self.sparse_or_causal_mask(
            &selected_tokens,
            &query_ends,
            &complete_counts,
            key_count,
            kept_block_count,
            compress_ratio,
        )
    }

    /// `[1, tokens, kept]`: each query's best-scoring complete blocks.
    fn highest_scoring_blocks(
        &self,
        queries: &IndexerQueries,
        block_keys: &MlxArray,
        complete_counts: &MlxArray,
        kept_block_count: i32,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let block_count = block_keys.shape()[2];
        let scores = self.block_scores(queries, block_keys, architecture)?;
        let valid_blocks = runtime.less(
            &runtime.reshape(&runtime.arange_i32(0, block_count)?, &[1, 1, block_count])?,
            &runtime.reshape(complete_counts, &[1, queries.token_count, 1])?,
        )?;
        let negative_infinity = runtime.array_from_f32(&[f32::NEG_INFINITY], &[1])?;
        let scores = runtime.where_select(&valid_blocks, &scores, &negative_infinity)?;
        let partitioned = runtime.argpartition_axis(&scores, block_count - kept_block_count, -1)?;
        Ok(runtime.slice(
            &partitioned,
            &[0, 0, block_count - kept_block_count],
            &[1, queries.token_count, block_count],
            &[1, 1, 1],
        )?)
    }

    /// `[1, tokens, blocks]` float32 scores.
    fn block_scores(
        &self,
        queries: &IndexerQueries,
        block_keys: &MlxArray,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        // Which blocks win is a discrete choice, so score in float32 as the
        // reference does: rounded products flip the blocks near the cut.
        let products = runtime.matmul(
            &runtime.astype(&queries.rotated, MlxDtype::Float32)?,
            &runtime.transpose_axes(
                &runtime.astype(block_keys, MlxDtype::Float32)?,
                &[0, 1, 3, 2],
            )?,
        )?;
        let zero = runtime.full(&[], 0.0, MlxDtype::Float32)?;
        let summed = runtime.sum_axis(&runtime.maximum(&products, &zero)?, 1, false)?;
        Ok(runtime.multiply_scalar(
            &summed,
            (architecture.indexer_head_dimension as f32).sqrt().recip(),
        )?)
    }

    /// `[1, tokens, keys]`: true on every token of a kept block.
    fn tokens_of_kept_blocks(
        &self,
        kept_blocks: &MlxArray,
        complete_counts: &MlxArray,
        token_count: i32,
        key_count: i32,
        compress_ratio: i32,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let (token_indices, is_valid) = self.kept_block_token_indices(
            kept_blocks,
            complete_counts,
            token_count,
            compress_ratio,
        )?;
        // An incomplete block points at one spare column, dropped below.
        let spare_column = runtime.array_from_i32(&[key_count], &[1])?;
        let token_indices = runtime.where_select(&is_valid, &token_indices, &spare_column)?;
        let empty = runtime.zeros(&[1, token_count, key_count + 1], MlxDtype::Bool)?;
        let marked = runtime.put_along_axis(&empty, &token_indices, &is_valid, -1)?;
        Ok(runtime.slice(
            &marked,
            &[0, 0, 0],
            &[1, token_count, key_count],
            &[1, 1, 1],
        )?)
    }

    /// Every token position of every kept block, with whether its block is
    /// complete; both `[1, tokens, kept * compress_ratio]`.
    fn kept_block_token_indices(
        &self,
        kept_blocks: &MlxArray,
        complete_counts: &MlxArray,
        token_count: i32,
        compress_ratio: i32,
    ) -> Result<(MlxArray, MlxArray), TextExecutionError> {
        let runtime = &self.runtime;
        let kept_block_count = kept_blocks.shape()[2];
        let kept_blocks = runtime.astype(kept_blocks, MlxDtype::Int32)?;
        let block_is_complete = runtime.less(
            &kept_blocks,
            &runtime.reshape(complete_counts, &[1, token_count, 1])?,
        )?;
        let first_tokens = runtime.multiply_scalar(&kept_blocks, compress_ratio as f32)?;
        let token_indices = runtime.add(
            &runtime.astype(&runtime.expand_dims(&first_tokens, -1)?, MlxDtype::Int32)?,
            &runtime.reshape(
                &runtime.arange_i32(0, compress_ratio)?,
                &[1, 1, 1, compress_ratio],
            )?,
        )?;
        let token_indices = runtime.reshape(
            &token_indices,
            &[1, token_count, kept_block_count * compress_ratio],
        )?;
        let is_valid = runtime.reshape(
            &runtime.broadcast_to(
                &runtime.expand_dims(&block_is_complete, -1)?,
                &[1, token_count, kept_block_count, compress_ratio],
            )?,
            &[1, token_count, kept_block_count * compress_ratio],
        )?;
        Ok((token_indices, is_valid))
    }

    /// Kept blocks plus the unfinished block for a query past the budget,
    /// every earlier token otherwise; `[1, 1, tokens, keys]`.
    #[allow(clippy::too_many_arguments)]
    fn sparse_or_causal_mask(
        &self,
        selected_tokens: &MlxArray,
        query_ends: &MlxArray,
        complete_counts: &MlxArray,
        key_count: i32,
        kept_block_count: i32,
        compress_ratio: i32,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let token_count = query_ends.shape()[0];
        let key_positions =
            runtime.reshape(&runtime.arange_i32(0, key_count)?, &[1, 1, key_count])?;
        let query_ends = runtime.reshape(query_ends, &[1, token_count, 1])?;
        let before_query_end = runtime.less(&key_positions, &query_ends)?;
        let tail = self.unfinished_block_tokens(
            &key_positions,
            &before_query_end,
            complete_counts,
            compress_ratio,
        )?;
        let past_budget = runtime.greater(
            &runtime.reshape(complete_counts, &[1, token_count, 1])?,
            &runtime.array_from_i32(&[kept_block_count], &[1])?,
        )?;
        let sparse = runtime.logical_or(selected_tokens, &tail)?;
        let mask = runtime.where_select(&past_budget, &sparse, &before_query_end)?;
        Ok(runtime.expand_dims(&mask, 1)?)
    }

    /// `[1, tokens, keys]`: the tokens after each query's last complete
    /// block, up to the query itself.
    fn unfinished_block_tokens(
        &self,
        key_positions: &MlxArray,
        before_query_end: &MlxArray,
        complete_counts: &MlxArray,
        compress_ratio: i32,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let token_count = complete_counts.shape()[0];
        let tail_starts = runtime.astype(
            &runtime.multiply_scalar(
                &runtime.reshape(complete_counts, &[1, token_count, 1])?,
                compress_ratio as f32,
            )?,
            MlxDtype::Int32,
        )?;
        Ok(runtime.logical_and(
            &runtime.greater_equal(key_positions, &tail_starts)?,
            before_query_end,
        )?)
    }

    /// The Qwen 3.5 attention under an explicit token mask.
    fn masked_full_attention(
        &self,
        block_input: &MlxArray,
        token_count: i32,
        rope_offset_tokens: i32,
        attention_weights: &Qwen3_5FullAttentionWeights,
        key_value_state: &mut FullAttentionKeyValueState,
        token_mask: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let inputs = self.rotated_attention_inputs(
            block_input,
            token_count,
            rope_offset_tokens,
            attention_weights,
            None,
        )?;
        let (active_keys, active_values) = self.append_rotated_tensors_to_kv_state(
            key_value_state,
            &inputs.rotated_keys,
            &inputs.transposed_values,
        )?;
        let head_dimension = self.config.head_dimension() as i32;
        let attended = runtime.masked_scaled_dot_product_attention(
            &inputs.rotated_queries,
            &active_keys,
            &active_values,
            (head_dimension as f32).sqrt().recip(),
            token_mask,
        )?;
        let attended = runtime.reshape(
            &runtime.transpose_axes(&attended, &[0, 2, 1, 3])?,
            &[1, token_count, -1],
        )?;
        let gated = runtime.multiply(&attended, &runtime.sigmoid(&inputs.output_gate)?)?;
        self.quantized_linear(&gated, &attention_weights.output_projection)
    }
}
