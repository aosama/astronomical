//! A draft tree's full attention scored a few rows a call, so that every
//! row gets the bits its one-row step gives (issue #2599).

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use super::Qwen3_5ModelBase;
use super::tree_layer_attention::TreeRowInputs;

/// Query vectors one call of MLX's vector attention kernel serves, and the
/// most query rows it takes: `scaled_dot_product_attention.cpp` in MLX 0.32.4
/// uses that kernel while the rows are at most 8 and rows times query heads
/// a key-value head are at most 32, and a matrix kernel beyond.
const MLX_VECTOR_ATTENTION_QUERY_VECTORS: u32 = 32;
const MLX_VECTOR_ATTENTION_MOST_ROWS: u32 = 8;

/// The most query rows one call of MLX's attention scores with the bits a
/// one-row step gives: as many as still take the kernel a one-row step
/// takes. 5 on the dense 27B (6 query heads a key-value head), 4 on the
/// 35B-A3B (8). The by-layer journeys of `cargo measure-dflash2-row-exactness`
/// read a 16-row tree bit-equal to the one-token step in groups of 4 and 5
/// on the dense 27B, and not in groups of 6 or 8.
#[must_use]
pub fn most_tree_rows_an_exact_attention_call_takes(
    query_head_count: u32,
    key_value_head_count: u32,
) -> i32 {
    let query_heads_a_key_value_head = (query_head_count / key_value_head_count.max(1)).max(1);
    let row_count = (MLX_VECTOR_ATTENTION_QUERY_VECTORS / query_heads_a_key_value_head)
        .clamp(1, MLX_VECTOR_ATTENTION_MOST_ROWS);
    row_count as i32
}

impl Qwen3_5ModelBase {
    /// MLX's masked attention over the tree's rows, a few rows a call
    /// (issue #2599).
    ///
    /// MLX's attention changes kernels with the number of query rows, and
    /// the two kernels round differently, so a row scored among sixteen
    /// does not get the bits its one-row step gives and a near-tied pick
    /// can flip. A dense target drafts trees of about sixteen rows, so its
    /// rows are scored in groups small enough for the one-row kernel
    /// (`most_tree_rows_an_exact_attention_call_takes`) and joined: with
    /// the projections and the DeltaNet steps already row-exact, a drafted
    /// greedy reply then equals the plain one.
    ///
    /// It is not free. Dense 27B at an 18 GB limit, the two ways alternated
    /// twice in one sitting, four calls a full-attention layer against one:
    /// the verify pass cost 90 and 92 ms a round against 81 and 81, a round
    /// kept 3.19 tokens against 2.98, and the reply drafted at 31.0 and 30.3
    /// tokens a second against 31.3 and 31.6. The one-row kernel is slower a
    /// row than the matrix kernel a sixteen-row call takes; an attention
    /// kernel that is exact at sixteen rows would give the exactness back
    /// its two percent.
    pub(super) fn masked_attention_in_exact_row_groups(
        &self,
        tree: &TreeRowInputs,
        rotated_queries: &MlxArray,
        active_keys: &MlxArray,
        active_values: &MlxArray,
        scale: f32,
    ) -> Result<MlxArray, TextExecutionError> {
        let most_rows_a_call = most_tree_rows_an_exact_attention_call_takes(
            self.config.query_head_count(),
            self.config.key_value_head_count(),
        );
        let mut attended_groups = Vec::new();
        let mut first_row = 0;
        while first_row < tree.row_count {
            let end_row = (first_row + most_rows_a_call).min(tree.row_count);
            let (group_queries, group_mask) =
                self.tree_row_group(rotated_queries, tree, first_row, end_row)?;
            attended_groups.push(self.runtime.masked_scaled_dot_product_attention(
                &group_queries,
                active_keys,
                active_values,
                scale,
                &group_mask,
            )?);
            first_row = end_row;
        }
        if attended_groups.len() == 1 {
            return attended_groups
                .pop()
                .ok_or(TextExecutionError::InvalidInput {
                    description: "a draft tree has no rows to attend over",
                });
        }
        let attended_group_references: Vec<&MlxArray> = attended_groups.iter().collect();
        Ok(self
            .runtime
            .concatenate_axis(&attended_group_references, 2)?)
    }

    /// The queries and the mask rows of tree rows `first_row..end_row`. A
    /// group that is the whole tree (a mixture-of-experts target's 4-row
    /// round) takes both as they are, with no slice in its graph.
    fn tree_row_group(
        &self,
        rotated_queries: &MlxArray,
        tree: &TreeRowInputs,
        first_row: i32,
        end_row: i32,
    ) -> Result<(MlxArray, MlxArray), TextExecutionError> {
        if first_row == 0 && end_row == tree.row_count {
            return Ok((rotated_queries.retain()?, tree.attention_mask.retain()?));
        }
        let query_shape = rotated_queries.shape();
        let group_queries = self.runtime.slice(
            rotated_queries,
            &[0, 0, first_row, 0],
            &[query_shape[0], query_shape[1], end_row, query_shape[3]],
            &[1, 1, 1, 1],
        )?;
        let mask_shape = tree.attention_mask.shape();
        let mask_row_axis = mask_shape.len() - 2;
        let mut mask_starts = vec![0; mask_shape.len()];
        let mut mask_stops = mask_shape.clone();
        mask_starts[mask_row_axis] = first_row;
        mask_stops[mask_row_axis] = end_row;
        let group_mask = self.runtime.slice(
            &tree.attention_mask,
            &mask_starts,
            &mask_stops,
            &vec![1; mask_shape.len()],
        )?;
        Ok((group_queries, group_mask))
    }
}

#[cfg(test)]
#[path = "exact_tree_attention_groups_tests.rs"]
mod exact_tree_attention_groups_tests;
