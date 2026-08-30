// Sequential one-query Scaled Dot-Product Attention (SDPA) for Multi-Token
// Prediction (MTP) verify. Native causal SDPA matches highest-logit tokens on this
// geometry, but the tiled kernel for query length two against a long prefix
// is slower than two vector passes.
use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

const SEQUENTIAL_ATTENTION_OPERATION: &str = "apply sequential target-verification attention rows";

pub(super) fn sequential_causal_attention(
    runtime: &MlxRuntime,
    rotated_queries: &MlxArray,
    active_keys: &MlxArray,
    active_values: &MlxArray,
    attention_scale: f32,
    query_token_count: i32,
    active_key_value_token_count: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let query_shape = rotated_queries.shape();
    let key_shape = active_keys.shape();
    let query_prefix_token_count = active_key_value_token_count - query_token_count;
    let mut sequential_attention_outputs = Vec::with_capacity(
        usize::try_from(query_token_count)
            .map_err(|_| sequential_attention_error("query token count exceeded host capacity"))?,
    );

    for query_row_index in 0..query_token_count {
        let query_row_end_token_count = query_row_index + 1;
        let active_key_value_end_token_count = query_prefix_token_count + query_row_end_token_count;
        let query_row = runtime.slice(
            rotated_queries,
            &[0, 0, query_row_index, 0],
            &[
                query_shape[0],
                query_shape[1],
                query_row_end_token_count,
                query_shape[3],
            ],
            &[1, 1, 1, 1],
        )?;
        let row_keys = runtime.slice(
            active_keys,
            &[0, 0, 0, 0],
            &[
                key_shape[0],
                key_shape[1],
                active_key_value_end_token_count,
                key_shape[3],
            ],
            &[1, 1, 1, 1],
        )?;
        let row_values = runtime.slice(
            active_values,
            &[0, 0, 0, 0],
            &[
                key_shape[0],
                key_shape[1],
                active_key_value_end_token_count,
                key_shape[3],
            ],
            &[1, 1, 1, 1],
        )?;
        sequential_attention_outputs.push(runtime.scaled_dot_product_attention(
            &query_row,
            &row_keys,
            &row_values,
            attention_scale,
        )?);
    }

    let sequential_attention_output_references =
        sequential_attention_outputs.iter().collect::<Vec<_>>();
    runtime.concatenate_axis(&sequential_attention_output_references, 2)
}

fn sequential_attention_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: SEQUENTIAL_ATTENTION_OPERATION,
        description: description.to_owned(),
    }
}
