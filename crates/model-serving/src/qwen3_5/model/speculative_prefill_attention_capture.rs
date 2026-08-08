use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError};

use super::Qwen3_5ExecutionError;

/// Full-attention tensors retained only for draft importance scoring.
pub(crate) struct Qwen3_5AttentionCapture {
    prompt_keys_by_layer: Vec<Option<MlxArray>>,
    lookahead_queries_by_layer: Vec<Vec<MlxArray>>,
    is_capturing_lookahead_queries: bool,
}

impl Qwen3_5AttentionCapture {
    pub(crate) fn new(decoder_layer_count: usize) -> Self {
        Self {
            prompt_keys_by_layer: (0..decoder_layer_count).map(|_| None).collect(),
            lookahead_queries_by_layer: (0..decoder_layer_count).map(|_| Vec::new()).collect(),
            is_capturing_lookahead_queries: false,
        }
    }

    pub(crate) fn begin_lookahead_capture(&mut self) {
        self.is_capturing_lookahead_queries = true;
    }

    pub(crate) fn record_full_attention_tensors(
        &mut self,
        decoder_layer_index: usize,
        rotated_queries: &MlxArray,
        active_keys: &MlxArray,
    ) -> Result<(), MlxRuntimeError> {
        if self.is_capturing_lookahead_queries {
            let lookahead_queries = self
                .lookahead_queries_by_layer
                .get_mut(decoder_layer_index)
                .ok_or_else(|| capture_error("decoder layer exceeded query capture capacity"))?;
            lookahead_queries.push(rotated_queries.retain()?);
        } else {
            let prompt_keys = self
                .prompt_keys_by_layer
                .get_mut(decoder_layer_index)
                .ok_or_else(|| capture_error("decoder layer exceeded key capture capacity"))?;
            *prompt_keys = Some(active_keys.retain()?);
        }
        Ok(())
    }

    #[must_use]
    pub(crate) fn prompt_keys_for_layer(&self, decoder_layer_index: usize) -> Option<&MlxArray> {
        self.prompt_keys_by_layer
            .get(decoder_layer_index)
            .and_then(Option::as_ref)
    }

    #[must_use]
    pub(crate) fn lookahead_queries_for_layer(
        &self,
        decoder_layer_index: usize,
    ) -> Option<&[MlxArray]> {
        self.lookahead_queries_by_layer
            .get(decoder_layer_index)
            .map(Vec::as_slice)
    }
}

pub fn qwen3_5_aggregate_speculative_prefill_attention_weights(
    runtime: &MlxRuntime,
    combined_layer_head_attention_weights: &MlxArray,
    importance_pooling_kernel_token_count: usize,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let attention_weight_shape = combined_layer_head_attention_weights.shape();
    if attention_weight_shape.len() != 3
        || attention_weight_shape
            .iter()
            .any(|dimension| *dimension <= 0)
    {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill attention weights must be layer-head by lookahead by prompt",
        });
    }
    let layer_head_count = attention_weight_shape[0];
    let lookahead_token_count = attention_weight_shape[1];
    let prompt_token_count = attention_weight_shape[2];
    let flattened_attention_map_count = layer_head_count.checked_mul(lookahead_token_count).ok_or(
        Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill attention map count overflowed",
        },
    )?;
    let flattened_attention_weights = runtime.reshape(
        combined_layer_head_attention_weights,
        &[flattened_attention_map_count, prompt_token_count],
    )?;
    let smoothed_flattened_attention_weights = average_pool_speculative_prefill_scores(
        runtime,
        &flattened_attention_weights,
        importance_pooling_kernel_token_count,
    )?;
    let smoothed_attention_weights = runtime.reshape(
        &smoothed_flattened_attention_weights,
        &[layer_head_count, lookahead_token_count, prompt_token_count],
    )?;
    let maximum_layer_head_scores = runtime.max_axis(&smoothed_attention_weights, 0, false)?;
    let summed_lookahead_scores = runtime.sum_axis(&maximum_layer_head_scores, 0, false)?;
    runtime
        .multiply_scalar(&summed_lookahead_scores, 1.0 / lookahead_token_count as f32)
        .map_err(Into::into)
}

fn average_pool_speculative_prefill_scores(
    runtime: &MlxRuntime,
    scores: &MlxArray,
    pooling_kernel_token_count: usize,
) -> Result<MlxArray, MlxRuntimeError> {
    if pooling_kernel_token_count <= 1 {
        return scores.retain();
    }
    let score_shape = scores.shape();
    if score_shape.len() != 2 || score_shape[0] <= 0 || score_shape[1] <= 0 {
        return Err(capture_error(
            "importance scores must be a non-empty lookahead-by-prompt matrix",
        ));
    }
    let lookahead_token_count = score_shape[0];
    let prompt_token_count = score_shape[1];
    let pooling_kernel_token_count_i32 = i32::try_from(pooling_kernel_token_count)
        .map_err(|_| capture_error("importance pooling kernel exceeds the MLX integer range"))?;
    let left_padding_token_count = pooling_kernel_token_count / 2;
    let right_padding_token_count = pooling_kernel_token_count
        .checked_sub(1 + left_padding_token_count)
        .ok_or_else(|| capture_error("importance pooling padding arithmetic underflowed"))?;
    let left_padding = runtime.zeros(
        &[
            lookahead_token_count,
            i32::try_from(left_padding_token_count)
                .map_err(|_| capture_error("left pooling padding exceeds the MLX range"))?,
        ],
        MlxDtype::Float32,
    )?;
    let right_padding = runtime.zeros(
        &[
            lookahead_token_count,
            i32::try_from(right_padding_token_count)
                .map_err(|_| capture_error("right pooling padding exceeds the MLX range"))?,
        ],
        MlxDtype::Float32,
    )?;
    let padded_scores = runtime.concatenate_axis(&[&left_padding, scores, &right_padding], 1)?;
    let prefix_zeroes = runtime.zeros(&[lookahead_token_count, 1], MlxDtype::Float32)?;
    let prefix_input = runtime.concatenate_axis(&[&prefix_zeroes, &padded_scores], 1)?;
    let prefix_scores = runtime.cumsum(&prefix_input, 1, false, true)?;
    let padded_prefix_end = prompt_token_count
        .checked_add(pooling_kernel_token_count_i32)
        .ok_or_else(|| capture_error("importance pooling shape arithmetic overflowed"))?;
    let window_end_scores = runtime.slice(
        &prefix_scores,
        &[0, pooling_kernel_token_count_i32],
        &[
            lookahead_token_count,
            i32::try_from(padded_prefix_end).map_err(|_| {
                capture_error("importance pooling prefix shape exceeds the MLX range")
            })?,
        ],
        &[1, 1],
    )?;
    let window_start_scores = runtime.slice(
        &prefix_scores,
        &[0, 0],
        &[lookahead_token_count, prompt_token_count],
        &[1, 1],
    )?;
    let window_sums = runtime.subtract(&window_end_scores, &window_start_scores)?;
    runtime.multiply_scalar(&window_sums, 1.0 / pooling_kernel_token_count as f32)
}

fn capture_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "capture Qwen3.5 speculative-prefill attention tensors",
        description: description.to_owned(),
    }
}
