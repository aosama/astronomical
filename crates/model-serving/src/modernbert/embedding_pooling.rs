//! Turns one sequence's trunk hidden states into a unit-norm embedding vector.
//!
//! The trunk produces a hidden state per token; an embedding is their mean
//! over the whole sequence, L2-normalized, and optionally truncated to a
//! requested width and normalized again. Pooling and normalization live in
//! Astronomical; no third-party embedding package is linked.

use astronomical_ipc_protocol::EmbeddingsFailureReason;
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxCError};

use crate::modernbert::token_batch::PaddedTokenBatch;
use crate::modernbert::trunk::{ModernBertTrunk, ModernBertTrunkError};

/// Computes one unit-norm pooled vector for one token sequence.
pub(super) fn embed_token_ids(
    trunk: &ModernBertTrunk<'_>,
    token_ids: &[u32],
    requested_dimensions: Option<u32>,
) -> Result<Vec<f32>, EmbeddingsFailureReason> {
    // One sequence per forward pass: a batch of one carries no padding, so
    // the trunk builds the same graph it always built for an embedding input.
    let token_batch =
        PaddedTokenBatch::from_sequences(&[token_ids], trunk.configuration.pad_token_id).map_err(
            |batch_error| EmbeddingsFailureReason::InvalidRequest {
                reason: format!("embedding input cannot be encoded: {batch_error}"),
            },
        )?;
    let final_states = trunk
        .encode_hidden_states(&token_batch)
        .map_err(trunk_failure)?;
    let runtime = trunk.runtime;
    let hidden_size = trunk.configuration.hidden_size;
    let pooled = mean_pool(runtime, &final_states, token_ids.len()).map_err(pooling_failure)?;
    let normalized = l2_normalize(runtime, &pooled).map_err(pooling_failure)?;
    let output_vector = match requested_dimensions {
        Some(requested_width) if requested_width < hidden_size => {
            let width_limited =
                slice_width(runtime, &normalized, requested_width).map_err(pooling_failure)?;
            l2_normalize(runtime, &width_limited).map_err(pooling_failure)?
        }
        _ => normalized,
    };
    output_vector.to_vec_f32().map_err(pooling_failure)
}

/// Reports a trunk failure with the messages the embeddings lane has always used.
fn trunk_failure(trunk_error: ModernBertTrunkError) -> EmbeddingsFailureReason {
    let reason = match trunk_error {
        ModernBertTrunkError::Runtime(runtime_error) => {
            format!("embedding forward pass failed: {runtime_error}")
        }
        ModernBertTrunkError::UnavailableTensor {
            tensor_name,
            reason,
        } => format!("embedding tensor {tensor_name} is unavailable: {reason}"),
    };
    bounded_fatal_execution(&reason)
}

fn pooling_failure(binding_error: MlxCError) -> EmbeddingsFailureReason {
    trunk_failure(ModernBertTrunkError::from(binding_error))
}

/// A fatal failure whose text is capped so one error cannot overrun an IPC frame.
fn bounded_fatal_execution(reason: &str) -> EmbeddingsFailureReason {
    EmbeddingsFailureReason::FatalExecution {
        reason: reason.chars().take(256).collect(),
    }
}

/// The mean of the hidden states over the whole sequence.
///
/// The upstream pipeline mean-pools over the full attention mask, which
/// includes the declared [CLS]/[SEP] specials. Reproducing the published
/// reference cosines requires the same pooling population; measured against
/// the upstream worked example, excluding the specials from the mean pushed
/// the unrelated pair's cosine higher and inverted the ordering further.
fn mean_pool(
    runtime: &MlxRuntime,
    final_states: &MlxArray,
    token_count: usize,
) -> Result<MlxArray, MlxCError> {
    let token_sum = runtime.sum_axis(final_states, 1, true)?;
    runtime.multiply_scalar(&token_sum, 1.0 / token_count as f32)
}

fn l2_normalize(runtime: &MlxRuntime, pooled: &MlxArray) -> Result<MlxArray, MlxCError> {
    let squared = runtime.multiply(pooled, pooled)?;
    let squared_sum = runtime.sum_axis(&squared, 2, true)?;
    let norm = runtime.sqrt(&squared_sum)?;
    runtime.divide(pooled, &norm)
}

fn slice_width(
    runtime: &MlxRuntime,
    normalized: &MlxArray,
    requested_width: u32,
) -> Result<MlxArray, MlxCError> {
    let requested_width_i32 = i32::try_from(requested_width).unwrap_or(i32::MAX);
    runtime.slice(
        normalized,
        &[0, 0, 0],
        &[1, 1, requested_width_i32],
        &[1, 1, 1],
    )
}

#[cfg(test)]
#[path = "embedding_pooling_tests.rs"]
mod tests;
