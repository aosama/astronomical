//! What a layer-major prefill keeps for each span across its layers, and the
//! astro-ssd-prompt-cache checkpoints it hands back (issue #1985).
//!
//! A span's hidden states pass from one layer to the next; its collector
//! gathers the boundaries strictly inside it; and a span that ends on a
//! boundary short of the last keeps each layer's state as the layer leaves
//! it, which is what a chunk ending there publishes.

use std::collections::HashMap;

use mlx_bridge::MlxArray;

use super::layer_major_prefill::{LayerMajorCheckpointContract, LayerMajorSpan};
use crate::AstroDecoderStateCacheState;
use crate::qwen3_5_core::decoder::{
    Qwen3_5AstroSsdPromptCacheBoundaryCheckpoint,
    Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector,
};
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

/// The per-span bookkeeping that lives across the layer loop.
pub(super) struct SpanState {
    pub(super) hidden_states: Option<MlxArray>,
    pub(super) collector: Option<Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector>,
    pub(super) end_state_snapshot: Option<HashMap<String, MlxArray>>,
}

/// One linear-attention layer's arrays within an end-state snapshot.
pub(super) fn layer_snapshot_arrays(
    end_state_snapshot: &HashMap<String, MlxArray>,
    layer_index: usize,
) -> Vec<&MlxArray> {
    let layer_tensor_prefix = format!("layer_{layer_index}_linear.");
    end_state_snapshot
        .iter()
        .filter(|(tensor_name, _)| tensor_name.starts_with(&layer_tensor_prefix))
        .map(|(_, snapshot_tensor)| snapshot_tensor)
        .collect()
}

/// Drops one linear-attention layer's arrays from an end-state snapshot, so
/// a layer whose forward failed leaves none of its unevaluated state behind
/// before it is run again.
pub(super) fn forget_layer_end_state(
    end_state_snapshot: &mut HashMap<String, MlxArray>,
    layer_index: usize,
) {
    end_state_snapshot.remove(&format!("layer_{layer_index}_linear.convolution"));
    end_state_snapshot.remove(&format!("layer_{layer_index}_linear.gated_delta_recurrent"));
}

/// A span's checkpoint collector, when it has boundaries strictly inside it.
pub(super) fn span_collector(
    span: &LayerMajorSpan,
    checkpoint_contract: Option<LayerMajorCheckpointContract>,
) -> Result<Option<Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector>, Qwen3_5ExecutionError> {
    let Some(checkpoint_contract) = checkpoint_contract else {
        return Ok(None);
    };
    if span.interior_boundary_tokens.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector::new(
            span.interior_boundary_tokens.clone(),
            checkpoint_contract.boundary_tensor_count,
            checkpoint_contract.block_token_count,
        )?,
    ))
}

/// Keeps one linear-attention layer's state as a span leaves it: the same
/// arrays a chunk ending there publishes from its final state.
pub(super) fn record_end_state_snapshot(
    end_state_snapshot: &mut HashMap<String, MlxArray>,
    layer_index: usize,
    layer_state: &AstroDecoderStateCacheState,
) -> Result<(), Qwen3_5ExecutionError> {
    let AstroDecoderStateCacheState::Composite {
        convolution,
        recurrent,
    } = layer_state
    else {
        return Ok(());
    };
    let (Some(convolution_state), Some(recurrent_state)) = (convolution.state(), recurrent.state())
    else {
        return Err(Qwen3_5ExecutionError::InvalidRequestDecoderState {
            layer_index,
            description: "a linear-attention layer left a span without its state",
        });
    };
    end_state_snapshot.insert(
        format!("layer_{layer_index}_linear.convolution"),
        convolution_state.retain()?,
    );
    end_state_snapshot.insert(
        format!("layer_{layer_index}_linear.gated_delta_recurrent"),
        recurrent_state.retain()?,
    );
    Ok(())
}

/// Every span's checkpoints, moved to positions relative to the first span.
///
/// An end-state snapshot must hold the full tensor layout, as a collector's
/// checkpoints must.
pub(super) fn completed_checkpoints(
    spans: &[LayerMajorSpan],
    span_states: Vec<SpanState>,
    checkpoint_contract: Option<LayerMajorCheckpointContract>,
) -> Result<Vec<Qwen3_5AstroSsdPromptCacheBoundaryCheckpoint>, Qwen3_5ExecutionError> {
    let expected_tensor_count =
        checkpoint_contract.map_or(0, |contract| contract.boundary_tensor_count);
    let mut boundary_checkpoints = Vec::new();
    for (span, span_state) in spans.iter().zip(span_states) {
        if let Some(collector) = span_state.collector {
            for mut boundary_checkpoint in collector.complete()? {
                boundary_checkpoint.completed_prefill_chunk_tokens += span.start_token;
                boundary_checkpoints.push(boundary_checkpoint);
            }
        }
        if let Some(recurrent_snapshot_tensors) = span_state.end_state_snapshot {
            if recurrent_snapshot_tensors.len() != expected_tensor_count {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "a layer-major end-state snapshot lacks part of the astro-decoder-state-cache tensor layout",
                });
            }
            boundary_checkpoints.push(Qwen3_5AstroSsdPromptCacheBoundaryCheckpoint {
                completed_prefill_chunk_tokens: span.end_token,
                recurrent_snapshot_tensors,
            });
        }
    }
    Ok(boundary_checkpoints)
}
