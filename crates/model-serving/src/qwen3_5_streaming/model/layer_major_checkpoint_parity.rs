//! Both ways of prefilling a prompt with prompt-cache checkpoints, for the
//! test that holds them bit-identical (issue #1985).
//!
//! The engine publishes a block from the decoder state at each boundary. A
//! layer-major prefill must hand it the same state the chunk path does: the
//! same tensors, bit for bit, at the same positions. These entry points run
//! the engine's planned spans each way from fresh state and copy every
//! published snapshot to the host, widened to Float32, which is exact for
//! the bfloat16 and Float32 tensors the snapshots hold.

use std::collections::BTreeMap;

use mlx_bridge::{MlxArray, MlxDtype};

use super::Qwen3_5Model;
use super::layer_major_prefill::{
    LayerMajorCheckpointContract, LayerMajorPrefillRequest, LayerMajorSpan,
};
use crate::persistent_cache::persistent_prompt_cache_boundary_completed_prefill_chunk_tokens;
use crate::qwen3_5_core::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpoint;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::inference_execution::layer_major_span_plan::layer_major_span_plan;
use crate::{PerformanceAttribution, RequestDecoderStateStack};

/// One published boundary: its token position and every snapshot tensor.
pub type PublishedBoundarySnapshot = (usize, BTreeMap<String, Vec<f32>>);

impl Qwen3_5Model {
    /// Prefills `token_ids` layer by layer over the spans the engine plans
    /// for `chunk_token_count`-token chunks and `block_token_count`-token
    /// cache blocks, returning every boundary it would publish.
    #[doc(hidden)]
    pub fn prefill_layer_major_with_prompt_cache_checkpoints(
        &self,
        token_ids: &[u32],
        chunk_token_count: usize,
        block_token_count: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<Vec<PublishedBoundarySnapshot>, Qwen3_5ExecutionError> {
        let (group_token_ids, spans) =
            planned_spans(token_ids, chunk_token_count, block_token_count)?;
        let layer_major_outcome = self.prefill_layer_major(
            group_token_ids,
            0,
            LayerMajorPrefillRequest {
                spans: &spans,
                checkpoint_contract: Some(LayerMajorCheckpointContract {
                    boundary_tensor_count: self.decoder_cache_layout.boundary_tensor_count(),
                    block_token_count,
                }),
                should_materialize_logits: false,
            },
            request_decoder_state,
            &mut PerformanceAttribution::disabled(),
        )?;
        let mut boundary_checkpoints = layer_major_outcome.boundary_checkpoints;
        push_final_state_boundary(
            &mut boundary_checkpoints,
            group_token_ids.len(),
            block_token_count,
            request_decoder_state,
        )?;
        self.host_snapshots(boundary_checkpoints)
    }

    /// Prefills the same planned spans one chunk at a time, as the engine
    /// did before layer-major prefill, returning every boundary it publishes.
    #[doc(hidden)]
    pub fn prefill_chunk_by_chunk_with_prompt_cache_checkpoints(
        &self,
        token_ids: &[u32],
        chunk_token_count: usize,
        block_token_count: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<Vec<PublishedBoundarySnapshot>, Qwen3_5ExecutionError> {
        let (group_token_ids, spans) =
            planned_spans(token_ids, chunk_token_count, block_token_count)?;
        let mut boundary_checkpoints = Vec::new();
        for span in &spans {
            let span_token_ids = &group_token_ids[span.start_token..span.end_token];
            let starting_position_tokens = u32::try_from(span.start_token).map_err(|_| {
                Qwen3_5ExecutionError::InvalidInput {
                    description: "a planned span starts beyond the u32 range",
                }
            })?;
            let mut span_checkpoints = self.prefill_planned_chunk(
                span_token_ids,
                starting_position_tokens,
                span,
                block_token_count,
                request_decoder_state,
            )?;
            for boundary_checkpoint in &mut span_checkpoints {
                boundary_checkpoint.completed_prefill_chunk_tokens += span.start_token;
            }
            boundary_checkpoints.extend(span_checkpoints);
            push_final_state_boundary(
                &mut boundary_checkpoints,
                span.end_token,
                block_token_count,
                request_decoder_state,
            )?;
        }
        self.host_snapshots(boundary_checkpoints)
    }

    /// One chunk forward, collecting the boundaries strictly inside it.
    fn prefill_planned_chunk(
        &self,
        span_token_ids: &[u32],
        starting_position_tokens: u32,
        span: &LayerMajorSpan,
        block_token_count: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>, Qwen3_5ExecutionError> {
        let mut performance_attribution = PerformanceAttribution::disabled();
        if span.interior_boundary_tokens.is_empty() {
            self.prefill_chunk_with_performance_attribution(
                span_token_ids,
                starting_position_tokens,
                request_decoder_state,
                &mut performance_attribution,
            )?;
            return Ok(Vec::new());
        }
        Ok(self
            .prefill_chunk_with_boundary_checkpoints_and_performance_attribution(
                span_token_ids,
                starting_position_tokens,
                request_decoder_state,
                span.interior_boundary_tokens.clone(),
                block_token_count,
                &mut performance_attribution,
            )?
            .boundary_checkpoints)
    }

    /// Copies every snapshot tensor to the host as Float32, in boundary order.
    fn host_snapshots(
        &self,
        boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
    ) -> Result<Vec<PublishedBoundarySnapshot>, Qwen3_5ExecutionError> {
        let mut published_boundary_snapshots = Vec::with_capacity(boundary_checkpoints.len());
        for boundary_checkpoint in boundary_checkpoints {
            let mut host_tensors = BTreeMap::new();
            for (tensor_name, snapshot_tensor) in &boundary_checkpoint.recurrent_snapshot_tensors {
                host_tensors.insert(tensor_name.clone(), self.host_float32(snapshot_tensor)?);
            }
            published_boundary_snapshots.push((
                boundary_checkpoint.completed_prefill_chunk_tokens,
                host_tensors,
            ));
        }
        published_boundary_snapshots.sort_by_key(|(boundary_token, _)| *boundary_token);
        Ok(published_boundary_snapshots)
    }

    fn host_float32(&self, snapshot_tensor: &MlxArray) -> Result<Vec<f32>, Qwen3_5ExecutionError> {
        Ok(self
            .runtime
            .astype(snapshot_tensor, MlxDtype::Float32)?
            .to_vec_f32()?)
    }
}

/// The tokens the engine's span plan covers and its spans.
fn planned_spans(
    token_ids: &[u32],
    chunk_token_count: usize,
    block_token_count: usize,
) -> Result<(&[u32], Vec<LayerMajorSpan>), Qwen3_5ExecutionError> {
    let span_plan = layer_major_span_plan(
        0,
        token_ids.len(),
        chunk_token_count,
        Some(block_token_count),
    )
    .ok_or(Qwen3_5ExecutionError::InvalidInput {
        description: "the prompt plans fewer than two spans",
    })?;
    Ok((&token_ids[..span_plan.group_end_token], span_plan.spans))
}

/// Adds the boundary a forward ending at `end_token` publishes from its
/// final state, as the engine does after each chunk, when one falls there.
fn push_final_state_boundary(
    boundary_checkpoints: &mut Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
    end_token: usize,
    block_token_count: usize,
    request_decoder_state: &RequestDecoderStateStack,
) -> Result<(), Qwen3_5ExecutionError> {
    let ends_on_boundary = persistent_prompt_cache_boundary_completed_prefill_chunk_tokens(
        0,
        end_token,
        block_token_count,
    )
    .last()
        == Some(&end_token);
    if !ends_on_boundary {
        return Ok(());
    }
    // The engine records this boundary once, from the final state; a second
    // capture here would be a duplicate publication, so it fails the check.
    if boundary_checkpoints
        .iter()
        .any(|checkpoint| checkpoint.completed_prefill_chunk_tokens == end_token)
    {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "a boundary was captured twice",
        });
    }
    boundary_checkpoints.push(Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
        completed_prefill_chunk_tokens: end_token,
        recurrent_snapshot_tensors: request_decoder_state
            .extract_persistent_prompt_cache_recurrent_snapshot_tensors()
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "the decoder state lacks a published snapshot tensor",
            })?,
    });
    Ok(())
}
