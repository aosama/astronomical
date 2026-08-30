//! Dynamic 2-4 row target verification with exact prefix-boundary capture.

use std::collections::HashMap;

use crate::qwen3_5::decoder::{
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector, RequestDecoderStateStack,
};
use crate::qwen3_5::model::{
    Qwen3_5ExecutionError, Qwen3_5Model, Qwen3_5TargetForwardOutput, forward_state_arrays,
    validate_forward_input,
};
use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::{PerformanceAttribution, PerformanceOperation};

pub(crate) struct TargetVerificationOutput {
    pub(crate) target_forward_output: Qwen3_5TargetForwardOutput,
    pub(crate) target_token_ids: Vec<u32>,
    pub(crate) prefix_boundaries: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

pub(in crate::qwen3_5) fn forward_target_verification_window_with_performance_attribution(
    model: &Qwen3_5Model,
    token_ids: &[u32],
    starting_position_tokens: u32,
    request_decoder_state: &mut RequestDecoderStateStack,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<TargetVerificationOutput, Qwen3_5ExecutionError> {
    if !(2..=4).contains(&token_ids.len()) {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "target verification window requires two through four target tokens",
        });
    }
    let token_count = validate_forward_input(
        token_ids,
        starting_position_tokens,
        None,
        request_decoder_state.layer_count(),
        model.config().layer_count() as usize,
        model.config().vocabulary_size(),
        model.config().maximum_position_count(),
    )?;
    let signed_token_ids = token_ids
        .iter()
        .map(|token_id| {
            i32::try_from(*token_id).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "token ID exceeds the MLX int32 range",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let token_indices = model
        .runtime()
        .array_from_i32(&signed_token_ids, &[1, token_count])?;
    let completed_verifier_prefix_rows = (1..token_count).collect::<Vec<_>>();
    let recurrent_boundary_tensor_count = model.decoder_cache_layout().boundary_tensor_count();
    let mut boundary_collector = if recurrent_boundary_tensor_count == 0 {
        None
    } else {
        Some(
            Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(
                completed_verifier_prefix_rows
                    .iter()
                    .map(|completed_rows| *completed_rows as usize)
                    .collect(),
                recurrent_boundary_tensor_count,
                1,
            )?,
        )
    };
    let target_forward_output = model.build_target_forward_graph_from_token_indices(
        &token_indices,
        token_count,
        starting_position_tokens,
        request_decoder_state,
        boundary_collector.as_mut(),
        Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow,
        performance_attribution,
        true,
    )?;
    let all_position_logits =
        target_forward_output
            .all_position_logits()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "target verification forward did not retain all-position logits",
            })?;
    let target_token_indices = model.select_highest_logit_token(all_position_logits)?;
    let target_token_ids = performance_attribution.measure_operation(
        PerformanceOperation::MtpTargetVerificationSynchronizationWait,
        |_performance_attribution| -> Result<Vec<u32>, Qwen3_5ExecutionError> {
            let mut evaluation_roots =
                forward_state_arrays(&target_token_indices, request_decoder_state)?;
            evaluation_roots.push(target_forward_output.pre_final_normalization_hidden_states());
            if let Some(boundary_collector) = boundary_collector.as_ref() {
                evaluation_roots.extend(boundary_collector.evaluation_arrays());
            }
            model.runtime().evaluate_arrays(&evaluation_roots)?;
            Ok(target_token_indices.to_vec_u32()?)
        },
    )?;
    let prefix_boundaries = match boundary_collector {
        Some(boundary_collector) => boundary_collector.complete()?,
        None => completed_verifier_prefix_rows
            .into_iter()
            .map(
                |completed_rows| Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
                    completed_prefill_chunck_tokens: completed_rows as usize,
                    recurrent_snapshot_tensors: HashMap::new(),
                },
            )
            .collect(),
    };
    if prefix_boundaries.len() + 1 != token_ids.len() {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "target verification returned an unexpected boundary count",
        });
    }
    Ok(TargetVerificationOutput {
        target_forward_output,
        target_token_ids,
        prefix_boundaries,
    })
}
