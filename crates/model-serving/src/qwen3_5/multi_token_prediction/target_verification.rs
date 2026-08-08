use crate::{PerformanceAttribution, PerformanceOperation};

use crate::qwen3_5::decoder::{
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector, RequestDecoderStateStack,
};
use crate::qwen3_5::model::{
    Qwen3_5ExecutionError, Qwen3_5Model, Qwen3_5TargetForwardOutput, forward_state_arrays,
    validate_forward_input,
};
use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;

pub(in crate::qwen3_5) fn forward_target_verification_window_with_performance_attribution(
    model: &Qwen3_5Model,
    token_ids: &[u32],
    starting_position_tokens: u32,
    request_decoder_state: &mut RequestDecoderStateStack,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<
    (
        Qwen3_5TargetForwardOutput,
        Vec<u32>,
        Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    ),
    Qwen3_5ExecutionError,
> {
    if token_ids.len() != 2 {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "target verification window requires exactly two target tokens",
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
    let recurrent_boundary_tensor_count = model.decoder_cache_layout().boundary_tensor_count();
    let mut verified_prefix_boundary_checkpoint_collector = if recurrent_boundary_tensor_count == 0
    {
        None
    } else {
        Some(
            Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(
                vec![1],
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
        verified_prefix_boundary_checkpoint_collector.as_mut(),
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
    let target_verify_token_indices = model.build_greedy_token(all_position_logits)?;
    let target_verify_token_ids = performance_attribution.measure_operation(
        PerformanceOperation::MtpTargetVerificationSynchronizationWait,
        |_performance_attribution| -> Result<Vec<u32>, Qwen3_5ExecutionError> {
            let mut target_verification_evaluation_arrays =
                forward_state_arrays(&target_verify_token_indices, request_decoder_state)?;
            target_verification_evaluation_arrays
                .push(target_forward_output.pre_final_normalization_hidden_states());
            if let Some(verified_prefix_boundary_checkpoint_collector) =
                verified_prefix_boundary_checkpoint_collector.as_ref()
            {
                target_verification_evaluation_arrays
                    .extend(verified_prefix_boundary_checkpoint_collector.evaluation_arrays());
            }
            model
                .runtime()
                .evaluate_arrays(&target_verification_evaluation_arrays)?;
            Ok(target_verify_token_indices.to_vec_u32()?)
        },
    )?;
    let verified_prefix_boundary_checkpoint = match verified_prefix_boundary_checkpoint_collector {
        Some(verified_prefix_boundary_checkpoint_collector) => {
            let mut verified_prefix_boundary_checkpoints =
                verified_prefix_boundary_checkpoint_collector.complete()?;
            let verified_prefix_boundary_checkpoint = verified_prefix_boundary_checkpoints
                .pop()
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "target verification did not retain its first-row boundary",
                })?;
            if !verified_prefix_boundary_checkpoints.is_empty() {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "target verification retained unexpected extra boundaries",
                });
            }
            verified_prefix_boundary_checkpoint
        }
        None => Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
            completed_prefill_chunck_tokens: 1,
            recurrent_snapshot_tensors: std::collections::HashMap::new(),
        },
    };
    Ok((
        target_forward_output,
        target_verify_token_ids,
        verified_prefix_boundary_checkpoint,
    ))
}
