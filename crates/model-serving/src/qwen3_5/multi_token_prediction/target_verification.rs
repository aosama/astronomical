//! Dynamic 2-4 row target verification with exact prefix-boundary capture.
//!
//! The window forward is shared between the greedy (argmax) and sampled
//! (`min(1, p/q)`) verifiers; each wrapper evaluates the window's lazy graph
//! with its own decision arrays and completes the boundary checkpoints after
//! that evaluation so every recurrent snapshot tensor is materialized.

use std::collections::HashMap;

use astronomical_mlx_c_rust::MlxArray;

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

/// One verification-window forward before any token selection.
pub(super) struct MtpVerificationWindow {
    pub(super) target_forward_output: Qwen3_5TargetForwardOutput,
    pub(super) boundary_collector: Option<Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector>,
    pub(super) completed_verifier_prefix_rows: Vec<i32>,
}

/// Materializes the window's boundary snapshots after the caller's evaluation.
pub(super) fn completed_verification_prefix_boundaries(
    boundary_collector: Option<Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector>,
    completed_verifier_prefix_rows: &[i32],
    token_ids: &[u32],
) -> Result<Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>, Qwen3_5ExecutionError> {
    let prefix_boundaries = match boundary_collector {
        Some(boundary_collector) => boundary_collector.complete()?,
        None => completed_verifier_prefix_rows
            .iter()
            .map(
                |completed_rows| Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
                    completed_prefill_chunk_tokens: *completed_rows as usize,
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
    Ok(prefix_boundaries)
}

pub(in crate::qwen3_5) fn forward_mtp_verification_window_with_performance_attribution(
    model: &Qwen3_5Model,
    token_ids: &[u32],
    starting_position_tokens: u32,
    request_decoder_state: &mut RequestDecoderStateStack,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MtpVerificationWindow, Qwen3_5ExecutionError> {
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
    if target_forward_output.all_position_logits().is_none() {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "target verification forward did not retain all-position logits",
        });
    }
    Ok(MtpVerificationWindow {
        target_forward_output,
        boundary_collector,
        completed_verifier_prefix_rows,
    })
}

/// Completes target-token selection and boundary materialization for a window
/// produced by either the eager or compiled forward.
pub(in crate::qwen3_5) fn complete_mtp_verification_window_with_performance_attribution(
    model: &Qwen3_5Model,
    verification_window: MtpVerificationWindow,
    token_ids: &[u32],
    request_decoder_state: &RequestDecoderStateStack,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<TargetVerificationOutput, Qwen3_5ExecutionError> {
    let MtpVerificationWindow {
        target_forward_output,
        boundary_collector,
        completed_verifier_prefix_rows,
    } = verification_window;
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
    let prefix_boundaries = completed_verification_prefix_boundaries(
        boundary_collector,
        &completed_verifier_prefix_rows,
        token_ids,
    )?;
    Ok(TargetVerificationOutput {
        target_forward_output,
        target_token_ids,
        prefix_boundaries,
    })
}

impl crate::qwen3_5::model::Qwen3_5Model {
    /// Test-facing eager verification window returning materialized float32
    /// all-position logits — the A/B reference for the compiled lane.
    #[doc(hidden)]
    pub fn eager_verification_window_logits_for_tests(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        let verification_window = forward_mtp_verification_window_with_performance_attribution(
            self,
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            &mut disabled_performance_attribution,
        )?;
        let all_position_logits = verification_window
            .target_forward_output
            .all_position_logits()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "the eager verification window retained no all-position logits",
            })?;
        self.runtime()
            .evaluate_arrays(&[all_position_logits])
            .map_err(Qwen3_5ExecutionError::from)?;
        Ok(all_position_logits
            .retain()
            .map_err(Qwen3_5ExecutionError::from)?)
    }
}

pub(in crate::qwen3_5) fn forward_target_verification_window_with_performance_attribution(
    model: &Qwen3_5Model,
    token_ids: &[u32],
    starting_position_tokens: u32,
    request_decoder_state: &mut RequestDecoderStateStack,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<TargetVerificationOutput, Qwen3_5ExecutionError> {
    let verification_window = forward_mtp_verification_window_with_performance_attribution(
        model,
        token_ids,
        starting_position_tokens,
        request_decoder_state,
        performance_attribution,
    )?;
    complete_mtp_verification_window_with_performance_attribution(
        model,
        verification_window,
        token_ids,
        request_decoder_state,
        performance_attribution,
    )
}

/// Runs the compiled verification window and assembles the shared
/// [`MtpVerificationWindow`] shape the sampled verifier consumes, declining
/// with `Err` whenever the compiled lane does not fit.
pub(in crate::qwen3_5) fn compiled_mtp_verification_window_with_performance_attribution(
    model: &Qwen3_5Model,
    token_ids: &[u32],
    starting_position_tokens: u32,
    request_decoder_state: &mut RequestDecoderStateStack,
) -> Result<MtpVerificationWindow, Qwen3_5ExecutionError> {
    if !(2..=4).contains(&token_ids.len()) {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "the compiled verification window requires two through four tokens",
        });
    }
    let compiled_window = model
        .run_compiled_verification_window(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
        )
        .map_err(|description| Qwen3_5ExecutionError::InvalidDecoderCacheLayout { description })?;
    let token_count =
        i32::try_from(token_ids.len()).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
            description: "the verification window exceeds the Int32 row range",
        })?;
    let target_forward_output = Qwen3_5TargetForwardOutput::from_all_position_logits(
        model.runtime(),
        compiled_window.all_position_logits,
        compiled_window.pre_final_normalization_hidden_states,
        token_count,
        model.config().vocabulary_size() as i32,
    )
    .map_err(Qwen3_5ExecutionError::from)?;
    Ok(MtpVerificationWindow {
        target_forward_output,
        boundary_collector: compiled_window.boundary_collector,
        completed_verifier_prefix_rows: (1..token_count).collect(),
    })
}
