use astronomical_ipc_protocol::{
    WorkerPrefillOptimizerCandidateEvidence, WorkerPrefillOptimizerContext,
    WorkerPrefillOptimizerDecisionReason, WorkerPrefillOptimizerInsight,
};

use super::WorkerRuntimeError;
use crate::{PrefillChunckOptimizerInsight, PrefillChunckSizeOptimizerDecisionReason};

pub(super) fn to_worker_prefill_optimizer_insight(
    prefill_optimizer_insight: PrefillChunckOptimizerInsight,
) -> Result<WorkerPrefillOptimizerInsight, WorkerRuntimeError> {
    Ok(WorkerPrefillOptimizerInsight {
        requested_prefill_chunck_tokens: bounded_token_count(
            prefill_optimizer_insight.requested_prefill_chunck_tokens,
        )?,
        actual_prefill_chunck_tokens: bounded_token_count(
            prefill_optimizer_insight.actual_prefill_chunck_tokens,
        )?,
        elapsed_millis: prefill_optimizer_insight.elapsed_millis,
        decision_reason: match prefill_optimizer_insight.decision_reason {
            PrefillChunckSizeOptimizerDecisionReason::InitialExploration => {
                WorkerPrefillOptimizerDecisionReason::InitialExploration
            }
            PrefillChunckSizeOptimizerDecisionReason::StaleObservationProbe => {
                WorkerPrefillOptimizerDecisionReason::StaleObservationProbe
            }
            PrefillChunckSizeOptimizerDecisionReason::CumulativeLatencyPlanning => {
                WorkerPrefillOptimizerDecisionReason::CumulativeLatencyPlanning
            }
            PrefillChunckSizeOptimizerDecisionReason::Fallback => {
                WorkerPrefillOptimizerDecisionReason::Fallback
            }
            PrefillChunckSizeOptimizerDecisionReason::TerminalRemainder => {
                WorkerPrefillOptimizerDecisionReason::TerminalRemainder
            }
        },
        has_observed_prefill_capacity_constraint: prefill_optimizer_insight
            .has_observed_prefill_capacity_constraint,
        has_observations_for_every_candidate: prefill_optimizer_insight
            .has_observations_for_every_candidate,
        context: WorkerPrefillOptimizerContext {
            prompt_position_tokens: bounded_token_count(
                prefill_optimizer_insight.context.prompt_position_tokens,
            )?,
            has_restored_prefix: prefill_optimizer_insight.context.has_restored_prefix,
            is_first_chunck_after_restore: prefill_optimizer_insight
                .context
                .is_first_chunck_after_restore,
            has_visual_embeddings: prefill_optimizer_insight.context.has_visual_embeddings,
            is_mtp_active: prefill_optimizer_insight.context.is_mtp_active,
            are_sparse_experts_paged: prefill_optimizer_insight.context.are_sparse_experts_paged,
            is_prompt_cache_capture_eligible: prefill_optimizer_insight
                .context
                .is_prompt_cache_capture_eligible,
            has_prior_capacity_reduction: prefill_optimizer_insight
                .context
                .has_prior_capacity_reduction,
        },
        candidate_evidence: prefill_optimizer_insight
            .candidate_evidence
            .into_iter()
            .map(|candidate_evidence| {
                Ok(WorkerPrefillOptimizerCandidateEvidence {
                    candidate_prefill_chunck_tokens: bounded_token_count(
                        candidate_evidence.candidate_prefill_chunck_tokens,
                    )?,
                    observation_count: u32::try_from(candidate_evidence.observation_count)
                        .unwrap_or(u32::MAX),
                    average_actual_prefill_chunck_tokens: bounded_token_count(
                        candidate_evidence.average_actual_prefill_chunck_tokens,
                    )?,
                    average_elapsed_millis: candidate_evidence.average_elapsed_millis,
                    decisions_since_last_observation: candidate_evidence
                        .decisions_since_last_observation,
                })
            })
            .collect::<Result<Vec<_>, WorkerRuntimeError>>()?,
    })
}

fn bounded_token_count(token_count: usize) -> Result<u32, WorkerRuntimeError> {
    u32::try_from(token_count).map_err(|_| WorkerRuntimeError::InferenceEngineGenerationFailed {
        reason: "prefill optimizer telemetry token count exceeds the u32 range".to_owned(),
    })
}
