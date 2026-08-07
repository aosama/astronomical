use crate::{
    PrefillChunckOptimizerCandidateInsight, PrefillChunckOptimizerContextInsight,
    PrefillChunckOptimizerInsight, PrefillChunckSizeOptimizer, PrefillChunckSizeOptimizerContext,
    PrefillChunckSizeOptimizerDecisionReason,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_optimizer_insight(
    prefill_chunck_size_optimizer: &PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    requested_prefill_chunck_tokens: usize,
    actual_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
    decision_reason: PrefillChunckSizeOptimizerDecisionReason,
    has_observed_prefill_capacity_constraint: bool,
    optimizer_context_insight: PrefillChunckOptimizerContextInsight,
) -> PrefillChunckOptimizerInsight {
    let context_evidence =
        prefill_chunck_size_optimizer.context_evidence(prompt_processing_context);
    PrefillChunckOptimizerInsight {
        requested_prefill_chunck_tokens,
        actual_prefill_chunck_tokens,
        elapsed_millis,
        decision_reason,
        has_observed_prefill_capacity_constraint,
        has_observations_for_every_candidate: context_evidence.has_observations_for_every_candidate,
        context: optimizer_context_insight,
        candidate_evidence: context_evidence
            .candidate_evidence
            .into_iter()
            .map(
                |candidate_evidence| PrefillChunckOptimizerCandidateInsight {
                    candidate_prefill_chunck_tokens: candidate_evidence
                        .candidate_prefill_chunck_tokens,
                    observation_count: candidate_evidence.observation_count,
                    average_actual_prefill_chunck_tokens: candidate_evidence
                        .average_actual_prefill_chunck_tokens,
                    average_elapsed_millis: candidate_evidence.average_elapsed_millis,
                    decisions_since_last_observation: candidate_evidence
                        .decisions_since_last_observation,
                },
            )
            .collect(),
    }
}
