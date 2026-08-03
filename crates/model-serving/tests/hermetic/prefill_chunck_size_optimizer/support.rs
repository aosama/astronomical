use astronomical_model_serving::{
    PrefillChunckSizeOptimizer, PrefillChunckSizeOptimizerContext,
    PrefillChunckSizeOptimizerDecisionReason, PrefillChunckSizeOptimizerObservation,
};

const TRUSTED_OBSERVATION_COUNT: usize = 3;
const SLIDING_WINDOW_OBSERVATION_COUNT: usize = 5;
const DRIFT_TRIGGER_FACTOR: u64 = 2;

pub(super) fn one_full_observation(
    candidate_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
) -> PrefillChunckSizeOptimizerObservation {
    PrefillChunckSizeOptimizerObservation::full_prefill_chunck(
        candidate_prefill_chunck_tokens,
        elapsed_millis,
    )
}

pub(super) fn record_full_observation(
    prefill_chunck_size_optimizer: &mut PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    candidate_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
) {
    prefill_chunck_size_optimizer
        .tell(
            prompt_processing_context,
            candidate_prefill_chunck_tokens,
            one_full_observation(candidate_prefill_chunck_tokens, elapsed_millis),
        )
        .expect("full observation should be accepted");
}

pub(super) fn record_full_observations(
    prefill_chunck_size_optimizer: &mut PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    candidate_prefill_chunck_tokens: usize,
    elapsed_millis_values: &[u64],
) {
    for &elapsed_millis in elapsed_millis_values {
        record_full_observation(
            prefill_chunck_size_optimizer,
            prompt_processing_context,
            candidate_prefill_chunck_tokens,
            elapsed_millis,
        );
    }
}

pub(super) fn ask_candidate_prefill_chunck_tokens(
    prefill_chunck_size_optimizer: &mut PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
) -> usize {
    prefill_chunck_size_optimizer
        .ask(prompt_processing_context)
        .candidate_prefill_chunck_tokens
}

pub(super) fn ask_decision(
    prefill_chunck_size_optimizer: &mut PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
) -> (usize, PrefillChunckSizeOptimizerDecisionReason) {
    let decision = prefill_chunck_size_optimizer.ask(prompt_processing_context);
    (decision.candidate_prefill_chunck_tokens, decision.reason)
}

pub(super) fn three_candidate_optimizer() -> PrefillChunckSizeOptimizer {
    PrefillChunckSizeOptimizer::new(
        vec![256, 512, 1_024],
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    )
    .expect("three candidate optimizer should be valid")
}
