use astronomical_model_serving::{
    PrefillChunckSizeOptimizer, PrefillChunckSizeOptimizerContext,
    PrefillChunckSizeOptimizerObservation,
};

const SLIDING_WINDOW_OBSERVATION_COUNT: usize = 5;

pub(super) fn one_transition_observation(
    actual_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
    next_prompt_processing_context: PrefillChunckSizeOptimizerContext,
) -> PrefillChunckSizeOptimizerObservation {
    PrefillChunckSizeOptimizerObservation::transition(
        actual_prefill_chunck_tokens,
        elapsed_millis,
        next_prompt_processing_context,
    )
}

pub(super) fn record_transition_observation(
    prefill_chunck_size_optimizer: &mut PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    candidate_prefill_chunck_tokens: usize,
    actual_prefill_chunck_tokens: usize,
    elapsed_millis: u64,
    next_prompt_processing_context: PrefillChunckSizeOptimizerContext,
) {
    prefill_chunck_size_optimizer
        .tell(
            prompt_processing_context,
            candidate_prefill_chunck_tokens,
            one_transition_observation(
                actual_prefill_chunck_tokens,
                elapsed_millis,
                next_prompt_processing_context,
            ),
        )
        .expect("transition observation should be accepted");
}

pub(super) fn record_self_transition_observations(
    prefill_chunck_size_optimizer: &mut PrefillChunckSizeOptimizer,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    candidate_prefill_chunck_tokens: usize,
    elapsed_millis_values: &[u64],
) {
    for &elapsed_millis in elapsed_millis_values {
        record_transition_observation(
            prefill_chunck_size_optimizer,
            prompt_processing_context,
            candidate_prefill_chunck_tokens,
            candidate_prefill_chunck_tokens,
            elapsed_millis,
            prompt_processing_context,
        );
    }
}

pub(super) fn three_candidate_optimizer() -> PrefillChunckSizeOptimizer {
    PrefillChunckSizeOptimizer::new(vec![256, 512, 1_024], SLIDING_WINDOW_OBSERVATION_COUNT)
        .expect("three candidate optimizer should be valid")
}
