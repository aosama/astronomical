use astronomical_model_serving::{
    PromptProcessingChunkMeasurement, PromptProcessingChunkSizeOptimizer,
    PromptProcessingMeasurementContext,
};

const MAXIMUM_RETAINED_MEASUREMENTS: usize = 5;

pub(super) fn one_chunk_measurement(
    processed_prompt_token_count: usize,
    forward_elapsed_millis: u64,
    next_measurement_context: PromptProcessingMeasurementContext,
) -> PromptProcessingChunkMeasurement {
    PromptProcessingChunkMeasurement::transition(
        processed_prompt_token_count,
        forward_elapsed_millis,
        next_measurement_context,
    )
}

pub(super) fn record_chunk_measurement(
    chunk_size_optimizer: &mut PromptProcessingChunkSizeOptimizer,
    measurement_context: PromptProcessingMeasurementContext,
    selected_candidate_chunk_size_tokens: usize,
    processed_prompt_token_count: usize,
    forward_elapsed_millis: u64,
    next_measurement_context: PromptProcessingMeasurementContext,
) {
    chunk_size_optimizer
        .record_measurement(
            measurement_context,
            selected_candidate_chunk_size_tokens,
            one_chunk_measurement(
                processed_prompt_token_count,
                forward_elapsed_millis,
                next_measurement_context,
            ),
        )
        .expect("chunk measurement should be accepted");
}

pub(super) fn record_same_context_measurements(
    chunk_size_optimizer: &mut PromptProcessingChunkSizeOptimizer,
    measurement_context: PromptProcessingMeasurementContext,
    selected_candidate_chunk_size_tokens: usize,
    forward_elapsed_millis_values: &[u64],
) {
    for &forward_elapsed_millis in forward_elapsed_millis_values {
        record_chunk_measurement(
            chunk_size_optimizer,
            measurement_context,
            selected_candidate_chunk_size_tokens,
            selected_candidate_chunk_size_tokens,
            forward_elapsed_millis,
            measurement_context,
        );
    }
}

pub(super) fn three_candidate_optimizer() -> PromptProcessingChunkSizeOptimizer {
    PromptProcessingChunkSizeOptimizer::new(vec![256, 512, 1_024], MAXIMUM_RETAINED_MEASUREMENTS)
        .expect("three candidate optimizer should be valid")
}
