use super::super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
use super::{
    can_redo_prefill_after_memory_failure, known_memory_shortfall_bytes, ran_out_of_memory,
};

#[test]
fn should_read_the_shortfall_from_a_limit_failure() {
    assert_eq!(
        known_memory_shortfall_bytes(&PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
            active_memory_bytes: 9,
            attempted_allocation_bytes: 4,
            allowed_active_memory_bytes: 10,
        }),
        3
    );
}

#[test]
fn should_report_no_shortfall_when_the_figures_fit_the_limit() {
    assert_eq!(
        known_memory_shortfall_bytes(&PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
            active_memory_bytes: 5,
            attempted_allocation_bytes: 5,
            allowed_active_memory_bytes: 10,
        }),
        0
    );
}

#[test]
fn should_report_no_shortfall_for_a_failure_without_figures() {
    assert_eq!(
        known_memory_shortfall_bytes(
            &PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted {
                reason: "out of memory".to_owned(),
            }
        ),
        0
    );
}

#[test]
fn should_redo_a_text_request_that_ran_out_of_memory() {
    assert!(can_redo_prefill_after_memory_failure(false));
}

#[test]
fn should_never_redo_a_request_with_images() {
    assert!(!can_redo_prefill_after_memory_failure(true));
}

#[test]
fn should_tell_a_limit_failure_from_an_engine_failure() {
    assert!(ran_out_of_memory(
        &PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
            active_memory_bytes: 9,
            attempted_allocation_bytes: 2,
            allowed_active_memory_bytes: 10,
        }
    ));
    assert!(ran_out_of_memory(
        &PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted {
            reason: "out of memory".to_owned(),
        }
    ));
    assert!(!ran_out_of_memory(&PromptPrefillChunkAttemptError::Engine(
        crate::qwen3_5_text::inference_execution::fatal_engine_error("lost the model"),
    )));
}
