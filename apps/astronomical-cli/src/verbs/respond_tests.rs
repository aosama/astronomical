//! Unit contracts for the words `respond` prints when the worker reports that
//! a generation failed.

use astronomical_ipc_protocol::{ChatGenerationFailureReason, OUT_OF_GRAPHICS_MEMORY_ADVICE};

use super::chat_generation_failure_reason_text;

/// The terminal is the only place a command-line user reads this, so it must
/// carry the advice itself and not send them to a log.
#[test]
fn should_tell_a_command_line_user_what_to_do_when_graphics_memory_runs_out() {
    let printed_reason =
        chat_generation_failure_reason_text(&ChatGenerationFailureReason::OutOfGraphicsMemory);

    assert_eq!(printed_reason, OUT_OF_GRAPHICS_MEMORY_ADVICE);
    assert!(printed_reason.contains("close other apps"));
    assert!(printed_reason.contains("use a smaller model"));
    assert!(!printed_reason.contains("log"));
}

#[test]
fn should_print_the_workers_own_bounded_reason_for_other_fatal_failures() {
    let printed_reason =
        chat_generation_failure_reason_text(&ChatGenerationFailureReason::FatalExecution {
            reason: "model execution failed inside the local worker".to_owned(),
        });

    assert_eq!(
        printed_reason,
        "model execution failed inside the local worker"
    );
}

#[test]
fn should_assemble_the_context_overflow_sentence_from_both_token_counts() {
    let printed_reason =
        chat_generation_failure_reason_text(&ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens: 300_000,
            maximum_context_tokens: 262_144,
        });

    assert!(printed_reason.contains("300000 of 262144 tokens"));
}
