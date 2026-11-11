//! Unit tests for `crate::rest::openai::chat_failure_wording`: the one table
//! of sentences and codes the OpenAI-shaped surfaces share.

use astronomical_ipc_protocol::{ChatGenerationFailureReason, OUT_OF_GRAPHICS_MEMORY_ADVICE};

use super::chat_failure_wording::{OUT_OF_GRAPHICS_MEMORY_CODE, chat_failure_wording};

#[test]
fn should_give_out_of_graphics_memory_its_own_code_and_the_shared_advice() {
    let wording = chat_failure_wording(&ChatGenerationFailureReason::OutOfGraphicsMemory);

    assert_eq!(wording.code, "chat_out_of_memory");
    assert_eq!(wording.code, OUT_OF_GRAPHICS_MEMORY_CODE);
    assert_eq!(wording.message, OUT_OF_GRAPHICS_MEMORY_ADVICE);
}

/// Clients match on these codes, so a renamed one is a broken client.
#[test]
fn should_keep_the_published_code_of_every_other_failure() {
    let published_codes = [
        (
            ChatGenerationFailureReason::InvalidRequest {
                reason: "tool arguments were not an object".to_owned(),
            },
            "chat_invalid_request",
        ),
        (
            ChatGenerationFailureReason::FatalExecution {
                reason: "model execution failed inside the local worker".to_owned(),
            },
            "chat_worker_unavailable",
        ),
        (ChatGenerationFailureReason::EngineBusy, "chat_engine_busy"),
        (
            ChatGenerationFailureReason::MalformedModelOutput,
            "chat_malformed_model_output",
        ),
        (
            ChatGenerationFailureReason::ContextLengthExceeded {
                actual_total_context_tokens: 300_000,
                maximum_context_tokens: 262_144,
            },
            "context_length_exceeded",
        ),
    ];

    for (failure_reason, published_code) in published_codes {
        let wording = chat_failure_wording(&failure_reason);

        assert_eq!(wording.code, published_code);
        assert_ne!(wording.code, OUT_OF_GRAPHICS_MEMORY_CODE);
        assert!(!wording.message.is_empty());
    }
}

#[test]
fn should_carry_the_workers_own_reason_and_both_token_counts_into_the_sentence() {
    let fatal_wording = chat_failure_wording(&ChatGenerationFailureReason::FatalExecution {
        reason: "GPU allocation exceeded the platform buffer limit".to_owned(),
    });
    let overflow_wording =
        chat_failure_wording(&ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens: 300_000,
            maximum_context_tokens: 262_144,
        });

    assert!(
        fatal_wording
            .message
            .ends_with("GPU allocation exceeded the platform buffer limit")
    );
    assert!(overflow_wording.message.contains("300000 tokens"));
    assert!(overflow_wording.message.contains("262144-token"));
}

/// An empty worker reason must still leave a sentence, not a bare prefix that
/// reads as if something was cut off mid-word.
#[test]
fn should_still_form_a_sentence_when_the_worker_gives_an_empty_reason() {
    let wording = chat_failure_wording(&ChatGenerationFailureReason::InvalidRequest {
        reason: String::new(),
    });

    assert_eq!(wording.code, "chat_invalid_request");
    assert!(
        wording
            .message
            .starts_with("the local worker rejected the chat request")
    );
}
