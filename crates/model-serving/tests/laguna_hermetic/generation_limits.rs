//! Request output capability and total-context admission contracts for Laguna.

use astronomical_model_serving::{LagunaGenerationProcessor, LagunaPreparationError};

use super::text_support::{
    SYNTHETIC_LAGUNA_MODEL_ID, SyntheticLagunaTextArtifact, romeo_and_juliet_command,
};

#[test]
fn should_accept_an_explicit_output_budget_larger_than_a_request_default_when_context_fits() {
    let text_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    let maximum_context_tokens = text_descriptor.maximum_context_tokens();
    let processor = LagunaGenerationProcessor::new_with_performance_attribution(
        SYNTHETIC_LAGUNA_MODEL_ID,
        text_descriptor,
        maximum_context_tokens,
        u32::from(u16::MAX).min(maximum_context_tokens - 1),
        false,
    )
    .expect("the protocol-bounded model capability should construct a processor");
    let mut chat_command = romeo_and_juliet_command(9_818, None);
    chat_command.settings.max_output_tokens = 20_000;

    processor
        .prepare_chat(&chat_command)
        .expect("a request larger than a small configured default should fit the model context");
}

#[test]
fn should_reject_prompt_plus_requested_output_context_overflow_without_truncation() {
    let text_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    let baseline_processor =
        LagunaGenerationProcessor::new(SYNTHETIC_LAGUNA_MODEL_ID, text_descriptor.clone())
            .expect("the baseline processor should expose the fixture prompt size");
    let baseline_command = romeo_and_juliet_command(9_819, None);
    let prompt_token_count = baseline_processor
        .prepare_chat(&baseline_command)
        .expect("the fixture prompt should fit its artifact context")
        .prompt_token_ids()
        .len();
    let maximum_context_tokens = u32::try_from(prompt_token_count + 100)
        .expect("the synthetic fixture prompt should fit u32");
    let processor = LagunaGenerationProcessor::new_with_performance_attribution(
        SYNTHETIC_LAGUNA_MODEL_ID,
        text_descriptor,
        maximum_context_tokens,
        maximum_context_tokens - 1,
        false,
    )
    .expect("the context-bounded processor should construct");
    let mut overflowing_command = baseline_command;
    overflowing_command.settings.max_output_tokens = 101;

    let preparation_error = processor
        .prepare_chat(&overflowing_command)
        .expect_err("the requested output must not be truncated to make total context fit");

    assert!(matches!(
        preparation_error,
        LagunaPreparationError::ContextLengthExceeded {
            actual_context_tokens,
            maximum_context_tokens: rejected_maximum_context_tokens,
        } if actual_context_tokens == prompt_token_count + 101
            && rejected_maximum_context_tokens == maximum_context_tokens
    ));
}
