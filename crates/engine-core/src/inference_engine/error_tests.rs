//! The sampling failure crosses into the engine error with its reason intact,
//! so the text a client sees is the same as before sampling had its own type.

use astronomical_gpu_building_blocks::gpu_token_sampling::TokenSamplingError;

use super::InferenceEngineError;

#[test]
fn should_carry_the_sampling_reason_word_for_word_into_a_fatal_engine_error() {
    let sampling_error = TokenSamplingError {
        reason: "sampled logits must have shape [1, rows, vocabulary]".to_owned(),
    };

    let engine_error = InferenceEngineError::from(sampling_error);

    assert_eq!(
        engine_error.to_string(),
        "fatal engine error: sampled logits must have shape [1, rows, vocabulary]"
    );
    assert!(matches!(engine_error, InferenceEngineError::Fatal { .. }));
}

#[test]
fn should_keep_an_empty_sampling_reason_empty() {
    let engine_error = InferenceEngineError::from(TokenSamplingError {
        reason: String::new(),
    });

    match engine_error {
        InferenceEngineError::Fatal { reason } => assert_eq!(reason, ""),
        other_error => panic!("an empty sampling reason should stay fatal, got {other_error:?}"),
    }
}

#[test]
fn should_not_add_the_engine_prefix_before_the_engine_boundary() {
    let sampling_error = TokenSamplingError {
        reason: "top-k must be positive".to_owned(),
    };

    assert_eq!(sampling_error.to_string(), "top-k must be positive");
}
