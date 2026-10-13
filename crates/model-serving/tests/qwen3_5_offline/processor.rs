use astronomical_ipc_protocol::ChatGenerationFailureReason;
use astronomical_model_serving::{
    MalformedModelOutputDiagnostic, ModelGenerationOutputError, Qwen3_5ImageProcessingError,
    Qwen3_5OutputParserError, Qwen3_5RequestOutputError, Qwen3_5TokenizerError,
    qwen3_5_request_enables_thinking, translate_qwen3_5_preparation_error,
    translate_request_output_error,
};

#[test]
fn should_disable_thinking_for_a_zero_budget_without_removing_model_capability() {
    assert!(!qwen3_5_request_enables_thinking(true, Some(0)));
    assert!(qwen3_5_request_enables_thinking(true, Some(512)));
    assert!(qwen3_5_request_enables_thinking(true, None));
    assert!(!qwen3_5_request_enables_thinking(false, None));
    assert!(!qwen3_5_request_enables_thinking(false, Some(512)));
}

#[test]
fn should_map_a_tokenizer_error_as_fatal_not_malformed() {
    let tokenizer_error = Qwen3_5TokenizerError::GeneratedTokenOutOfVocabulary {
        generated_token_id: 999_999,
        model_vocabulary_size: 248_320,
    };
    let request_output_error = Qwen3_5RequestOutputError::Tokenizer(tokenizer_error);

    let output_error = translate_request_output_error(request_output_error);

    assert!(
        matches!(output_error, ModelGenerationOutputError::Fatal { .. }),
        "a tokenizer error during generation must be Fatal so the worker terminates \
         rather than reporting a reusable malformed-output failure"
    );
}

#[test]
fn should_map_model_context_overflow_to_the_typed_failure() {
    let failure_reason =
        translate_qwen3_5_preparation_error(Qwen3_5TokenizerError::TotalContextTooLarge {
            actual_total_context_tokens: 262_145,
            maximum_total_context_tokens: 262_144,
        });

    assert_eq!(
        failure_reason,
        ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens: 262_145,
            maximum_context_tokens: 262_144,
        }
    );
}

#[test]
fn should_include_the_image_processing_reason_in_an_invalid_request_failure() {
    let image_processing_failure = Qwen3_5ImageProcessingError::AspectRatioTooLarge {
        height_pixels: 1,
        width_pixels: 201,
        aspect_ratio: 201.0,
        maximum_aspect_ratio: 200.0,
    };

    let failure_reason = translate_qwen3_5_preparation_error(
        Qwen3_5TokenizerError::ImageProcessing(image_processing_failure),
    );
    let expected_image_processing_failure_reason = concat!(
        "failed to process chat image input through the vision pipeline: ",
        "image aspect ratio 201.00 exceeds maximum 200.00 for 1x201 image"
    );

    assert_eq!(
        failure_reason,
        ChatGenerationFailureReason::InvalidRequest {
            reason: expected_image_processing_failure_reason.to_owned(),
        }
    );
}

#[test]
fn should_map_a_parser_error_as_malformed_output() {
    let parser_error = Qwen3_5OutputParserError::IncompleteControlMarker;
    let malformed_output_diagnostic = MalformedModelOutputDiagnostic {
        diagnostic_code: "incomplete_control_marker",
        parser_error: "Qwen3.5 output ended in a partial control marker".to_owned(),
        generated_token_ids: vec![101, 202],
        pending_token_ids: vec![202],
        decoded_output_text: "<".to_owned(),
        parser_state: "text",
        parser_pending_output_text: "<".to_owned(),
    };
    let request_output_error = Qwen3_5RequestOutputError::Parser {
        source: parser_error,
        diagnostic: Box::new(malformed_output_diagnostic.clone()),
    };

    let output_error = translate_request_output_error(request_output_error);

    assert_eq!(
        output_error,
        ModelGenerationOutputError::MalformedOutput {
            diagnostic: Box::new(malformed_output_diagnostic),
        }
    );
}
