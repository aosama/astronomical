use runtime_integration::MlxRuntimeError;

use super::super::PrismHadamardModelError;
use crate::InferenceEngineError;
use text_generation::execution_error::TextExecutionError;

#[test]
fn should_keep_a_memory_ceiling_refusal_a_refused_request() {
    let runtime_error = MlxRuntimeError::ActiveMemoryLimitExceeded {
        active_memory_bytes: 1_073_741_824,
        attempted_allocation_bytes: 1_073_741_824,
        allowed_active_memory_bytes: 1_610_612_736,
    };
    let execution_error =
        TextExecutionError::from(PrismHadamardModelError::ForwardRuntime(runtime_error));
    assert!(matches!(
        InferenceEngineError::from(execution_error),
        InferenceEngineError::InvalidRequest { .. }
    ));
}

#[test]
fn should_name_the_decoder_layer_whose_state_kind_mismatched() {
    let execution_error =
        TextExecutionError::from(PrismHadamardModelError::DecoderStateLayerMismatch {
            layer_index: 3,
        });
    assert!(matches!(
        execution_error,
        TextExecutionError::InvalidRequestDecoderState { layer_index: 3, .. }
    ));
}

#[test]
fn should_carry_a_failure_without_a_qwen_counterpart_by_its_rendering() {
    let prism_model_error = PrismHadamardModelError::InvalidGenerationRequest {
        description: "the starting position exceeds the MLX int32 range".to_owned(),
    };
    let expected_description = prism_model_error.to_string();
    let execution_error = TextExecutionError::from(prism_model_error);
    let TextExecutionError::TextModelForward { description } = &execution_error else {
        panic!("a Prism-only failure must become a text model forward failure");
    };
    assert_eq!(description, &expected_description);
    assert!(matches!(
        InferenceEngineError::from(execution_error),
        InferenceEngineError::Fatal { .. }
    ));
}
