use astronomical_model_serving::{
    DeepSeekV4UnavailableGenerationProcessor, ModelFamilyGenerationProcessor,
    ModelFamilyInferenceRequest, ModelFamilyRequestOutput, PreparedInferenceRequest,
    deepseek_v4_unavailable_reason,
};

#[test]
fn should_keep_the_deepseek_runtime_boundary_as_an_unavailable_structural_stub() {
    let deepseek_processor =
        ModelFamilyGenerationProcessor::DeepSeekV4(DeepSeekV4UnavailableGenerationProcessor);
    assert_eq!(
        deepseek_v4_unavailable_reason(),
        "DeepSeek-V4 model execution is not implemented in this build"
    );
    assert!(matches!(
        deepseek_processor,
        ModelFamilyGenerationProcessor::DeepSeekV4(_)
    ));
}

#[test]
fn should_keep_family_request_variants_explicit() {
    let deepseek_request = ModelFamilyInferenceRequest::DeepSeekV4(
        astronomical_model_serving::DeepSeekV4UnavailableInferenceRequest,
    );
    assert_eq!(deepseek_request.prompt_token_count(), 0);
    let _deepseek_output = ModelFamilyRequestOutput::DeepSeekV4(
        astronomical_model_serving::DeepSeekV4UnavailableRequestOutput,
    );
}
