use crate::{
    DeepSeekV4UnavailableInferenceRequest, LagunaInferenceRequest, PreparedInferenceRequest,
    Qwen3_5InferenceRequest,
};

/// Family-tagged prepared request accepted by the model-family engine boundary.
#[derive(Debug)]
pub enum ModelFamilyInferenceRequest {
    Qwen3_5(Qwen3_5InferenceRequest),
    Laguna(LagunaInferenceRequest),
    DeepSeekV4(DeepSeekV4UnavailableInferenceRequest),
}

impl PreparedInferenceRequest for ModelFamilyInferenceRequest {
    fn prompt_token_count(&self) -> usize {
        match self {
            Self::Qwen3_5(inference_request) => inference_request.prompt_token_count(),
            Self::Laguna(inference_request) => inference_request.prompt_token_count(),
            Self::DeepSeekV4(inference_request) => inference_request.prompt_token_count(),
        }
    }
}
