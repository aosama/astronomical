use crate::{K2HorizonMoVAInferenceRequest, PreparedInferenceRequest, Qwen3_5InferenceRequest};

/// Family-tagged prepared request accepted by the model-family engine boundary.
#[derive(Debug)]
pub enum ModelFamilyInferenceRequest {
    Qwen3_5(Qwen3_5InferenceRequest),
    K2HorizonMoVA(K2HorizonMoVAInferenceRequest),
}

impl PreparedInferenceRequest for ModelFamilyInferenceRequest {
    fn prompt_token_count(&self) -> usize {
        match self {
            Self::Qwen3_5(inference_request) => inference_request.prompt_token_count(),
            Self::K2HorizonMoVA(inference_request) => inference_request.prompt_token_count(),
        }
    }

    fn clone_for_streaming_retry(&self) -> Option<Self> {
        match self {
            Self::Qwen3_5(inference_request) => Some(Self::Qwen3_5(inference_request.clone())),
            Self::K2HorizonMoVA(_) => None,
        }
    }

    fn record_streaming_retry_interval(
        &mut self,
        started_at: std::time::Instant,
        ended_at: std::time::Instant,
    ) {
        match self {
            Self::Qwen3_5(inference_request) => {
                inference_request.record_streaming_retry_interval(started_at, ended_at);
            }
            Self::K2HorizonMoVA(_) => {}
        }
    }
}
