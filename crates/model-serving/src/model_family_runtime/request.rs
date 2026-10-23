use crate::Qwen3_5InferenceRequest;
use crate::inference_engine::contract::PreparedInferenceRequest;

/// Family-tagged prepared request accepted by the model-family engine boundary.
#[derive(Debug)]
pub enum ModelFamilyInferenceRequest {
    Qwen3_5(Qwen3_5InferenceRequest),
    /// The folded ternary family reuses the Qwen3.5 prepared request: the
    /// pack speaks the same chat dialect and consumes the same request data.
    PrismHadamardQwen35(Qwen3_5InferenceRequest),
}

impl PreparedInferenceRequest for ModelFamilyInferenceRequest {
    fn prompt_token_count(&self) -> usize {
        match self {
            Self::Qwen3_5(inference_request) => inference_request.prompt_token_count(),
            Self::PrismHadamardQwen35(inference_request) => inference_request.prompt_token_count(),
        }
    }

    fn clone_for_streaming_retry(&self) -> Option<Self> {
        match self {
            Self::Qwen3_5(inference_request) => Some(Self::Qwen3_5(inference_request.clone())),
            // This family has no streaming replacement engine, even though its
            // prepared request reuses the Qwen3.5 request representation.
            Self::PrismHadamardQwen35(_) => None,
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
            Self::PrismHadamardQwen35(inference_request) => {
                inference_request.record_streaming_retry_interval(started_at, ended_at);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use astronomical_ipc_protocol::RequestId;

    use super::{ModelFamilyInferenceRequest, PreparedInferenceRequest};
    use crate::Qwen3_5InferenceRequest;

    #[test]
    fn should_keep_prism_resident_only_without_changing_qwen_retry_support() {
        let request = Qwen3_5InferenceRequest::new(RequestId::new(0), Vec::new(), 1);
        assert!(
            ModelFamilyInferenceRequest::PrismHadamardQwen35(request.clone())
                .clone_for_streaming_retry()
                .is_none()
        );
        assert!(
            ModelFamilyInferenceRequest::Qwen3_5(request)
                .clone_for_streaming_retry()
                .is_some()
        );
    }
}
