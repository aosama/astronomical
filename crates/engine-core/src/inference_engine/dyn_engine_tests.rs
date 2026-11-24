use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ipc_protocol::RequestId;

use super::DynInferenceEngine;
use crate::inference_engine::contract::{
    EngineGenerationStart, GeneratedToken, GenerationFinalization, InferenceEngine,
    PreparedInferenceRequest,
};
use crate::inference_engine::{EngineLoadResult, InferenceEngineError};

struct CountedRequest;

impl PreparedInferenceRequest for CountedRequest {
    fn prompt_token_count(&self) -> usize {
        3
    }
}

/// An engine that answers every call with a refusal and counts the decodes.
struct RefusingEngine {
    decode_call_count: Arc<AtomicUsize>,
}

impl InferenceEngine for RefusingEngine {
    type Request = CountedRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Err(InferenceEngineError::EngineBusy)
    }

    async fn start_generation(
        &mut self,
        _refused_counted_request: CountedRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        Err(InferenceEngineError::EngineBusy)
    }

    async fn decode_next_token(
        &mut self,
        _counted_decode_request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        self.decode_call_count.fetch_add(1, Ordering::SeqCst);
        Err(InferenceEngineError::Fatal {
            reason: "refused decode".to_owned(),
        })
    }

    async fn inject_input_tokens(
        &mut self,
        _ignored_feedback_request_id: RequestId,
        _ignored_feedback_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }

    async fn cancel_generation(
        &mut self,
        _refused_cancellation_request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Err(InferenceEngineError::EngineBusy)
    }
}

fn boxed_refusing_engine(
    decode_call_count: &Arc<AtomicUsize>,
) -> Box<dyn DynInferenceEngine<CountedRequest> + Sync> {
    Box::new(RefusingEngine {
        decode_call_count: Arc::clone(decode_call_count),
    })
}

#[tokio::test]
async fn should_forward_each_decode_to_the_boxed_engine() {
    let decode_call_count = Arc::new(AtomicUsize::new(0));
    let mut boxed_engine = boxed_refusing_engine(&decode_call_count);

    let first_decode =
        InferenceEngine::decode_next_token(&mut boxed_engine, RequestId::new(1)).await;
    let second_decode =
        InferenceEngine::decode_next_token(&mut boxed_engine, RequestId::new(1)).await;

    assert!(matches!(
        first_decode,
        Err(InferenceEngineError::Fatal { .. })
    ));
    assert!(matches!(
        second_decode,
        Err(InferenceEngineError::Fatal { .. })
    ));
    assert_eq!(decode_call_count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn should_pass_the_engine_refusal_through_unchanged() {
    let mut boxed_engine = boxed_refusing_engine(&Arc::new(AtomicUsize::new(0)));

    let start_outcome = InferenceEngine::start_generation(&mut boxed_engine, CountedRequest).await;

    assert!(matches!(
        start_outcome,
        Err(InferenceEngineError::EngineBusy)
    ));
}

#[tokio::test]
async fn should_keep_the_engine_default_answers_behind_the_box() {
    let mut boxed_engine = boxed_refusing_engine(&Arc::new(AtomicUsize::new(0)));

    let preparation_status =
        InferenceEngine::advance_neural_engine_preparation(&mut boxed_engine).await;
    let astro_ssd_prompt_cache_stats =
        InferenceEngine::collect_astro_ssd_prompt_cache_stats(&boxed_engine).await;

    assert!(matches!(preparation_status, Ok(None)));
    assert!(matches!(astro_ssd_prompt_cache_stats, Ok(None)));
}
