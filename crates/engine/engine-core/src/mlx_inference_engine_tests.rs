use ipc_protocol::{NeuralEnginePrefillStatus, RequestId};

use super::MlxInferenceEngine;
use crate::inference_engine::contract::PreparedInferenceRequest;
use crate::inference_engine::{
    EngineGenerationStart, EngineLoadResult, GeneratedToken, GenerationFinalization,
    InferenceEngine, InferenceEngineError, MlxInferenceExecution,
};
use crate::memory::telemetry::MlxMemoryLimitAdjustment;

struct UnusedRequest;

impl PreparedInferenceRequest for UnusedRequest {
    fn prompt_token_count(&self) -> usize {
        0
    }
}

/// An execution that always reports Neural Engine preparation as off, and
/// says through `PREPARES` whether the engine should ask it at all. Every
/// other call is outside these tests and refuses.
struct PreparationReportingExecution<const PREPARES: bool>;

impl<const PREPARES: bool> MlxInferenceExecution for PreparationReportingExecution<PREPARES> {
    type Request = UnusedRequest;

    const PREPARES_NEURAL_ENGINE_PROGRAMS: bool = PREPARES;

    fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Err(InferenceEngineError::EngineBusy)
    }

    fn start_generation(
        &mut self,
        _unused_request: UnusedRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        Err(InferenceEngineError::EngineBusy)
    }

    fn decode_next_token(
        &mut self,
        _unused_decode_request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        Err(InferenceEngineError::EngineBusy)
    }

    fn inject_input_tokens(
        &mut self,
        _unused_feedback_request_id: RequestId,
        _unused_feedback_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        Err(InferenceEngineError::EngineBusy)
    }

    fn cancel_generation(
        &mut self,
        _unused_cancellation_request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Err(InferenceEngineError::EngineBusy)
    }

    fn advance_neural_engine_preparation(&mut self) -> Option<NeuralEnginePrefillStatus> {
        Some(NeuralEnginePrefillStatus::Off)
    }

    fn update_mlx_memory_limit(
        &mut self,
        _unused_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        Err(InferenceEngineError::EngineBusy)
    }
}

fn preparation_reporting_execution<const PREPARES: bool>() -> PreparationReportingExecution<PREPARES>
{
    PreparationReportingExecution
}

/// The owner thread panics while building its execution, so the engine is
/// left with an owner that is gone before the first command.
fn execution_whose_owner_thread_dies<const PREPARES: bool>()
-> PreparationReportingExecution<PREPARES> {
    panic!("the owner thread stops before it builds its execution")
}

#[tokio::test]
async fn should_pass_on_the_status_of_an_execution_that_prepares_programs() {
    let mut preparing_engine =
        MlxInferenceEngine::new(preparation_reporting_execution::<true>).expect("engine starts");

    let preparation_status = preparing_engine
        .advance_neural_engine_preparation()
        .await
        .expect("a running owner thread answers");

    assert_eq!(preparation_status, Some(NeuralEnginePrefillStatus::Off));
}

#[tokio::test]
async fn should_answer_no_preparation_without_asking_an_execution_that_prepares_none() {
    let mut non_preparing_engine =
        MlxInferenceEngine::new(preparation_reporting_execution::<false>).expect("engine starts");

    let preparation_status = non_preparing_engine
        .advance_neural_engine_preparation()
        .await
        .expect("the engine answers on its own side");

    assert_eq!(preparation_status, None);
}

#[tokio::test]
async fn should_not_report_a_gone_owner_thread_for_an_execution_that_prepares_none() {
    let mut non_preparing_engine =
        MlxInferenceEngine::new(execution_whose_owner_thread_dies::<false>).expect("engine starts");

    let preparation_status = non_preparing_engine
        .advance_neural_engine_preparation()
        .await
        .expect("no owner thread is asked, so its loss is not a failure here");

    assert_eq!(preparation_status, None);
}

#[tokio::test]
async fn should_report_a_gone_owner_thread_for_an_execution_that_prepares_programs() {
    let mut preparing_engine =
        MlxInferenceEngine::new(execution_whose_owner_thread_dies::<true>).expect("engine starts");

    let preparation_refusal = preparing_engine
        .advance_neural_engine_preparation()
        .await
        .expect_err("the owner thread that would prepare programs is gone");

    assert!(
        matches!(preparation_refusal, InferenceEngineError::Fatal { .. }),
        "{preparation_refusal:?}"
    );
}
