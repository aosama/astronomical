use std::sync::Mutex;

use super::*;

#[tokio::test]
async fn should_inject_model_visible_correction_tokens_and_continue_generation() {
    let injected_model_feedback_token_batches = Arc::new(Mutex::new(Vec::new()));
    let engine_worker = EngineBackedWorker::new(
        CorrectionRequestingProcessor,
        CorrectionAwareChatEngine::new(Arc::clone(&injected_model_feedback_token_batches)),
    );
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut supervisor_reader).await, ready_event());
    supervisor_writer
        .send_command(&WorkerCommand::Generate(chat_command(751, 13)))
        .await
        .expect("the worker should receive a chat request");
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Output {
            generated_token_count: 1,
            ..
        }
    ));
    assert_eq!(
        injected_model_feedback_token_batches
            .lock()
            .expect("the injected-token recorder should not be poisoned")
            .as_slice(),
        &[vec![81, 82, 83]],
    );
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Output {
            generated_token_count: 2,
            ..
        }
    ));
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Completed {
            reason: ChatGenerationCompletionReason::ToolCalls,
            ..
        }
    ));
    close_worker_transport(supervisor_writer, worker_task).await;
}

struct CorrectionAwareChatEngine {
    injected_model_feedback_token_batches: Arc<Mutex<Vec<Vec<u32>>>>,
    next_token_id: u32,
}

impl CorrectionAwareChatEngine {
    fn new(injected_model_feedback_token_batches: Arc<Mutex<Vec<Vec<u32>>>>) -> Self {
        Self {
            injected_model_feedback_token_batches,
            next_token_id: 1,
        }
    }
}

impl InferenceEngine for CorrectionAwareChatEngine {
    type Request = ScriptedInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Ok(EngineLoadResult::new())
    }

    async fn start_generation(
        &mut self,
        _generation_request: ScriptedInferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        self.next_token_id = 1;
        Ok(EngineGenerationStart::new(0))
    }

    async fn decode_next_token(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let generated_token = GeneratedToken::TokenId {
            token_id: self.next_token_id,
            is_reasoning_token: false,
            expert_memory_mode: None,
            mlx_memory_telemetry: None,
            first_decode_forward_elapsed_millis: None,
            generation_finalization: None,
        };
        self.next_token_id += 1;
        Ok(generated_token)
    }

    async fn inject_input_tokens(
        &mut self,
        _request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        self.injected_model_feedback_token_batches
            .lock()
            .expect("the injected-token recorder should not be poisoned")
            .push(input_token_ids);
        Ok(())
    }

    async fn cancel_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Ok(GenerationFinalization::default())
    }
}
