use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationFailureReason, ChatGenerationOutput,
    ChatGenerationSettings, ChatMessage, ChatModelCapabilities, ChatToolChoice,
    MAX_IPC_FRAME_BYTES, MtpRuntimeState, ProtocolReader, ProtocolWriter, RequestId,
    SpeculativePrefillRuntimeState, WorkerCommand, WorkerEvent,
};
use astronomical_model_serving::{
    EngineBackedWorker, EngineGenerationStart, EngineLoadResult, GeneratedToken,
    GenerationFinalization, InferenceEngine, InferenceEngineError, ModelGeneratedTokenTranslation,
    ModelGenerationOutputError, ModelGenerationProcessor, PreparedInferenceRequest,
    PreparedModelGeneration,
};
use tokio::io::{duplex, split};
use tokio::time::timeout;

#[tokio::test]
async fn should_report_memory_admission_failure_without_stopping_the_worker() {
    let engine_worker = EngineBackedWorker::new(PassthroughProcessor, RejectingEngine);
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

    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Ready { .. }
    ));
    for request_number in [1_u64, 2_u64] {
        supervisor_writer
            .send_command(&WorkerCommand::Generate(chat_command(request_number)))
            .await
            .expect("the worker should receive the generation request");
        assert_eq!(
            next_event(&mut supervisor_reader).await,
            WorkerEvent::Failed {
                request_id: RequestId::new(request_number),
                reason: ChatGenerationFailureReason::invalid_request(
                    "generation context exceeds available GPU wired memory",
                ),
            }
        );
    }

    supervisor_writer
        .close()
        .await
        .expect("the supervisor should close the worker transport");
    assert!(
        timeout(Duration::from_secs(1), worker_task)
            .await
            .expect("the worker should stop after transport closure")
            .expect("the worker task should not panic")
            .is_ok()
    );
}

#[tokio::test]
async fn should_report_mid_generation_memory_growth_rejection_without_stopping_the_worker() {
    let engine_worker = EngineBackedWorker::new(PassthroughProcessor, GrowthRejectingEngine);
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

    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Ready { .. }
    ));
    for request_number in [1_u64, 2_u64] {
        supervisor_writer
            .send_command(&WorkerCommand::Generate(chat_command(request_number)))
            .await
            .expect("the worker should receive the generation request");
        assert_eq!(
            next_event(&mut supervisor_reader).await,
            WorkerEvent::Failed {
                request_id: RequestId::new(request_number),
                reason: ChatGenerationFailureReason::invalid_request(
                    "adaptive RAM growth would exceed the machine limit",
                ),
            }
        );
    }

    supervisor_writer
        .close()
        .await
        .expect("the supervisor should close the worker transport");
    assert!(
        timeout(Duration::from_secs(1), worker_task)
            .await
            .expect("the worker should stop after transport closure")
            .expect("the worker task should not panic")
            .is_ok()
    );
}

#[tokio::test]
async fn should_report_injected_token_memory_growth_rejection_without_stopping_the_worker() {
    let engine_worker = EngineBackedWorker::new(FeedbackProcessor, FeedbackGrowthRejectingEngine);
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

    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Ready { .. }
    ));
    for request_number in [1_u64, 2_u64] {
        supervisor_writer
            .send_command(&WorkerCommand::Generate(chat_command(request_number)))
            .await
            .expect("the worker should receive the generation request");
        let worker_event = next_event(&mut supervisor_reader).await;
        assert!(
            matches!(
                worker_event,
                WorkerEvent::GenerationProgress {
                    request_id,
                    generated_token_count: 1,
                    mlx_memory_snapshot: None,
                    ..
                } if request_id == RequestId::new(request_number)
            ),
            "expected internal generation progress, got {worker_event:?}"
        );
        assert_eq!(
            next_event(&mut supervisor_reader).await,
            WorkerEvent::Failed {
                request_id: RequestId::new(request_number),
                reason: ChatGenerationFailureReason::invalid_request(
                    "injected token growth would exceed the machine limit",
                ),
            }
        );
    }

    supervisor_writer
        .close()
        .await
        .expect("the supervisor should close the worker transport");
    assert!(
        timeout(Duration::from_secs(1), worker_task)
            .await
            .expect("the worker should stop after transport closure")
            .expect("the worker task should not panic")
            .is_ok()
    );
}

fn chat_command(request_number: u64) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_number),
        model: "example/rejecting-engine".to_owned(),
        messages: vec![ChatMessage::User {
            content: "Use more memory than this machine can safely provide.".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 1,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
    }
}

async fn next_event<ReadTransport>(
    supervisor_reader: &mut ProtocolReader<ReadTransport>,
) -> WorkerEvent
where
    ReadTransport: tokio::io::AsyncRead + Unpin,
{
    supervisor_reader
        .next_event()
        .await
        .expect("the worker should write a valid event")
        .expect("the worker transport should remain open")
}

struct PassthroughProcessor;

struct FeedbackProcessor;

struct TestInferenceRequest {
    prompt_token_count: usize,
}

impl TestInferenceRequest {
    const fn new(prompt_token_count: usize) -> Self {
        Self { prompt_token_count }
    }
}

impl PreparedInferenceRequest for TestInferenceRequest {
    fn prompt_token_count(&self) -> usize {
        self.prompt_token_count
    }
}

impl ModelGenerationProcessor for PassthroughProcessor {
    type InferenceRequest = TestInferenceRequest;
    type RequestOutput = ();

    fn ready_event(
        &self,
        _mtp_runtime_state: MtpRuntimeState,
        _mtp_unavailable_reason: Option<String>,
        _speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        _speculative_prefill_unavailable_reason: Option<String>,
        _speculative_prefill_draft_model_id: Option<String>,
        _speculative_prefill_draft_model_revision: Option<String>,
    ) -> WorkerEvent {
        WorkerEvent::Ready {
            model_id: "example/rejecting-engine".to_owned(),
            capabilities: ChatModelCapabilities {
                supports_reasoning: false,
                supports_tool_calls: false,
                has_vision: false,
                max_input_tokens: 241_664,
                max_output_tokens: 20_480,
                context_window: 262_144,
            },
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
        }
    }

    fn prepare_chat_generation(
        &self,
        _generation_command: &ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    > {
        Ok(PreparedModelGeneration::new(
            TestInferenceRequest::new(1),
            (),
        ))
    }

    fn is_end_of_sequence_token(&self, _generated_token_id: u32) -> bool {
        false
    }

    fn translate_generated_token(
        &self,
        _request_output: &mut Self::RequestOutput,
        _generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        Ok(ModelGeneratedTokenTranslation::from_outputs(Vec::new()))
    }

    fn finish_request_output(
        &self,
        _request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        Ok(Vec::new())
    }
}

impl ModelGenerationProcessor for FeedbackProcessor {
    type InferenceRequest = TestInferenceRequest;
    type RequestOutput = ();

    fn ready_event(
        &self,
        _mtp_runtime_state: MtpRuntimeState,
        _mtp_unavailable_reason: Option<String>,
        _speculative_prefill_runtime_state: SpeculativePrefillRuntimeState,
        _speculative_prefill_unavailable_reason: Option<String>,
        _speculative_prefill_draft_model_id: Option<String>,
        _speculative_prefill_draft_model_revision: Option<String>,
    ) -> WorkerEvent {
        WorkerEvent::Ready {
            model_id: "example/feedback-engine".to_owned(),
            capabilities: ChatModelCapabilities {
                supports_reasoning: false,
                supports_tool_calls: false,
                has_vision: false,
                max_input_tokens: 241_664,
                max_output_tokens: 20_480,
                context_window: 262_144,
            },
            mtp_runtime_state: MtpRuntimeState::Disabled,
            mtp_unavailable_reason: None,
            speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
            speculative_prefill_unavailable_reason: None,
            speculative_prefill_draft_model_id: None,
            speculative_prefill_draft_model_revision: None,
        }
    }

    fn prepare_chat_generation(
        &self,
        _generation_command: &ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    > {
        Ok(PreparedModelGeneration::new(
            TestInferenceRequest::new(1),
            (),
        ))
    }

    fn is_end_of_sequence_token(&self, _generated_token_id: u32) -> bool {
        false
    }

    fn translate_generated_token(
        &self,
        _request_output: &mut Self::RequestOutput,
        _generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        Ok(ModelGeneratedTokenTranslation::new(Vec::new(), vec![81]))
    }

    fn finish_request_output(
        &self,
        _request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        Ok(Vec::new())
    }
}

struct RejectingEngine;

struct GrowthRejectingEngine;

struct FeedbackGrowthRejectingEngine;

impl InferenceEngine for RejectingEngine {
    type Request = TestInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Ok(EngineLoadResult::new())
    }

    async fn start_generation(
        &mut self,
        _generation_request: TestInferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        Err(InferenceEngineError::InvalidRequest {
            reason: "generation context exceeds available GPU wired memory".to_owned(),
        })
    }

    async fn decode_next_token(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        unreachable!("a rejected request must not advance generation")
    }

    async fn inject_input_tokens(
        &mut self,
        _request_id: RequestId,
        _input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        unreachable!("a rejected request must not receive injected input tokens")
    }

    async fn cancel_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Ok(GenerationFinalization::default())
    }
}

impl InferenceEngine for GrowthRejectingEngine {
    type Request = TestInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Ok(EngineLoadResult::new())
    }

    async fn start_generation(
        &mut self,
        _generation_request: TestInferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        Ok(EngineGenerationStart::new(0))
    }

    async fn decode_next_token(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        Err(InferenceEngineError::InvalidRequest {
            reason: "adaptive RAM growth would exceed the machine limit".to_owned(),
        })
    }

    async fn inject_input_tokens(
        &mut self,
        _request_id: RequestId,
        _input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        unreachable!("a growth-rejected request must not receive injected input tokens")
    }

    async fn cancel_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Ok(GenerationFinalization::default())
    }
}

impl InferenceEngine for FeedbackGrowthRejectingEngine {
    type Request = TestInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Ok(EngineLoadResult::new())
    }

    async fn start_generation(
        &mut self,
        _generation_request: TestInferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        Ok(EngineGenerationStart::new(0))
    }

    async fn decode_next_token(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        Ok(GeneratedToken::TokenId {
            token_id: 1,
            is_reasoning_token: false,
            expert_memory_mode: None,
            mlx_memory_telemetry: None,
            generation_finalization: None,
        })
    }

    async fn inject_input_tokens(
        &mut self,
        _request_id: RequestId,
        _input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        Err(InferenceEngineError::InvalidRequest {
            reason: "injected token growth would exceed the machine limit".to_owned(),
        })
    }

    async fn cancel_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Ok(GenerationFinalization::default())
    }
}
