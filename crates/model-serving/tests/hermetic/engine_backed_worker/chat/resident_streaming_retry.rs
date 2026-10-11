use super::*;
use tokio::sync::Notify;

#[derive(Clone, Copy)]
enum RetryScenario {
    ResidentStartFork,
    ResidentDecodeForkBeforeOutput,
    ResidentDecodeForkAfterOutput,
    StreamingFork,
}

struct RetryInferenceRequest {
    persistent_cache_identity: String,
    streaming_retry_drop_count: Arc<AtomicUsize>,
    is_streaming_retry: bool,
}

impl Clone for RetryInferenceRequest {
    fn clone(&self) -> Self {
        Self {
            persistent_cache_identity: self.persistent_cache_identity.clone(),
            streaming_retry_drop_count: Arc::clone(&self.streaming_retry_drop_count),
            is_streaming_retry: true,
        }
    }
}

impl Drop for RetryInferenceRequest {
    fn drop(&mut self) {
        if self.is_streaming_retry {
            self.streaming_retry_drop_count
                .fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl PreparedInferenceRequest for RetryInferenceRequest {
    fn prompt_token_count(&self) -> usize {
        1
    }

    fn clone_for_streaming_retry(&self) -> Option<Self> {
        Some(self.clone())
    }
}

struct RetryProcessor {
    streaming_retry_drop_count: Arc<AtomicUsize>,
}

impl ModelGenerationProcessor for RetryProcessor {
    type InferenceRequest = RetryInferenceRequest;
    type RequestOutput = ();

    fn ready_event(&self) -> WorkerEvent {
        ready_event()
    }

    fn prepare_chat_generation(
        &self,
        generation_command: &ChatGenerationCommand,
    ) -> Result<
        PreparedModelGeneration<Self::InferenceRequest, Self::RequestOutput>,
        ChatGenerationFailureReason,
    > {
        Ok(PreparedModelGeneration::new(
            RetryInferenceRequest {
                persistent_cache_identity: generation_command.model.clone(),
                streaming_retry_drop_count: Arc::clone(&self.streaming_retry_drop_count),
                is_streaming_retry: false,
            },
            (),
        ))
    }

    fn is_end_of_sequence_token(&self, generated_token_id: u32) -> bool {
        generated_token_id == 2
    }

    fn translate_generated_token(
        &self,
        _request_output: &mut Self::RequestOutput,
        generated_token_id: u32,
    ) -> Result<ModelGeneratedTokenTranslation, ModelGenerationOutputError> {
        Ok(ModelGeneratedTokenTranslation::from_outputs(vec![
            ChatGenerationOutput::Text {
                text: format!("token-{generated_token_id}"),
            },
        ]))
    }

    fn finish_request_output(
        &self,
        _request_output: &mut Self::RequestOutput,
    ) -> Result<Vec<ChatGenerationOutput>, ModelGenerationOutputError> {
        Ok(Vec::new())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetryEngineRole {
    Resident,
    Streaming,
}

struct RetryEngine {
    scenario: RetryScenario,
    role: RetryEngineRole,
    decode_count: usize,
    request_cache_identities: Arc<Mutex<Vec<(RetryEngineRole, String)>>>,
    continue_after_visible_output: Arc<Notify>,
}

impl InferenceEngine for RetryEngine {
    type Request = RetryInferenceRequest;

    async fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        Ok(EngineLoadResult::new())
    }

    async fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        self.request_cache_identities
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .push((
                self.role,
                inference_request.persistent_cache_identity.clone(),
            ));

        if matches!(
            (self.role, self.scenario),
            (RetryEngineRole::Resident, RetryScenario::ResidentStartFork)
                | (RetryEngineRole::Resident, RetryScenario::StreamingFork)
                | (RetryEngineRole::Streaming, RetryScenario::StreamingFork)
        ) {
            return Err(InferenceEngineError::ResidentForkRequired {
                reason: "complete resident experts do not fit the request memory budget".to_owned(),
            });
        }

        Ok(EngineGenerationStart::new(u32::from(
            self.role == RetryEngineRole::Streaming,
        )))
    }

    async fn decode_next_token(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        if self.role == RetryEngineRole::Resident
            && matches!(self.scenario, RetryScenario::ResidentDecodeForkAfterOutput)
            && self.decode_count == 1
        {
            self.continue_after_visible_output.notified().await;
        }
        if self.role == RetryEngineRole::Resident
            && matches!(
                (self.scenario, self.decode_count),
                (RetryScenario::ResidentDecodeForkBeforeOutput, 0)
                    | (RetryScenario::ResidentDecodeForkAfterOutput, 1)
            )
        {
            return Err(InferenceEngineError::ResidentForkRequired {
                reason: "resident prefill cannot continue within the memory budget".to_owned(),
            });
        }

        let generated_token_id = match self.role {
            RetryEngineRole::Resident => 1,
            RetryEngineRole::Streaming => 2,
        };
        self.decode_count += 1;
        Ok(GeneratedToken::TokenId {
            token_id: generated_token_id,
            is_reasoning_token: false,
            expert_memory_mode: None,
            mlx_memory_telemetry: None,
            first_decode_forward_elapsed_millis: None,
            generation_finalization: None,
        })
    }

    async fn inject_input_tokens(
        &mut self,
        _request_id: RequestId,
        _input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }

    async fn cancel_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        Ok(GenerationFinalization::default())
    }
}

struct RetryModelFactory {
    scenario: RetryScenario,
    streaming_retry_count: Arc<AtomicUsize>,
    request_cache_identities: Arc<Mutex<Vec<(RetryEngineRole, String)>>>,
    streaming_retry_drop_count: Arc<AtomicUsize>,
    continue_after_visible_output: Arc<Notify>,
}

impl ModelFactory<RetryProcessor, RetryEngine> for RetryModelFactory {
    async fn create(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFactoryRuntime<RetryProcessor, RetryEngine>, String> {
        Ok(ModelFactoryRuntime::autoregressive(
            RetryProcessor {
                streaming_retry_drop_count: Arc::clone(&self.streaming_retry_drop_count),
            },
            RetryEngine {
                scenario: self.scenario,
                role: RetryEngineRole::Resident,
                decode_count: 0,
                request_cache_identities: Arc::clone(&self.request_cache_identities),
                continue_after_visible_output: Arc::clone(&self.continue_after_visible_output),
            },
        ))
    }

    async fn create_streaming_retry(
        &self,
        _model_directory: &str,
        _model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFactoryRuntime<RetryProcessor, RetryEngine>, String> {
        self.streaming_retry_count.fetch_add(1, Ordering::SeqCst);
        Ok(ModelFactoryRuntime::autoregressive(
            RetryProcessor {
                streaming_retry_drop_count: Arc::clone(&self.streaming_retry_drop_count),
            },
            RetryEngine {
                scenario: self.scenario,
                role: RetryEngineRole::Streaming,
                decode_count: 0,
                request_cache_identities: Arc::clone(&self.request_cache_identities),
                continue_after_visible_output: Arc::clone(&self.continue_after_visible_output),
            },
        ))
    }

    fn supports_streaming_retry(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn should_retry_a_resident_start_fork_once_with_the_same_cache_identity() {
    let (mut supervisor_reader, mut supervisor_writer, worker_task, retry_count, identities, _, _) =
        launch_retry_worker(RetryScenario::ResidentStartFork).await;
    load_retry_model(&mut supervisor_reader, &mut supervisor_writer).await;
    submit_retry_generation(&mut supervisor_writer, 901).await;

    assert!(matches!(
        next_retry_event(&mut supervisor_reader).await,
        WorkerEvent::Output { request_id, .. } if request_id == RequestId::new(901)
    ));
    assert!(matches!(
        next_retry_event(&mut supervisor_reader).await,
        WorkerEvent::Completed {
            request_id,
            cached_token_count: 1,
            ..
        } if request_id == RequestId::new(901)
    ));
    assert_eq!(retry_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        identities
            .lock()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .as_slice(),
        &[
            (
                RetryEngineRole::Resident,
                "example/scripted-chat".to_owned()
            ),
            (
                RetryEngineRole::Streaming,
                "example/scripted-chat".to_owned()
            ),
        ]
    );

    close_worker_transport(supervisor_writer, worker_task).await;
}

#[tokio::test]
async fn should_retry_a_pre_output_resident_decode_fork_once() {
    let (mut supervisor_reader, mut supervisor_writer, worker_task, retry_count, _, _, _) =
        launch_retry_worker(RetryScenario::ResidentDecodeForkBeforeOutput).await;
    load_retry_model(&mut supervisor_reader, &mut supervisor_writer).await;
    submit_retry_generation(&mut supervisor_writer, 902).await;

    assert!(matches!(
        next_retry_event(&mut supervisor_reader).await,
        WorkerEvent::Output { request_id, .. } if request_id == RequestId::new(902)
    ));
    assert!(matches!(
        next_retry_event(&mut supervisor_reader).await,
        WorkerEvent::Completed { request_id, .. } if request_id == RequestId::new(902)
    ));
    assert_eq!(retry_count.load(Ordering::SeqCst), 1);

    close_worker_transport(supervisor_writer, worker_task).await;
}

#[tokio::test]
async fn should_not_retry_after_a_response_fragment_is_visible() {
    let (
        mut supervisor_reader,
        mut supervisor_writer,
        worker_task,
        retry_count,
        _,
        streaming_retry_drop_count,
        continue_after_visible_output,
    ) = launch_retry_worker(RetryScenario::ResidentDecodeForkAfterOutput).await;
    load_retry_model(&mut supervisor_reader, &mut supervisor_writer).await;
    submit_retry_generation(&mut supervisor_writer, 903).await;

    assert!(matches!(
        next_retry_event(&mut supervisor_reader).await,
        WorkerEvent::Output { request_id, .. } if request_id == RequestId::new(903)
    ));
    assert_eq!(streaming_retry_drop_count.load(Ordering::SeqCst), 1);
    continue_after_visible_output.notify_one();
    assert!(matches!(
        next_retry_event(&mut supervisor_reader).await,
        WorkerEvent::Failed { request_id, .. } if request_id == RequestId::new(903)
    ));
    assert_eq!(retry_count.load(Ordering::SeqCst), 0);
    close_worker_transport(supervisor_writer, worker_task).await;
}

#[tokio::test]
async fn should_not_retry_a_streaming_fork_more_than_once() {
    let (mut supervisor_reader, mut supervisor_writer, worker_task, retry_count, _, _, _) =
        launch_retry_worker(RetryScenario::StreamingFork).await;
    load_retry_model(&mut supervisor_reader, &mut supervisor_writer).await;
    submit_retry_generation(&mut supervisor_writer, 904).await;

    let failure_event = next_retry_event(&mut supervisor_reader).await;
    assert!(
        matches!(
            &failure_event,
            WorkerEvent::Failed { request_id, .. } if *request_id == RequestId::new(904)
        ),
        "expected the streaming rejection, got {failure_event:?}"
    );
    assert_eq!(retry_count.load(Ordering::SeqCst), 1);
    close_worker_transport(supervisor_writer, worker_task).await;
}

async fn launch_retry_worker(
    scenario: RetryScenario,
) -> (
    ProtocolReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>,
    ProtocolWriter<tokio::io::WriteHalf<tokio::io::DuplexStream>>,
    JoinHandle<Result<(), WorkerRuntimeError>>,
    Arc<AtomicUsize>,
    Arc<Mutex<Vec<(RetryEngineRole, String)>>>,
    Arc<AtomicUsize>,
    Arc<Notify>,
) {
    let streaming_retry_count = Arc::new(AtomicUsize::new(0));
    let request_cache_identities = Arc::new(Mutex::new(Vec::new()));
    let streaming_retry_drop_count = Arc::new(AtomicUsize::new(0));
    let continue_after_visible_output = Arc::new(Notify::new());
    let engine_worker = EngineBackedWorker::idle_with_model_factory(
        RetryModelFactory {
            scenario,
            streaming_retry_count: Arc::clone(&streaming_retry_count),
            request_cache_identities: Arc::clone(&request_cache_identities),
            streaming_retry_drop_count: Arc::clone(&streaming_retry_drop_count),
            continue_after_visible_output: Arc::clone(&continue_after_visible_output),
        },
        0,
    );
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });
    (
        ProtocolReader::new(supervisor_reader_transport),
        ProtocolWriter::new(supervisor_writer_transport),
        worker_task,
        streaming_retry_count,
        request_cache_identities,
        streaming_retry_drop_count,
        continue_after_visible_output,
    )
}

async fn load_retry_model<ReadTransport, WriteTransport>(
    supervisor_reader: &mut ProtocolReader<ReadTransport>,
    supervisor_writer: &mut ProtocolWriter<WriteTransport>,
) where
    ReadTransport: tokio::io::AsyncRead + Unpin,
    WriteTransport: AsyncWrite + Unpin,
{
    assert!(matches!(
        next_retry_event(supervisor_reader).await,
        WorkerEvent::Idle { .. }
    ));
    supervisor_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/requested-model".to_owned(),
            model_configuration: worker_model_configuration("requested-model"),
        })
        .await
        .expect("the worker should load the requested model");
    assert!(matches!(
        next_retry_event(supervisor_reader).await,
        WorkerEvent::ModelSwapped { .. }
    ));
}

async fn submit_retry_generation<WriteTransport>(
    supervisor_writer: &mut ProtocolWriter<WriteTransport>,
    request_id: u64,
) where
    WriteTransport: AsyncWrite + Unpin,
{
    supervisor_writer
        .send_command(&WorkerCommand::Generate(chat_command(request_id, 905)))
        .await
        .expect("the worker should receive the generation request");
}

async fn next_retry_event<ReadTransport>(
    supervisor_reader: &mut ProtocolReader<ReadTransport>,
) -> WorkerEvent
where
    ReadTransport: tokio::io::AsyncRead + Unpin,
{
    timeout(Duration::from_secs(5), next_event(supervisor_reader))
        .await
        .expect("the retry journey should emit an event within five seconds")
}
