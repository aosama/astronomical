//! Shared scaffolding for the daemon IPC hermetic suites: a stub executor
//! double, service startup, and the per-test instance state directory.

#![allow(dead_code)]

use std::{
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex, atomic::AtomicU64},
    time::Duration,
};

use astronomical_config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationSettings, ChatMessage,
    ChatModelCapabilities, DaemonRequest, EmbeddingsCommand, MtpRuntimeState,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, DaemonIpcGenerationContext,
    DaemonIpcService, EmbeddingsExecutionError, EmbeddingsOutput, GenerationStartError,
    ImageGenerationExecutor, SupervisorPerformanceAttributionLog, WorkerHealthSnapshot,
    start_daemon_ipc_service,
};
use tokio::{sync::mpsc, time::timeout};

pub(crate) const HANDSHAKE_TEST_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) fn fresh_instance_state_directory(test_name: &str) -> PathBuf {
    let nanos_since_epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should provide time after the epoch")
        .as_nanos();
    let state_directory = std::env::temp_dir().join(format!(
        "ast-dipc-{}-{}-{test_name}",
        std::process::id(),
        nanos_since_epoch % 1_000_000_000
    ));
    std::fs::create_dir_all(&state_directory)
        .expect("the instance state directory should be creatable");
    state_directory
}

pub(crate) fn disabled_supervisor_attribution_log(
    state_directory: &Path,
) -> SupervisorPerformanceAttributionLog {
    SupervisorPerformanceAttributionLog::open(state_directory, false)
        .expect("the disabled attribution log should open without writing")
}

pub(crate) fn unavailable_generation_context() -> DaemonIpcGenerationContext {
    DaemonIpcGenerationContext {
        executor: Arc::new(astronomical_supervisor::WorkerHandle::unavailable()),
        reloadable_config: None,
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
    }
}

/// Executor double standing in for the real worker: it reports a fixed health
/// snapshot and replays a fixed stream of events for every chat request.
pub(crate) struct StubGenerationExecutor {
    pub(crate) health_snapshot: WorkerHealthSnapshot,
    pub(crate) stream_events: Vec<ChatGenerationStreamEvent>,
    pub(crate) received_commands: Mutex<Vec<ChatGenerationCommand>>,
    /// Present when the stub should serve embeddings requests.
    pub(crate) embeddings_output: Option<Result<EmbeddingsOutput, EmbeddingsExecutionError>>,
    pub(crate) received_embeddings_commands: Mutex<Vec<EmbeddingsCommand>>,
}

impl ChatGenerationExecutor for StubGenerationExecutor {
    fn start_chat_generation(
        &self,
        generation_command: ChatGenerationCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<ChatGenerationStreamEvent>,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        self.received_commands
            .lock()
            .expect("the stub command log lock should not be poisoned")
            .push(generation_command.clone());
        let stream_events = self.stream_events.clone();
        Box::pin(async move {
            let (stream_event_sender, stream_event_receiver) = mpsc::channel(8);
            tokio::spawn(async move {
                for stream_event in stream_events {
                    let _ = stream_event_sender.send(stream_event).await;
                }
            });
            Ok(stream_event_receiver)
        })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        self.health_snapshot.clone()
    }
}

impl ImageGenerationExecutor for StubGenerationExecutor {
    fn start_embeddings_generation(
        &self,
        embeddings_command: EmbeddingsCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<Result<EmbeddingsOutput, EmbeddingsExecutionError>>,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        self.received_embeddings_commands
            .lock()
            .expect("the stub embeddings command log lock should not be poisoned")
            .push(embeddings_command.clone());
        let embeddings_output = self.embeddings_output.clone();
        Box::pin(async move {
            let Some(embeddings_output) = embeddings_output else {
                return Err(GenerationStartError::WorkerUnavailable);
            };
            let (embeddings_result_sender, embeddings_result_receiver) = mpsc::channel(1);
            tokio::spawn(async move {
                let _ = embeddings_result_sender.send(embeddings_output).await;
            });
            Ok(embeddings_result_receiver)
        })
    }
}

pub(crate) fn ready_stub_executor(ready_model_id: &str) -> Arc<StubGenerationExecutor> {
    let model_capabilities = ChatModelCapabilities {
        supports_reasoning: true,
        supports_tool_calls: false,
        has_vision: false,
        max_input_tokens: 1_024,
        max_output_tokens: 128,
        context_window: 2_048,
    };
    Arc::new(StubGenerationExecutor {
        health_snapshot: WorkerHealthSnapshot::ready_with_model(
            ready_model_id.to_owned(),
            model_capabilities,
            MtpRuntimeState::Disabled,
            None,
        ),
        stream_events: vec![
            ChatGenerationStreamEvent::TextFragment("Hello".to_owned()),
            ChatGenerationStreamEvent::TextFragment(" world".to_owned()),
            ChatGenerationStreamEvent::Completed {
                prompt_token_count: 5,
                generated_token_count: 2,
                reasoning_token_count: 0,
                cached_token_count: 0,
                reason: ChatGenerationCompletionReason::EndOfSequence,
            },
        ],
        received_commands: Mutex::new(Vec::new()),
        embeddings_output: None,
        received_embeddings_commands: Mutex::new(Vec::new()),
    })
}

pub(crate) fn ready_stub_executor_with_embeddings(
    ready_model_id: &str,
    embeddings_output: Result<EmbeddingsOutput, EmbeddingsExecutionError>,
) -> Arc<StubGenerationExecutor> {
    let mut stub_executor = ready_stub_executor(ready_model_id);
    Arc::get_mut(&mut stub_executor)
        .expect("the stub executor should be uniquely owned here")
        .embeddings_output = Some(embeddings_output);
    stub_executor
}

pub(crate) async fn start_stub_daemon_ipc_service(
    state_directory: &Path,
    stub_executor: Arc<StubGenerationExecutor>,
) -> DaemonIpcService {
    let instance_paths = AstronomicalInstancePaths::for_state_directory(
        state_directory.to_path_buf(),
        AstronomicalRuntimeInstance::Development,
    );
    let supervisor_attribution_log = disabled_supervisor_attribution_log(state_directory);
    let generation_context = DaemonIpcGenerationContext {
        executor: Arc::clone(&stub_executor) as Arc<dyn ImageGenerationExecutor>,
        reloadable_config: None,
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
    };
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        start_daemon_ipc_service(
            &instance_paths,
            generation_context,
            &supervisor_attribution_log,
        ),
    )
    .await
    .expect("the daemon IPC service start should finish inside the test timeout")
    .expect("the daemon IPC service should start on the instance socket")
}

pub(crate) fn text_only_chat_generate_request(model: &str) -> DaemonRequest {
    DaemonRequest::ChatGenerate {
        model: model.to_owned(),
        messages: vec![ChatMessage::User {
            content: "Say hello".to_owned(),
            images: vec![],
        }],
        settings: ChatGenerationSettings {
            max_output_tokens: 128,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
    }
}
