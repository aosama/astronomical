//! Ephemeral daemon IPC service for local one-shot CLI verbs.

use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{Arc, RwLock, atomic::AtomicU64},
};

use astronomical_config::AstronomicalInstancePaths;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationFailureReason, ChatGenerationSettings, ChatMessage,
    ChatToolChoice, DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION, DaemonIpcListener,
    DaemonRequest, DaemonResponse, DaemonTransportError, RequestId, StreamingResponseWriter,
};
use tokio::sync::{mpsc, watch};

use crate::{
    ChatGenerationStreamErrorCode, ChatGenerationStreamEvent, GenerationStartError,
    ImageGenerationExecutor,
    application::allocate_chat_request_id,
    config_reload::ResolvedRuntimeConfig,
    load_configured_qwen_thinking_channel_seed,
    request_generation_defaults::{RequestGenerationSettingsPresence, apply_generation_defaults},
    supervisor_performance_attribution::{
        SupervisorPerformanceAttributionLog, SupervisorPerformanceMeasurement,
        SupervisorPerformanceOperation,
    },
    worker_health::{WorkerHealthSnapshot, WorkerHealthStatus},
};

/// One running daemon IPC service bound to the instance socket.
pub struct DaemonIpcService {
    service_task: tokio::task::JoinHandle<Result<(), DaemonTransportError>>,
    shutdown_sender: watch::Sender<bool>,
    socket_path: PathBuf,
}

impl DaemonIpcService {
    /// Socket file path the service is serving on.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Stops serving, waits for the serving task to end, and removes the
    /// socket file so clients cannot discover a dead endpoint.
    pub async fn shutdown(self) -> Result<(), DaemonTransportError> {
        let _ = self.shutdown_sender.send(true);
        self.service_task
            .await
            .map_err(DaemonTransportError::ServiceTaskFailed)??;
        remove_socket_file_ignoring_missing(&self.socket_path)?;
        Ok(())
    }
}

/// Everything the daemon IPC service needs to serve ephemeral CLI processes
/// without going through the REST surface.
#[derive(Clone)]
pub struct DaemonIpcGenerationContext {
    /// Executor backed by the single resident local worker. Image-generation
    /// executor is the widest supervisor executor trait: it carries chat,
    /// image, and embeddings start methods with worker health.
    pub executor: Arc<dyn ImageGenerationExecutor>,
    /// Live model policy defaults, present only when config reloading is wired.
    pub reloadable_config: Option<Arc<RwLock<ResolvedRuntimeConfig>>>,
    /// Shared chat request identifier counter, also used by the REST surface.
    pub next_chat_request_id: Arc<AtomicU64>,
}

/// Starts serving ephemeral local daemon requests on the instance socket.
///
/// The service lives beside the REST API but shares no transport with it: CLI
/// processes talk over the owner-only unix socket and never need the loopback
/// HTTP port. One request is served per connection; generation requests
/// stream response frames until a terminal frame closes the connection.
pub async fn start_daemon_ipc_service(
    instance_paths: &AstronomicalInstancePaths,
    generation_context: DaemonIpcGenerationContext,
    supervisor_attribution_log: &SupervisorPerformanceAttributionLog,
) -> Result<DaemonIpcService, DaemonTransportError> {
    let socket_path = instance_paths.ipc_socket_file_path();
    let seed_instance_paths = instance_paths.clone();
    let mut daemon_listener = DaemonIpcListener::bind(socket_path.clone()).await?;
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    // The spawned task must own the log: the service loop runs far longer
    // than this function's borrow of the application state.
    let service_attribution_log = supervisor_attribution_log.clone();
    let service_task = tokio::spawn(async move {
        let mut shutdown_receiver = shutdown_receiver;
        loop {
            let generation_context = generation_context.clone();
            let seed_instance_paths = seed_instance_paths.clone();
            let supervisor_attribution_log = service_attribution_log.clone();
            tokio::select! {
                _ = shutdown_receiver.changed() => break,
                serve_outcome = daemon_listener.serve_streaming_request(
                    move |daemon_request, streaming_response_writer| {
                        handle_streaming_daemon_request(
                            generation_context,
                            seed_instance_paths,
                            supervisor_attribution_log,
                            daemon_request,
                            streaming_response_writer,
                        )
                    }
                ) =>
                {
                    // One misbehaving local client must not take the endpoint down.
                    match serve_outcome {
                        Err(accept_error @ DaemonTransportError::AcceptFailed { .. }) => {
                            return Err(accept_error);
                        }
                        Err(request_error) => {
                            tracing::warn!(
                                error = %request_error,
                                "daemon IPC request failed; listener stays available"
                            );
                        }
                        Ok(()) => {}
                    }
                }
            }
        }
        Ok(())
    });
    Ok(DaemonIpcService {
        service_task,
        shutdown_sender,
        socket_path,
    })
}

async fn handle_streaming_daemon_request(
    generation_context: DaemonIpcGenerationContext,
    seed_instance_paths: AstronomicalInstancePaths,
    supervisor_attribution_log: SupervisorPerformanceAttributionLog,
    daemon_request: DaemonRequest,
    mut streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    match daemon_request {
        DaemonRequest::Handshake => {
            let handshake_response = DaemonResponse::HandshakeAccepted {
                protocol_version: DAEMON_PROTOCOL_VERSION,
                application_name: DAEMON_APPLICATION_NAME.to_owned(),
            };
            send_attributed_streaming_response(
                &supervisor_attribution_log,
                SupervisorPerformanceOperation::DaemonIpcHandshake,
                &mut streaming_response_writer,
                &handshake_response,
            )
            .await?;
            streaming_response_writer.close().await
        }
        DaemonRequest::Status => {
            let health_snapshot = generation_context.executor.worker_health_snapshot();
            let status_response = DaemonResponse::Status {
                worker_status: health_snapshot.status.into(),
                ready_model_id: health_snapshot.ready_model_id,
            };
            send_attributed_streaming_response(
                &supervisor_attribution_log,
                SupervisorPerformanceOperation::DaemonIpcStatus,
                &mut streaming_response_writer,
                &status_response,
            )
            .await?;
            streaming_response_writer.close().await
        }
        DaemonRequest::ChatGenerate {
            model,
            messages,
            settings,
        } => {
            supervisor_attribution_log
                .measure_async_operation_best_effort(
                    SupervisorPerformanceOperation::DaemonIpcChatGenerate,
                    || {
                        stream_chat_generation(
                            &generation_context,
                            &seed_instance_paths,
                            &supervisor_attribution_log,
                            model,
                            messages,
                            settings,
                            streaming_response_writer,
                        )
                    },
                    |stream_outcome| {
                        if stream_outcome.is_ok() {
                            SupervisorPerformanceMeasurement::success()
                        } else {
                            SupervisorPerformanceMeasurement::failure()
                        }
                    },
                )
                .await
        }
        DaemonRequest::EmbedGenerate {
            model,
            inputs,
            dimensions,
        } => {
            crate::daemon_ipc_embeddings::run_embeddings_generation(
                &generation_context,
                &supervisor_attribution_log,
                model,
                inputs,
                dimensions,
                streaming_response_writer,
            )
            .await
        }
    }
}

async fn send_attributed_streaming_response(
    supervisor_attribution_log: &SupervisorPerformanceAttributionLog,
    operation: SupervisorPerformanceOperation,
    streaming_response_writer: &mut StreamingResponseWriter,
    daemon_response: &DaemonResponse,
) -> Result<(), DaemonTransportError> {
    supervisor_attribution_log
        .measure_async_operation_best_effort(
            operation,
            || streaming_response_writer.send_response(daemon_response),
            |send_outcome| {
                if send_outcome.is_ok() {
                    SupervisorPerformanceMeasurement::success()
                } else {
                    SupervisorPerformanceMeasurement::failure()
                }
            },
        )
        .await
}

/// Runs one IPC chat generation: gates on worker readiness, fills defaults,
/// and streams events to the client until a terminal frame.
async fn stream_chat_generation(
    generation_context: &DaemonIpcGenerationContext,
    seed_instance_paths: &AstronomicalInstancePaths,
    supervisor_attribution_log: &SupervisorPerformanceAttributionLog,
    model: String,
    messages: Vec<ChatMessage>,
    settings: ChatGenerationSettings,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let health_snapshot = generation_context.executor.worker_health_snapshot();
    if let Some(rejection_reason) = ipc_generation_rejection_reason(&health_snapshot, &model) {
        let rejection_response = DaemonResponse::GenerationRejected {
            reason: rejection_reason,
        };
        return send_terminal_streaming_response(streaming_response_writer, &rejection_response)
            .await;
    }
    let Some(request_id_raw) = allocate_chat_request_id(&generation_context.next_chat_request_id)
    else {
        let rejection_response = DaemonResponse::GenerationRejected {
            reason: "the local request identifier space is exhausted".to_owned(),
        };
        return send_terminal_streaming_response(streaming_response_writer, &rejection_response)
            .await;
    };
    let mut settings = settings;
    apply_ipc_generation_defaults(generation_context, &health_snapshot, &model, &mut settings);
    let generation_command = ChatGenerationCommand {
        request_id: RequestId::new(request_id_raw),
        model: model.clone(),
        messages,
        tools: vec![],
        tool_choice: ChatToolChoice::Auto,
        settings,
        qwen_thinking_channel_seed: load_configured_qwen_thinking_channel_seed(
            generation_context.reloadable_config.as_ref(),
            Some(seed_instance_paths),
            supervisor_attribution_log,
            &model,
        )
        .await,
        structured_generation: None,
    };
    let stream_event_receiver = match generation_context
        .executor
        .start_chat_generation(generation_command)
        .await
    {
        Ok(stream_event_receiver) => stream_event_receiver,
        Err(start_error) => {
            let rejection_response = DaemonResponse::GenerationRejected {
                reason: generation_start_rejection_reason(start_error),
            };
            return send_terminal_streaming_response(
                streaming_response_writer,
                &rejection_response,
            )
            .await;
        }
    };
    relay_stream_events(stream_event_receiver, streaming_response_writer).await
}

/// Rejects a generation when the worker is not ready with the requested
/// model; returns `None` when the generation may proceed.
fn ipc_generation_rejection_reason(
    health_snapshot: &WorkerHealthSnapshot,
    requested_model_id: &str,
) -> Option<String> {
    if health_snapshot.status != WorkerHealthStatus::Ready {
        return Some("the daemon worker is not ready to serve chat generation".to_owned());
    }
    match &health_snapshot.ready_model_id {
        Some(ready_model_id) if ready_model_id == requested_model_id => None,
        Some(ready_model_id) => Some(format!(
            "the model {requested_model_id} is not loaded; the resident model is {ready_model_id}"
        )),
        None => Some(format!(
            "the model {requested_model_id} is not loaded and no model is resident"
        )),
    }
}

/// Fills generation settings the CLI does not send: model policy defaults,
/// then the worker's advertised output ceiling when nothing else filled the
/// `max_output_tokens = 0` sentinel.
fn apply_ipc_generation_defaults(
    generation_context: &DaemonIpcGenerationContext,
    health_snapshot: &WorkerHealthSnapshot,
    model_id: &str,
    generation_settings: &mut ChatGenerationSettings,
) {
    let settings_presence = RequestGenerationSettingsPresence {
        maximum_output_tokens: generation_settings.max_output_tokens != 0,
        temperature: generation_settings.temperature_thousandths.is_some(),
        top_p: generation_settings.top_p_thousandths.is_some(),
    };
    apply_generation_defaults(
        generation_context.reloadable_config.as_ref(),
        model_id,
        settings_presence,
        generation_settings,
    );
    if generation_settings.max_output_tokens == 0 {
        if let Some(ready_model_capabilities) = &health_snapshot.ready_model_capabilities
            && let Some(chat_capabilities) = &ready_model_capabilities.chat
        {
            // The worker advertises u32 token counts; the wire settings field is u16.
            generation_settings.max_output_tokens =
                u16::try_from(chat_capabilities.max_output_tokens).unwrap_or(u16::MAX);
        }
    }
}

pub(crate) fn generation_start_rejection_reason(start_error: GenerationStartError) -> String {
    match start_error {
        GenerationStartError::CapacityUnavailable => {
            "no generation capacity is available".to_owned()
        }
        GenerationStartError::ModelLoadFailed {
            model_load_failure_reason,
        } => format!("the model could not be loaded: {model_load_failure_reason}"),
        GenerationStartError::RequestTooLarge {
            actual_ipc_message_bytes,
            maximum_ipc_message_bytes,
        } => format!(
            "the request is {actual_ipc_message_bytes} bytes but the IPC limit is \
             {maximum_ipc_message_bytes} bytes"
        ),
        GenerationStartError::WorkerUnavailable => "the worker is unavailable".to_owned(),
    }
}

/// Sends stream frames to the client until a terminal frame, then closes the
/// connection. A stream that ends without a terminal frame is a failure.
async fn relay_stream_events(
    mut stream_event_receiver: mpsc::Receiver<ChatGenerationStreamEvent>,
    mut streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    while let Some(stream_event) = stream_event_receiver.recv().await {
        let Some(daemon_response) = stream_event_to_daemon_response(stream_event) else {
            // PrefillProgress is worker-internal progress with no CLI presentation.
            continue;
        };
        streaming_response_writer
            .send_response(&daemon_response)
            .await?;
        let is_terminal_frame = matches!(
            daemon_response,
            DaemonResponse::ChatGenerationCompleted { .. }
                | DaemonResponse::ChatGenerationFailed { .. }
                | DaemonResponse::GenerationRejected { .. }
        );
        if is_terminal_frame {
            streaming_response_writer.close().await?;
            return Ok(());
        }
    }
    let eof_failure_response = DaemonResponse::ChatGenerationFailed {
        reason: ChatGenerationFailureReason::FatalExecution {
            reason: "the worker stream ended before completing the generation".to_owned(),
        },
    };
    send_terminal_streaming_response(streaming_response_writer, &eof_failure_response).await
}

fn stream_event_to_daemon_response(
    stream_event: ChatGenerationStreamEvent,
) -> Option<DaemonResponse> {
    match stream_event {
        ChatGenerationStreamEvent::TextFragment(text) => {
            Some(DaemonResponse::ChatGenerationText { text })
        }
        ChatGenerationStreamEvent::ReasoningFragment(text) => {
            Some(DaemonResponse::ChatGenerationReasoning { text })
        }
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index,
            function_name,
            arguments_json,
        } => Some(DaemonResponse::ChatGenerationToolCall {
            tool_call_index,
            function_name,
            arguments_json,
        }),
        ChatGenerationStreamEvent::PrefillProgress { .. } => None,
        ChatGenerationStreamEvent::Completed {
            prompt_token_count,
            generated_token_count,
            reasoning_token_count,
            cached_token_count,
            reason,
        } => Some(DaemonResponse::ChatGenerationCompleted {
            prompt_token_count,
            generated_token_count,
            reasoning_token_count,
            cached_token_count,
            reason,
        }),
        ChatGenerationStreamEvent::Failed { reason } => {
            Some(DaemonResponse::ChatGenerationFailed { reason })
        }
        ChatGenerationStreamEvent::Error(stream_error_code) => {
            Some(DaemonResponse::ChatGenerationFailed {
                reason: ChatGenerationFailureReason::FatalExecution {
                    reason: stream_error_code_to_failure_reason(stream_error_code),
                },
            })
        }
    }
}

fn stream_error_code_to_failure_reason(stream_error_code: ChatGenerationStreamErrorCode) -> String {
    match stream_error_code {
        ChatGenerationStreamErrorCode::WorkerUnavailable => {
            "the worker became unavailable during the generation".to_owned()
        }
    }
}

async fn send_terminal_streaming_response(
    mut streaming_response_writer: StreamingResponseWriter,
    daemon_response: &DaemonResponse,
) -> Result<(), DaemonTransportError> {
    streaming_response_writer
        .send_response(daemon_response)
        .await?;
    streaming_response_writer.close().await
}

fn remove_socket_file_ignoring_missing(socket_path: &Path) -> Result<(), DaemonTransportError> {
    match std::fs::remove_file(socket_path) {
        Ok(()) => Ok(()),
        Err(remove_error) if remove_error.kind() == ErrorKind::NotFound => Ok(()),
        Err(remove_error) => Err(DaemonTransportError::SocketCleanupFailed {
            socket_path: socket_path.to_path_buf(),
            source: remove_error,
        }),
    }
}
