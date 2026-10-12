//! Ephemeral daemon IPC service for local one-shot CLI verbs.

use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{Arc, RwLock, atomic::AtomicU64},
};

use astronomical_config::{AstronomicalConfig, AstronomicalInstancePaths};
use astronomical_ipc_protocol::{
    DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION, DaemonIpcListener, DaemonRequest,
    DaemonResponse, DaemonTransportError, StreamingResponseWriter,
};
use tokio::sync::watch;

use crate::{
    GenerationStartError, ImageGenerationExecutor,
    config_reload::ResolvedRuntimeConfig,
    library::{DownloadCatalog, LibraryDownloadCoordinator},
    supervisor_performance_attribution::{
        SupervisorPerformanceAttributionLog, SupervisorPerformanceMeasurement,
        SupervisorPerformanceOperation,
    },
};

/// Built-in chat model used when the client sends no model and the user has
/// configured no default; single source of truth lives in the config crate.
use astronomical_config::BUILTIN_DEFAULT_MODEL_ID;

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
    /// Bundled release catalog of downloadable models.
    pub download_catalog: Arc<DownloadCatalog>,
    /// Library download coordinator, present when the daemon owns Library state.
    pub library_download_coordinator: Option<Arc<LibraryDownloadCoordinator>>,
    /// Instance paths used to read and persist user configuration.
    pub instance_paths: AstronomicalInstancePaths,
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
    let mut daemon_listener = DaemonIpcListener::bind(socket_path.clone()).await?;
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    // The spawned task must own the log: the service loop runs far longer
    // than this function's borrow of the application state.
    let service_attribution_log = supervisor_attribution_log.clone();
    let service_task = tokio::spawn(async move {
        let mut shutdown_receiver = shutdown_receiver;
        loop {
            let generation_context = generation_context.clone();
            let supervisor_attribution_log = service_attribution_log.clone();
            tokio::select! {
                _ = shutdown_receiver.changed() => break,
                serve_outcome = daemon_listener.serve_streaming_request(
                    move |daemon_request, streaming_response_writer| {
                        handle_streaming_daemon_request(
                            generation_context,
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
                default_model_id: Some(effective_default_model_id(
                    &generation_context.instance_paths,
                )),
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
            schema_json,
        } => {
            supervisor_attribution_log
                .measure_async_operation_best_effort(
                    SupervisorPerformanceOperation::DaemonIpcChatGenerate,
                    || {
                        super::daemon_ipc_chat::stream_chat_generation(
                            &generation_context,
                            super::daemon_ipc_chat::DaemonIpcChatRequest {
                                model,
                                messages,
                                settings,
                                schema_json,
                            },
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
        DaemonRequest::ModelsList => {
            crate::daemon_ipc_models::handle_models_list(
                &generation_context,
                &supervisor_attribution_log,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::Catalog => {
            crate::daemon_ipc_models::handle_catalog(
                &generation_context,
                &supervisor_attribution_log,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::DownloadStart { model_id } => {
            crate::daemon_ipc_models::handle_download_start(
                &generation_context,
                &supervisor_attribution_log,
                model_id,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::DownloadStatus => {
            crate::daemon_ipc_models::handle_download_status(
                &generation_context,
                &supervisor_attribution_log,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::DefaultModelSet { model_id } => {
            crate::daemon_ipc_models::handle_default_model_set(
                &generation_context,
                &supervisor_attribution_log,
                model_id,
                streaming_response_writer,
            )
            .await
        }
    }
}

/// The model ID CLI verbs fall back to: the user-configured default model,
/// or the built-in fallback when none is configured. The config file is read
/// fresh so a `models default` made on another CLI process is visible here.
pub(super) fn effective_default_model_id(instance_paths: &AstronomicalInstancePaths) -> String {
    AstronomicalConfig::load_from_instance_paths(instance_paths.clone())
        .ok()
        .and_then(|user_config| user_config.default_model().map(str::to_owned))
        .unwrap_or_else(|| BUILTIN_DEFAULT_MODEL_ID.to_owned())
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

/// Builds the shared unknown-model rejection text with near-match suggestions.
pub(crate) fn unknown_model_rejection_reason(
    requested_model_id: &str,
    suggested_model_ids: &[String],
    base_message: &str,
) -> String {
    if suggested_model_ids.is_empty() {
        format!("the model {requested_model_id} is unknown — {base_message}")
    } else {
        format!(
            "the model {requested_model_id} is unknown — {base_message}; did you mean: {}",
            suggested_model_ids.join(", ")
        )
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

pub(super) async fn send_terminal_streaming_response(
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
