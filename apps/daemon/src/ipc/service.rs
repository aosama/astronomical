//! Ephemeral daemon IPC service for local one-shot CLI verbs.

use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{Arc, RwLock, atomic::AtomicU64},
    time::Duration,
};

use astronomical_config::{AstronomicalConfig, AstronomicalInstancePaths};
use astronomical_ipc_protocol::{
    AcceptedDaemonConnection, DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION, DaemonIpcListener,
    DaemonRequest, DaemonResponse, DaemonTransportError, StreamingResponseWriter,
};
use tokio::{sync::watch, task::JoinSet};

use crate::{
    GenerationStartError, ImageGenerationExecutor,
    config::config_reload::ResolvedRuntimeConfig,
    library::{DownloadCatalog, LibraryDownloadCoordinator},
    performance::attribution::{
        DaemonPerformanceAttributionLog, DaemonPerformanceMeasurement, DaemonPerformanceOperation,
    },
};

/// Built-in chat model used when the client sends no model and the user has
/// configured no default; single source of truth lives in the config crate.
use astronomical_config::BUILTIN_DEFAULT_MODEL_ID;

/// Long enough to ride out a brief shortage, short enough that clients barely notice.
const ACCEPT_RETRY_DELAY: Duration = Duration::from_millis(100);
/// CLI verbs send their request straight after connecting, so this only ever ends a
/// connection that was never going to ask anything.
const FIRST_REQUEST_WAIT: Duration = Duration::from_secs(30);

/// One running daemon IPC service bound to the instance socket.
pub struct DaemonIpcService {
    service_task: tokio::task::JoinHandle<()>,
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
        // Awaiting the join handle fails only when the service task panicked or was aborted.
        if let Err(join_error) = self.service_task.await {
            return Err(DaemonTransportError::ServiceTaskFailed(join_error));
        }
        remove_socket_file_ignoring_missing(&self.socket_path)?;
        Ok(())
    }
}

/// Everything the daemon IPC service needs to serve ephemeral CLI processes
/// without going through the REST surface.
#[derive(Clone)]
pub struct DaemonIpcGenerationContext {
    /// Executor backed by the single resident local worker. Image-generation
    /// executor is the widest daemon executor trait: it carries chat,
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
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
) -> Result<DaemonIpcService, DaemonTransportError> {
    let socket_path = instance_paths.ipc_socket_file_path();
    let daemon_listener = DaemonIpcListener::bind(socket_path.clone()).await?;
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    // The spawned task must own the log: the service loop runs far longer
    // than this function's borrow of the application state.
    let service_attribution_log = daemon_attribution_log.clone();
    let service_task = tokio::spawn(serve_connections_until_shutdown(
        daemon_listener,
        shutdown_receiver,
        generation_context,
        service_attribution_log,
    ));
    Ok(DaemonIpcService {
        service_task,
        shutdown_sender,
        socket_path,
    })
}

/// Serves every connection on its own task. A generation holds its connection for
/// as long as it streams, and that includes the minutes a first model load can take,
/// so serving them in turn would leave `status`, and the load report a waiting
/// `respond` polls for, unanswered until the generation ended. The executor behind
/// these handlers is the one the REST surface already calls concurrently.
async fn serve_connections_until_shutdown(
    mut daemon_listener: DaemonIpcListener,
    mut shutdown_receiver: watch::Receiver<bool>,
    generation_context: DaemonIpcGenerationContext,
    daemon_attribution_log: DaemonPerformanceAttributionLog,
) {
    // Dropping the set at shutdown aborts connections still being served, which is
    // what shutting down always did to the one connection in flight.
    let mut connection_tasks = JoinSet::new();
    loop {
        tokio::select! {
            _ = shutdown_receiver.changed() => return,
            accepted_connection = accept_next_connection(&mut daemon_listener) => {
                connection_tasks.spawn(serve_connection(
                    accepted_connection,
                    generation_context.clone(),
                    daemon_attribution_log.clone(),
                ));
            }
            // Reaps finished connections so the set does not grow with every request.
            Some(finished_connection) = connection_tasks.join_next() => {
                log_connection_panic(finished_connection);
            }
        }
    }
}

/// Keeps trying through accept failures. Leaving the loop would abort every live
/// generation in the set and strand new clients on a socket nobody accepts from, and
/// the failures that happen in practice pass: out of file descriptors, or a client
/// that hung up during the handshake. The delay stops a lasting failure from spinning.
async fn accept_next_connection(
    daemon_listener: &mut DaemonIpcListener,
) -> AcceptedDaemonConnection {
    loop {
        match daemon_listener.accept_connection().await {
            Ok(accepted_connection) => return accepted_connection,
            Err(accept_error) => {
                tracing::warn!(
                    error = %accept_error,
                    "daemon IPC could not accept a connection; retrying"
                );
                tokio::time::sleep(ACCEPT_RETRY_DELAY).await;
            }
        }
    }
}

/// A panic stays inside its own connection, so the client sees only a closed socket;
/// this line is the one record of why.
fn log_connection_panic(finished_connection: Result<(), tokio::task::JoinError>) {
    if let Err(join_error) = finished_connection
        && join_error.is_panic()
    {
        tracing::error!(error = %join_error, "daemon IPC connection panicked");
    }
}

/// One misbehaving local client must not take the endpoint down, so a failed
/// connection is logged and goes no further.
async fn serve_connection(
    accepted_connection: AcceptedDaemonConnection,
    generation_context: DaemonIpcGenerationContext,
    daemon_attribution_log: DaemonPerformanceAttributionLog,
) {
    let serve_outcome = match read_first_request(accepted_connection).await {
        Ok(Some((daemon_request, streaming_response_writer))) => {
            handle_streaming_daemon_request(
                generation_context,
                daemon_attribution_log,
                daemon_request,
                streaming_response_writer,
            )
            .await
        }
        Ok(None) => Ok(()),
        Err(request_read_error) => Err(request_read_error),
    };
    if let Err(request_error) = serve_outcome {
        tracing::warn!(
            error = %request_error,
            "daemon IPC request failed; listener stays available"
        );
    }
}

/// A client that connects and never asks anything is treated as having hung up, so
/// it cannot hold a file descriptor and a task until the daemon exits.
async fn read_first_request(
    accepted_connection: AcceptedDaemonConnection,
) -> Result<Option<(DaemonRequest, StreamingResponseWriter)>, DaemonTransportError> {
    match tokio::time::timeout(FIRST_REQUEST_WAIT, accepted_connection.read_request()).await {
        Ok(request_read) => request_read,
        Err(_elapsed) => Ok(None),
    }
}

async fn handle_streaming_daemon_request(
    generation_context: DaemonIpcGenerationContext,
    daemon_attribution_log: DaemonPerformanceAttributionLog,
    daemon_request: DaemonRequest,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    match daemon_request {
        DaemonRequest::Handshake => {
            answer_handshake(&daemon_attribution_log, streaming_response_writer).await
        }
        DaemonRequest::Status => {
            answer_status(
                &generation_context,
                &daemon_attribution_log,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::ChatGenerate {
            model,
            messages,
            settings,
            schema_json,
        } => {
            stream_chat_generate(
                &generation_context,
                &daemon_attribution_log,
                super::chat::DaemonIpcChatRequest {
                    model,
                    messages,
                    settings,
                    schema_json,
                },
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::EmbedGenerate {
            model,
            inputs,
            dimensions,
        } => {
            crate::ipc::embeddings::run_embeddings_generation(
                &generation_context,
                &daemon_attribution_log,
                model,
                inputs,
                dimensions,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::ImageGenerate(image_request) => {
            crate::ipc::images::run_image_generation(
                &generation_context,
                image_request,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::ModelsList => {
            crate::ipc::models::handle_models_list(
                &generation_context,
                &daemon_attribution_log,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::Catalog => {
            crate::ipc::models::handle_catalog(
                &generation_context,
                &daemon_attribution_log,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::DownloadStart { model_id } => {
            crate::ipc::models::handle_download_start(
                &generation_context,
                &daemon_attribution_log,
                model_id,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::DownloadStatus => {
            crate::ipc::models::handle_download_status(
                &generation_context,
                &daemon_attribution_log,
                streaming_response_writer,
            )
            .await
        }
        DaemonRequest::DefaultModelSet { model_id } => {
            crate::ipc::models::handle_default_model_set(
                &generation_context,
                &daemon_attribution_log,
                model_id,
                streaming_response_writer,
            )
            .await
        }
    }
}

/// Answers the IPC handshake with the daemon's protocol version.
async fn answer_handshake(
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    mut streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let handshake_response = DaemonResponse::HandshakeAccepted {
        protocol_version: DAEMON_PROTOCOL_VERSION,
        application_name: DAEMON_APPLICATION_NAME.to_owned(),
    };
    send_attributed_streaming_response(
        daemon_attribution_log,
        DaemonPerformanceOperation::DaemonIpcHandshake,
        &mut streaming_response_writer,
        &handshake_response,
    )
    .await?;
    streaming_response_writer.close().await
}

/// Answers the status request from the live worker health snapshot.
async fn answer_status(
    generation_context: &DaemonIpcGenerationContext,
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    mut streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let health_snapshot = generation_context.executor.worker_health_snapshot();
    let status_response = DaemonResponse::Status {
        worker_status: health_snapshot.status.into(),
        ready_model_id: health_snapshot.ready_model_id,
        default_model_id: Some(effective_default_model_id(
            &generation_context.instance_paths,
        )),
        model_load_in_flight: health_snapshot.model_load_in_flight,
        neural_engine_prefill: health_snapshot.neural_engine_prefill,
    };
    send_attributed_streaming_response(
        daemon_attribution_log,
        DaemonPerformanceOperation::DaemonIpcStatus,
        &mut streaming_response_writer,
        &status_response,
    )
    .await?;
    streaming_response_writer.close().await
}

/// Streams one chat generation under its attributed measurement region.
async fn stream_chat_generate(
    generation_context: &DaemonIpcGenerationContext,
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    chat_request: super::chat::DaemonIpcChatRequest,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    daemon_attribution_log
        .measure_async_operation_best_effort(
            DaemonPerformanceOperation::DaemonIpcChatGenerate,
            || {
                super::chat::stream_chat_generation(
                    generation_context,
                    chat_request,
                    streaming_response_writer,
                )
            },
            |stream_outcome| {
                if stream_outcome.is_ok() {
                    DaemonPerformanceMeasurement::success()
                } else {
                    DaemonPerformanceMeasurement::failure()
                }
            },
        )
        .await
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
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    operation: DaemonPerformanceOperation,
    streaming_response_writer: &mut StreamingResponseWriter,
    daemon_response: &DaemonResponse,
) -> Result<(), DaemonTransportError> {
    daemon_attribution_log
        .measure_async_operation_best_effort(
            operation,
            || streaming_response_writer.send_response(daemon_response),
            |send_outcome| {
                if send_outcome.is_ok() {
                    DaemonPerformanceMeasurement::success()
                } else {
                    DaemonPerformanceMeasurement::failure()
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
