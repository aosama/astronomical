//! Admits one embeddings command through the shared model-swap and active-request owner.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{EmbeddingsCommand, ProtocolError, RequestId};
use tokio::time::{Instant, timeout};

use crate::worker::loop_types::WorkerLoopState;
use crate::{
    CompletionAttributionLog, EmbeddingsExecutionError, EmbeddingsOutput, GenerationPerformanceLog,
    GenerationStartError, RuntimeModelPolicy, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot, WorkerProcess,
    worker::health::{clear_active_request_progress, publish_activity},
    worker::loop_types::{ActiveEmbeddingsGeneration, ActiveWorkerRequest},
    worker::process::containment::{cancel_active_generation, contain_worker_failure},
    worker::process::model_swap::{ModelSwapWaitOutcome, wait_for_model_swap},
    worker::process::startup_runtime::wait_for_startup_runtime_configuration,
};

/// Bounded embeddings execution deadline; the single forward pass has no
/// progress events, so one deadline protects against a stuck engine.
const EMBEDDINGS_EXECUTION_TIMEOUT_SECS: u64 = 60;

/// What the embeddings admission pipeline decided for one command.
///
/// Extracted admission units return these outcomes instead of answering the
/// client themselves: the owning handler performs every sender send and
/// containment call in the original order (`tokio::sync::oneshot::Sender`
/// is not cloneable, so the answer must stay with the owner).
enum EmbeddingsAdmissionOutcome {
    /// The worker accepted the command; admit the active request and confirm.
    Started {
        request_id: RequestId,
        execution_started_at: Instant,
    },
    /// The client result channel closed during admission; nothing to answer.
    ClientDisconnected,
    /// Admission refused before any worker execution; forward this error.
    StartRejected(GenerationStartError),
    /// The worker failed after the refusal notice; containment must follow.
    WorkerFailed(WorkerControlError),
}

/// Outcome of making the requested model resident before dispatch.
enum EmbeddingsModelSwapOutcome {
    /// The requested model is resident and dispatch can proceed.
    ModelReady,
    /// The client result channel closed while the swap was in flight.
    ClientDisconnected,
    /// Admission refused; forward this start error to the client.
    StartRejected(GenerationStartError),
    /// The worker failed; the handler must send the refusal, then contain.
    WorkerFailed(WorkerControlError),
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_generate_embeddings_command(
    worker_process: &mut WorkerProcess,
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    embeddings_command: EmbeddingsCommand,
    start_sender: tokio::sync::oneshot::Sender<Result<(), GenerationStartError>>,
    embeddings_result_sender: tokio::sync::mpsc::Sender<
        Result<EmbeddingsOutput, EmbeddingsExecutionError>,
    >,
    _admitted_at: Instant,
    _queue_wait_elapsed: std::time::Duration,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
    model_load_timeout: std::time::Duration,
    cancellation_acknowledgement_timeout: std::time::Duration,
) -> Result<(), WorkerControlError> {
    if active_request.is_some() {
        let _send_outcome = start_sender.send(Err(GenerationStartError::CapacityUnavailable));
        tracing::error!("received GenerateEmbeddings while another request is active");
        return Ok(());
    }
    match admit_embeddings_command(
        worker_process,
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_request,
        performance_log,
        completion_log,
        embeddings_command,
        model_policy_catalog,
        model_load_timeout,
        &embeddings_result_sender,
    )
    .await?
    {
        EmbeddingsAdmissionOutcome::Started {
            request_id,
            execution_started_at,
        } => {
            admit_embeddings_request(
                active_request,
                active_generation_permit,
                request_id,
                execution_started_at,
                embeddings_result_sender,
                health_snapshot,
            );
        }
        EmbeddingsAdmissionOutcome::ClientDisconnected => return Ok(()),
        EmbeddingsAdmissionOutcome::StartRejected(start_error) => {
            let _send_outcome = start_sender.send(Err(start_error));
            return Ok(());
        }
        EmbeddingsAdmissionOutcome::WorkerFailed(start_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_worker_failure(worker_process, health_snapshot, active_request, start_error)
                .await;
            *is_ready = false;
            return Ok(());
        }
    }
    if start_sender.send(Ok(())).is_err() {
        cancel_active_generation(
            worker_process,
            health_snapshot,
            active_request,
            cancellation_acknowledgement_timeout,
            model_load_timeout,
            is_ready,
        )
        .await;
    }
    Ok(())
}

/// Runs the whole pre-execution admission pipeline: startup runtime
/// configuration, model residency, the client-connection and capability
/// gates, and the worker command dispatch. Failures come back as outcomes
/// so the owning handler answers the client and contains the worker.
#[allow(clippy::too_many_arguments)]
async fn admit_embeddings_command(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    embeddings_command: EmbeddingsCommand,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
    model_load_timeout: std::time::Duration,
    embeddings_result_sender: &tokio::sync::mpsc::Sender<
        Result<EmbeddingsOutput, EmbeddingsExecutionError>,
    >,
) -> Result<EmbeddingsAdmissionOutcome, WorkerControlError> {
    wait_for_startup_runtime_configuration(
        worker_process,
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_request,
        performance_log,
        completion_log,
        model_load_timeout,
    )
    .await?;
    let loaded_model_id = health_snapshot
        .read()
        .ok()
        .and_then(|snapshot| snapshot.ready_model_id.clone());
    let requested_model = &embeddings_command.model;
    if loaded_model_id.as_deref() != Some(requested_model) {
        match wait_for_embeddings_model_swap(
            worker_process,
            health_snapshot,
            is_ready,
            model_load_deadline,
            active_request,
            performance_log,
            completion_log,
            requested_model,
            model_policy_catalog,
            model_load_timeout,
            embeddings_result_sender,
        )
        .await
        {
            EmbeddingsModelSwapOutcome::ModelReady => {}
            EmbeddingsModelSwapOutcome::ClientDisconnected => {
                return Ok(EmbeddingsAdmissionOutcome::ClientDisconnected);
            }
            EmbeddingsModelSwapOutcome::StartRejected(start_error) => {
                return Ok(EmbeddingsAdmissionOutcome::StartRejected(start_error));
            }
            EmbeddingsModelSwapOutcome::WorkerFailed(swap_error) => {
                return Ok(EmbeddingsAdmissionOutcome::WorkerFailed(swap_error));
            }
        }
    }
    if embeddings_result_sender.is_closed() {
        return Ok(EmbeddingsAdmissionOutcome::ClientDisconnected);
    }
    if !embeddings_command_supported(health_snapshot, embeddings_command.dimensions) {
        return Ok(EmbeddingsAdmissionOutcome::StartRejected(
            GenerationStartError::WorkerUnavailable,
        ));
    }
    Ok(dispatch_embeddings_command(worker_process, embeddings_command).await)
}

/// Drives the model swap when the requested model is not the resident one.
///
/// Rejection and worker-failure outcomes return the sender answer and the
/// contained failure separately: the owning handler sends first, then runs
/// containment, exactly as the original inline sequence did.
#[allow(clippy::too_many_arguments)]
async fn wait_for_embeddings_model_swap(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    requested_model: &str,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
    model_load_timeout: std::time::Duration,
    embeddings_result_sender: &tokio::sync::mpsc::Sender<
        Result<EmbeddingsOutput, EmbeddingsExecutionError>,
    >,
) -> EmbeddingsModelSwapOutcome {
    let mut _swap_load_elapsed = std::time::Duration::ZERO;
    let Some(model_policy) = model_policy_catalog.get(requested_model) else {
        return EmbeddingsModelSwapOutcome::StartRejected(GenerationStartError::WorkerUnavailable);
    };
    let expected_configuration_generation = health_snapshot.read().ok().and_then(|snapshot| {
        snapshot
            .worker_runtime_feature_configuration
            .as_ref()
            .map(|configuration| configuration.configuration_generation.clone())
    });
    let expected_model_runtime_configuration = model_policy
        .worker_model_configuration
        .runtime_configuration();
    let swap_load_started_at = Instant::now();
    if let Err(swap_error) = worker_process
        .swap_model(
            model_policy.model_directory.to_string_lossy().into_owned(),
            model_policy.worker_model_configuration.clone(),
        )
        .await
    {
        return EmbeddingsModelSwapOutcome::WorkerFailed(swap_error);
    }
    let mut swap_loop_state = WorkerLoopState {
        worker_process: &mut *worker_process,
        health_snapshot,
        is_ready: &mut *is_ready,
        model_load_deadline: &mut *model_load_deadline,
        active_request: &mut *active_request,
        performance_log: &mut *performance_log,
        completion_log: &mut *completion_log,
        model_load_timeout,
    };
    let mut swap_wait = Box::pin(timeout(
        model_load_timeout,
        wait_for_model_swap(
            &mut swap_loop_state,
            expected_configuration_generation.as_deref(),
            &expected_model_runtime_configuration,
        ),
    ));
    let (swap_outcome, client_disconnected_during_swap) = tokio::select! {
        swap_outcome = &mut swap_wait => (swap_outcome, false),
        () = embeddings_result_sender.closed() => ((&mut swap_wait).await, true),
    };
    drop(swap_wait);
    _swap_load_elapsed = swap_load_started_at.elapsed();
    let swap_outcome = swap_outcome
        .map_err(|_| WorkerControlError::ModelLoadTimeout {
            model_load_timeout_millis: model_load_timeout.as_millis(),
        })
        .and_then(|outcome| outcome);
    match swap_outcome {
        Ok(ModelSwapWaitOutcome::Loaded) => {}
        Ok(ModelSwapWaitOutcome::Rejected {
            model_load_failure_reason,
        }) => {
            return EmbeddingsModelSwapOutcome::StartRejected(
                GenerationStartError::ModelLoadFailed {
                    model_load_failure_reason,
                },
            );
        }
        Err(swap_error) => {
            return EmbeddingsModelSwapOutcome::WorkerFailed(swap_error);
        }
    }
    if client_disconnected_during_swap {
        return EmbeddingsModelSwapOutcome::ClientDisconnected;
    }
    EmbeddingsModelSwapOutcome::ModelReady
}

/// Hands the command to the worker process and classifies the start result.
async fn dispatch_embeddings_command(
    worker_process: &mut WorkerProcess,
    embeddings_command: EmbeddingsCommand,
) -> EmbeddingsAdmissionOutcome {
    let request_id = embeddings_command.request_id;
    let execution_started_at = Instant::now();
    match worker_process
        .start_embeddings_generation(embeddings_command)
        .await
    {
        Ok(()) => EmbeddingsAdmissionOutcome::Started {
            request_id,
            execution_started_at,
        },
        Err(WorkerControlError::Protocol(ProtocolError::OutgoingMessageTooLarge {
            actual_message_bytes,
            maximum_message_bytes,
        })) => EmbeddingsAdmissionOutcome::StartRejected(GenerationStartError::RequestTooLarge {
            actual_ipc_message_bytes: actual_message_bytes,
            maximum_ipc_message_bytes: maximum_message_bytes,
        }),
        Err(start_error) => EmbeddingsAdmissionOutcome::WorkerFailed(start_error),
    }
}

/// Checks the resident model's advertised embedding surface: the vector
/// width must be positive, and explicit dimensions may only truncate.
fn embeddings_command_supported(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    requested_dimensions: Option<u32>,
) -> bool {
    health_snapshot.read().ok().is_some_and(|snapshot| {
        snapshot
            .ready_model_capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.embeddings.as_ref())
            .is_some_and(|embedding_capabilities| {
                embedding_capabilities.vector_width > 0
                    && requested_dimensions
                        .is_none_or(|dimensions| dimensions <= embedding_capabilities.vector_width)
            })
    })
}

/// Takes ownership of the active-request slot for this command and clears
/// the stale per-request progress the previous request left behind.
fn admit_embeddings_request(
    active_request: &mut Option<ActiveWorkerRequest>,
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    request_id: RequestId,
    execution_started_at: Instant,
    embeddings_result_sender: tokio::sync::mpsc::Sender<
        Result<EmbeddingsOutput, EmbeddingsExecutionError>,
    >,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    *active_request = Some(ActiveWorkerRequest::Embeddings(
        ActiveEmbeddingsGeneration {
            _active_generation_permit: active_generation_permit,
            request_id,
            terminal_received_at: None,
            execution_deadline: execution_started_at
                + std::time::Duration::from_secs(EMBEDDINGS_EXECUTION_TIMEOUT_SECS),
            embeddings_result_sender,
            terminal_outcome: None,
        },
    ));
    clear_active_request_progress(health_snapshot);
    publish_activity(health_snapshot, WorkerActivity::Idle);
}

#[cfg(test)]
#[path = "embeddings_request_tests.rs"]
mod embeddings_request_tests;
