//! Admits one embeddings command through the shared model-swap and active-request owner.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{EmbeddingsCommand, ProtocolError};
use tokio::time::{Instant, timeout};

use crate::{
    CompletionAttributionLog, EmbeddingsExecutionError, EmbeddingsOutput, GenerationPerformanceLog,
    GenerationStartError, RuntimeModelPolicy, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot, WorkerProcess,
    worker_containment::{cancel_active_generation, contain_worker_failure},
    worker_health::{clear_active_request_progress, publish_activity},
    worker_loop_types::{ActiveEmbeddingsGeneration, ActiveWorkerRequest},
    worker_model_swap::{ModelSwapWaitOutcome, wait_for_model_swap},
    worker_startup_runtime::wait_for_startup_runtime_configuration,
};

/// Bounded embeddings execution deadline; the single forward pass has no
/// progress events, so one deadline protects against a stuck engine.
const EMBEDDINGS_EXECUTION_TIMEOUT_SECS: u64 = 60;

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
    let mut _swap_load_elapsed = std::time::Duration::ZERO;
    if loaded_model_id.as_deref() != Some(requested_model) {
        let Some(model_policy) = model_policy_catalog.get(requested_model) else {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            return Ok(());
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
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_worker_failure(worker_process, health_snapshot, active_request, swap_error)
                .await;
            *is_ready = false;
            return Ok(());
        }
        let mut swap_wait = Box::pin(timeout(
            model_load_timeout,
            wait_for_model_swap(
                worker_process,
                health_snapshot,
                is_ready,
                model_load_deadline,
                active_request,
                performance_log,
                completion_log,
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
                let _send_outcome = start_sender.send(Err(GenerationStartError::ModelLoadFailed {
                    model_load_failure_reason,
                }));
                return Ok(());
            }
            Err(swap_error) => {
                let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
                contain_worker_failure(worker_process, health_snapshot, active_request, swap_error)
                    .await;
                *is_ready = false;
                return Ok(());
            }
        }
        if client_disconnected_during_swap {
            return Ok(());
        }
    }
    if embeddings_result_sender.is_closed() {
        return Ok(());
    }
    let supports_request = health_snapshot.read().ok().is_some_and(|snapshot| {
        snapshot
            .ready_model_capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.embeddings.as_ref())
            .is_some_and(|embedding_capabilities| {
                embedding_capabilities.vector_width > 0
                    && embeddings_command
                        .dimensions
                        .is_none_or(|dimensions| dimensions <= embedding_capabilities.vector_width)
            })
    });
    if !supports_request {
        let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
        return Ok(());
    }

    let request_id = embeddings_command.request_id;
    let execution_started_at = Instant::now();
    match worker_process
        .start_embeddings_generation(embeddings_command)
        .await
    {
        Ok(()) => {}
        Err(WorkerControlError::Protocol(ProtocolError::OutgoingMessageTooLarge {
            actual_message_bytes,
            maximum_message_bytes,
        })) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::RequestTooLarge {
                actual_ipc_message_bytes: actual_message_bytes,
                maximum_ipc_message_bytes: maximum_message_bytes,
            }));
            return Ok(());
        }
        Err(start_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_worker_failure(worker_process, health_snapshot, active_request, start_error)
                .await;
            *is_ready = false;
            return Ok(());
        }
    }
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
