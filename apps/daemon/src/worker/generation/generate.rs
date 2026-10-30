//! Owns generation admission for the worker loop.
//!
//! Keeping model selection, IPC-size rejection, and active-request publication
//! together prevents the event loop from exposing partially started requests.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use astronomical_ipc_protocol::ProtocolError;
use tokio::time::{Instant, timeout};

use crate::worker::loop_types::WorkerLoopState;
use crate::{
    GenerationStartError, RuntimeModelPolicy, WorkerActivity, WorkerControlError,
    worker::health::{clear_active_request_progress, publish_activity},
    worker::loop_types::{ActiveGeneration, ActiveWorkerRequest},
    worker::process::containment::{cancel_active_generation, contain_worker_failure},
    worker::process::model_swap::{ModelSwapWaitOutcome, wait_for_model_swap},
    worker::process::startup_runtime::wait_for_startup_runtime_configuration,
};

/// The model-selection outcome: proceed, or the start failure the caller
/// answers with (the swap-failure case carries the process error because the
/// caller contains it after answering).
enum ModelSelectionOutcome {
    Proceed,
    Rejected(GenerationStartError),
    SwapFailed(WorkerControlError),
}

/// The generation-start outcome: running on the worker, or the start failure
/// the caller answers with (the failed case carries the process error for
/// containment after the answer).
enum GenerationStartOutcome {
    Started,
    Rejected(GenerationStartError),
    Failed(WorkerControlError),
}

pub(in crate::worker) async fn handle_generate_command(
    worker_loop_state: &mut WorkerLoopState<'_>,
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    generation_command: astronomical_ipc_protocol::ChatGenerationCommand,
    start_sender: tokio::sync::oneshot::Sender<Result<(), GenerationStartError>>,
    stream_event_sender: tokio::sync::mpsc::Sender<crate::ChatGenerationStreamEvent>,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
    cancellation_acknowledgement_timeout: Duration,
) -> Result<(), WorkerControlError> {
    // The handle reserves this request's permit before sending the command, so
    // permit availability cannot reveal whether the loop owns an earlier request.
    if worker_loop_state.active_request.is_some() {
        let _send_outcome = start_sender.send(Err(GenerationStartError::CapacityUnavailable));
        tracing::error!(
            "received a Generate command while a generation is active; this indicates a queue bug"
        );
        return Ok(());
    }
    wait_for_startup_runtime_configuration(
        worker_loop_state.worker_process,
        worker_loop_state.health_snapshot,
        worker_loop_state.is_ready,
        worker_loop_state.model_load_deadline,
        worker_loop_state.active_request,
        worker_loop_state.performance_log,
        worker_loop_state.completion_log,
        worker_loop_state.model_load_timeout,
    )
    .await?;
    match ensure_requested_model_loaded(
        worker_loop_state,
        &generation_command.model,
        model_policy_catalog,
    )
    .await?
    {
        ModelSelectionOutcome::Proceed => {}
        ModelSelectionOutcome::Rejected(start_failure) => {
            let _send_outcome = start_sender.send(Err(start_failure));
            return Ok(());
        }
        ModelSelectionOutcome::SwapFailed(swap_error) => {
            contain_worker_failure(
                worker_loop_state.worker_process,
                worker_loop_state.health_snapshot,
                worker_loop_state.active_request,
                swap_error,
            )
            .await;
            *worker_loop_state.is_ready = false;
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            return Ok(());
        }
    }
    let request_id = generation_command.request_id;
    let request_max_output_tokens = generation_command.settings.max_output_tokens;
    tracing::info!(
        request_id = request_id.value(),
        max_output_tokens = request_max_output_tokens,
        "starting worker generation"
    );
    match start_worker_generation_or_reject(worker_loop_state, request_id, generation_command)
        .await?
    {
        GenerationStartOutcome::Started => {}
        GenerationStartOutcome::Rejected(start_failure) => {
            let _send_outcome = start_sender.send(Err(start_failure));
            return Ok(());
        }
        GenerationStartOutcome::Failed(start_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_worker_failure(
                worker_loop_state.worker_process,
                worker_loop_state.health_snapshot,
                worker_loop_state.active_request,
                start_error,
            )
            .await;
            *worker_loop_state.is_ready = false;
            return Ok(());
        }
    }
    *worker_loop_state.active_request = Some(assemble_active_chat_generation(
        active_generation_permit,
        request_id,
        request_max_output_tokens,
        stream_event_sender,
    ));
    clear_active_request_progress(worker_loop_state.health_snapshot);
    publish_activity(
        worker_loop_state.health_snapshot,
        WorkerActivity::PromptProcessing,
    );
    if start_sender.send(Ok(())).is_err() {
        cancel_active_generation(
            worker_loop_state.worker_process,
            worker_loop_state.health_snapshot,
            worker_loop_state.active_request,
            cancellation_acknowledgement_timeout,
            worker_loop_state.model_load_timeout,
            worker_loop_state.is_ready,
        )
        .await;
    }
    Ok(())
}

/// Ensures the requested model is the loaded one: an unmapped model with no
/// worker is rejected, a policy-backed different model triggers a swap with
/// its bounded wait, and every rejection answers the start sender before the
/// `false` outcome. `Err` carries only the model-load timeout, the one
/// failure that escapes the request lifecycle.
async fn ensure_requested_model_loaded(
    worker_loop_state: &mut WorkerLoopState<'_>,
    requested_model: &str,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
) -> Result<ModelSelectionOutcome, WorkerControlError> {
    // REST validates model IDs, but direct WorkerHandle callers share this
    // boundary and therefore still need the empty-worker guard.
    let loaded_model_id = worker_loop_state
        .health_snapshot
        .read()
        .ok()
        .and_then(|snapshot| snapshot.ready_model_id.clone());
    if loaded_model_id.as_deref() == Some(requested_model) {
        return Ok(ModelSelectionOutcome::Proceed);
    }
    let requested_model_policy = model_policy_catalog.get(requested_model);
    if requested_model_policy.is_none() && loaded_model_id.is_none() {
        tracing::warn!(
            requested_model = %requested_model,
            loaded_model = ?loaded_model_id,
            "rejected generation for an unmapped model"
        );
        return Ok(ModelSelectionOutcome::Rejected(
            GenerationStartError::WorkerUnavailable,
        ));
    }
    let Some(model_policy) = requested_model_policy else {
        return Ok(ModelSelectionOutcome::Proceed);
    };
    let expected_configuration_generation =
        worker_loop_state
            .health_snapshot
            .read()
            .ok()
            .and_then(|snapshot| {
                snapshot
                    .worker_runtime_feature_configuration
                    .as_ref()
                    .map(|configuration| configuration.configuration_generation.clone())
            });
    let expected_model_runtime_configuration = model_policy
        .worker_model_configuration
        .runtime_configuration();
    tracing::info!(
        requested_model = %requested_model,
        loaded_model = ?loaded_model_id,
        model_directory = %model_policy.model_directory.display(),
        "loading model to match request"
    );
    if let Err(swap_error) = worker_loop_state
        .worker_process
        .swap_model(
            model_policy.model_directory.to_string_lossy().into_owned(),
            model_policy.worker_model_configuration.clone(),
        )
        .await
    {
        tracing::error!(error = %swap_error, "SwapModel command failed");
        return Ok(ModelSelectionOutcome::SwapFailed(swap_error));
    }
    let model_swap_outcome = timeout(
        worker_loop_state.model_load_timeout,
        wait_for_model_swap(
            &mut WorkerLoopState {
                worker_process: &mut *worker_loop_state.worker_process,
                health_snapshot: worker_loop_state.health_snapshot,
                is_ready: &mut *worker_loop_state.is_ready,
                model_load_deadline: &mut *worker_loop_state.model_load_deadline,
                active_request: &mut *worker_loop_state.active_request,
                performance_log: &mut *worker_loop_state.performance_log,
                completion_log: &mut *worker_loop_state.completion_log,
                model_load_timeout: worker_loop_state.model_load_timeout,
            },
            expected_configuration_generation.as_deref(),
            &expected_model_runtime_configuration,
        ),
    )
    .await
    .map_err(|_| WorkerControlError::ModelLoadTimeout {
        model_load_timeout_millis: worker_loop_state.model_load_timeout.as_millis(),
    })
    .and_then(|model_swap_outcome| model_swap_outcome);
    match model_swap_outcome {
        Ok(ModelSwapWaitOutcome::Loaded) => {}
        Ok(ModelSwapWaitOutcome::Rejected {
            model_load_failure_reason,
        }) => {
            return Ok(ModelSelectionOutcome::Rejected(
                GenerationStartError::ModelLoadFailed {
                    model_load_failure_reason,
                },
            ));
        }
        Err(swap_error) => {
            tracing::error!(error = %swap_error, "model swap failed during wait for ModelSwapped");
            return Ok(ModelSelectionOutcome::SwapFailed(swap_error));
        }
    }
    tracing::info!(
        requested_model = %requested_model,
        "model swap completed successfully"
    );
    Ok(ModelSelectionOutcome::Proceed)
}

/// Starts the generation on the worker process. The `Rejected` and
/// `Failed` outcomes carry what the caller answers and contains.
async fn start_worker_generation_or_reject(
    worker_loop_state: &mut WorkerLoopState<'_>,
    request_id: astronomical_ipc_protocol::RequestId,
    generation_command: astronomical_ipc_protocol::ChatGenerationCommand,
) -> Result<GenerationStartOutcome, WorkerControlError> {
    tracing::info!(
        request_id = request_id.value(),
        "starting worker generation"
    );
    match worker_loop_state
        .worker_process
        .start_generation(generation_command)
        .await
    {
        Ok(()) => Ok(GenerationStartOutcome::Started),
        Err(WorkerControlError::Protocol(ProtocolError::OutgoingMessageTooLarge {
            actual_message_bytes: actual_ipc_message_bytes,
            maximum_message_bytes: maximum_ipc_message_bytes,
        })) => {
            tracing::warn!(
                request_id = request_id.value(),
                actual_ipc_message_bytes,
                maximum_ipc_message_bytes,
                "rejected generation command that exceeds the IPC frame limit"
            );
            Ok(GenerationStartOutcome::Rejected(
                GenerationStartError::RequestTooLarge {
                    actual_ipc_message_bytes,
                    maximum_ipc_message_bytes,
                },
            ))
        }
        Err(start_error) => Ok(GenerationStartOutcome::Failed(start_error)),
    }
}

/// Pure: assembles the active-request state a running chat generation owns
/// for its whole lifetime.
fn assemble_active_chat_generation(
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    request_id: astronomical_ipc_protocol::RequestId,
    request_max_output_tokens: u16,
    stream_event_sender: tokio::sync::mpsc::Sender<crate::ChatGenerationStreamEvent>,
) -> ActiveWorkerRequest {
    ActiveWorkerRequest::Chat(ActiveGeneration {
        _active_generation_permit: active_generation_permit,
        generated_token_count: 0,
        generation_started_at: None,
        generation_preparation_started_at: None,
        generation_preparation_elapsed_millis: None,
        first_decode_forward_elapsed_millis: None,
        time_to_first_output_millis: None,
        final_resident_expert_count: None,
        final_resident_expert_payload_bytes: None,
        latest_generation_progress_token_count: 0,
        max_output_tokens: request_max_output_tokens,
        next_sequence_number: 0,
        next_tool_call_index: 0,
        completed_tool_calls: Vec::new(),
        request_started_at: Instant::now(),
        prefill_elapsed_millis: 0,
        maximum_mlx_peak_memory_bytes: None,
        last_mlx_active_memory_bytes: None,
        request_id,
        stream_event_sender,
    })
}
