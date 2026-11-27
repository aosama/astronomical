//! Owns generation admission for the worker loop.
//!
//! Keeping model selection, IPC-size rejection, and active-request publication
//! together prevents the event loop from exposing partially started requests.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ipc_protocol::ProtocolError;
use tokio::time::Instant;

use crate::worker::loop_types::WorkerLoopState;
use crate::worker::request_start::{
    GenerationStartSender, contain_failed_worker, loaded_model_id,
    published_configuration_generation,
};
use crate::{
    GenerationStartError, RuntimeModelPolicy, WorkerActivity, WorkerControlError,
    worker::health::{clear_active_request_progress, publish_activity},
    worker::loop_types::{ActiveGeneration, ActiveWorkerRequest},
    worker::process::containment::cancel_active_generation,
    worker::process::model_swap::{ModelSwapWaitOutcome, wait_for_model_swap_within_budget},
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

/// What the loop owns once the worker has started a chat generation.
struct StartedChatGeneration {
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    request_id: ipc_protocol::RequestId,
    request_max_output_tokens: u16,
    stream_event_sender: tokio::sync::mpsc::Sender<crate::ChatGenerationStreamEvent>,
    chat_work: crate::worker::loop_types::ChatWork,
}

pub(in crate::worker) async fn handle_generate_command(
    worker_loop_state: &mut WorkerLoopState<'_>,
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    queued_chat_request: crate::worker::loop_types::QueuedChatRequest,
    start_sender: GenerationStartSender,
    stream_event_sender: tokio::sync::mpsc::Sender<crate::ChatGenerationStreamEvent>,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
    cancellation_acknowledgement_timeout: Duration,
) -> Result<(), WorkerControlError> {
    let chat_work = queued_chat_request.chat_work;
    let generation_command = queued_chat_request.generation_command;
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
    let model_selection_outcome = ensure_requested_model_loaded(
        worker_loop_state,
        &generation_command.model,
        model_policy_catalog,
    )
    .await?;
    let Some(start_sender) =
        settle_model_selection(worker_loop_state, model_selection_outcome, start_sender).await
    else {
        return Ok(());
    };
    let request_id = generation_command.request_id;
    let request_max_output_tokens = generation_command.settings.max_output_tokens;
    let generation_start_outcome = start_worker_generation_or_reject(
        worker_loop_state,
        request_id,
        generation_command,
        chat_work,
    )
    .await?;
    let Some(start_sender) =
        settle_generation_start(worker_loop_state, generation_start_outcome, start_sender).await
    else {
        return Ok(());
    };
    activate_started_generation(
        worker_loop_state,
        StartedChatGeneration {
            active_generation_permit,
            request_id,
            request_max_output_tokens,
            stream_event_sender,
            chat_work,
        },
        start_sender,
        cancellation_acknowledgement_timeout,
    )
    .await;
    Ok(())
}

/// Ends the request when model selection stopped it, answering the start
/// sender, and hands the sender back when the request goes on. A failed swap
/// contains the worker before the answer goes out.
async fn settle_model_selection(
    worker_loop_state: &mut WorkerLoopState<'_>,
    model_selection_outcome: ModelSelectionOutcome,
    start_sender: GenerationStartSender,
) -> Option<GenerationStartSender> {
    match model_selection_outcome {
        ModelSelectionOutcome::Proceed => Some(start_sender),
        ModelSelectionOutcome::Rejected(start_failure) => {
            let _send_outcome = start_sender.send(Err(start_failure));
            None
        }
        ModelSelectionOutcome::SwapFailed(swap_error) => {
            contain_failed_worker(worker_loop_state, swap_error).await;
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            None
        }
    }
}

/// Ends the request when the worker did not start it, answering the start
/// sender, and hands the sender back when the generation is running. A
/// failed start answers before the worker is contained.
async fn settle_generation_start(
    worker_loop_state: &mut WorkerLoopState<'_>,
    generation_start_outcome: GenerationStartOutcome,
    start_sender: GenerationStartSender,
) -> Option<GenerationStartSender> {
    match generation_start_outcome {
        GenerationStartOutcome::Started => Some(start_sender),
        GenerationStartOutcome::Rejected(start_failure) => {
            let _send_outcome = start_sender.send(Err(start_failure));
            None
        }
        GenerationStartOutcome::Failed(start_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_failed_worker(worker_loop_state, start_error).await;
            None
        }
    }
}

/// Publishes the started generation as the loop's active request and tells
/// the caller it started. A caller that already hung up has its generation
/// cancelled at once.
async fn activate_started_generation(
    worker_loop_state: &mut WorkerLoopState<'_>,
    started_generation: StartedChatGeneration,
    start_sender: GenerationStartSender,
    cancellation_acknowledgement_timeout: Duration,
) {
    *worker_loop_state.active_request = Some(assemble_active_chat_generation(
        started_generation.active_generation_permit,
        started_generation.request_id,
        started_generation.request_max_output_tokens,
        started_generation.stream_event_sender,
        started_generation.chat_work,
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
    let loaded_model_id = loaded_model_id(worker_loop_state.health_snapshot);
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
        published_configuration_generation(worker_loop_state.health_snapshot);
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
    let model_swap_outcome = wait_for_model_swap_within_budget(
        worker_loop_state,
        expected_configuration_generation.as_deref(),
        &expected_model_runtime_configuration,
    )
    .await;
    let model_selection_outcome = model_selection_after_swap_wait(model_swap_outcome);
    if matches!(model_selection_outcome, ModelSelectionOutcome::Proceed) {
        tracing::info!(
            requested_model = %requested_model,
            "model swap completed successfully"
        );
    }
    Ok(model_selection_outcome)
}

/// Turns the end of the model-swap wait into what model selection answers: the
/// request goes on once the model is loaded, is rejected when the worker
/// refused the load, and treats any other wait failure as a failed swap.
fn model_selection_after_swap_wait(
    model_swap_outcome: Result<ModelSwapWaitOutcome, WorkerControlError>,
) -> ModelSelectionOutcome {
    match model_swap_outcome {
        Ok(ModelSwapWaitOutcome::Loaded) => ModelSelectionOutcome::Proceed,
        Ok(ModelSwapWaitOutcome::Rejected {
            model_load_failure_reason,
        }) => ModelSelectionOutcome::Rejected(GenerationStartError::ModelLoadFailed {
            model_load_failure_reason,
        }),
        Err(swap_error) => {
            tracing::error!(error = %swap_error, "model swap failed during wait for ModelSwapped");
            ModelSelectionOutcome::SwapFailed(swap_error)
        }
    }
}

/// Starts the generation on the worker process. The `Rejected` and
/// `Failed` outcomes carry what the caller answers and contains.
async fn start_worker_generation_or_reject(
    worker_loop_state: &mut WorkerLoopState<'_>,
    request_id: ipc_protocol::RequestId,
    generation_command: ipc_protocol::ChatGenerationCommand,
    chat_work: crate::worker::loop_types::ChatWork,
) -> Result<GenerationStartOutcome, WorkerControlError> {
    tracing::info!(
        request_id = request_id.value(),
        max_output_tokens = generation_command.settings.max_output_tokens,
        ?chat_work,
        "starting worker generation"
    );
    let worker_dispatch_outcome = dispatch_chat_work(
        worker_loop_state.worker_process,
        generation_command,
        chat_work,
    )
    .await;
    match worker_dispatch_outcome {
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

/// Sends the worker command the queued chat work asks for: a generation, or
/// a prompt-token count that runs only the generation preparation.
async fn dispatch_chat_work(
    worker_process: &mut crate::WorkerProcess,
    generation_command: ipc_protocol::ChatGenerationCommand,
    chat_work: crate::worker::loop_types::ChatWork,
) -> Result<(), WorkerControlError> {
    match chat_work {
        crate::worker::loop_types::ChatWork::Generate => {
            worker_process.start_generation(generation_command).await
        }
        crate::worker::loop_types::ChatWork::CountPromptTokens => {
            worker_process.count_chat_tokens(generation_command).await
        }
    }
}

/// Pure: assembles the active-request state a running chat generation owns
/// for its whole lifetime.
fn assemble_active_chat_generation(
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    request_id: ipc_protocol::RequestId,
    request_max_output_tokens: u16,
    stream_event_sender: tokio::sync::mpsc::Sender<crate::ChatGenerationStreamEvent>,
    chat_work: crate::worker::loop_types::ChatWork,
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
        prefill_work: Default::default(),
        maximum_mlx_peak_memory_bytes: None,
        last_mlx_active_memory_bytes: None,
        request_id,
        stream_event_sender,
        chat_work,
        drafting_statistics: None,
    })
}

#[cfg(test)]
#[path = "generate_tests.rs"]
mod generate_tests;
