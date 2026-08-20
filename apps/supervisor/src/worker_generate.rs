//! Owns generation admission for the worker loop.
//!
//! Keeping model selection, IPC-size rejection, and active-request publication
//! together prevents the event loop from exposing partially started requests.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_ipc_protocol::ProtocolError;
use tokio::time::{Instant, timeout};

use crate::{
    GenerationPerformanceLog, GenerationStartError, RuntimeModelPolicy, WorkerActivity,
    WorkerControlError, WorkerHealthSnapshot, WorkerProcess,
    worker_containment::{cancel_active_generation, contain_worker_failure},
    worker_health::{clear_active_request_progress, publish_activity},
    worker_loop_types::ActiveGeneration,
    worker_model_swap::{ModelSwapWaitOutcome, wait_for_model_swap},
};

pub(super) async fn handle_generate_command(
    worker_process: &mut WorkerProcess,
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    generation_command: astronomical_ipc_protocol::ChatGenerationCommand,
    start_sender: tokio::sync::oneshot::Sender<Result<(), GenerationStartError>>,
    stream_event_sender: tokio::sync::mpsc::Sender<crate::ChatGenerationStreamEvent>,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_generation: &mut Option<ActiveGeneration>,
    performance_log: &mut GenerationPerformanceLog,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
    model_load_timeout: Duration,
    cancellation_acknowledgement_timeout: Duration,
) -> Result<(), WorkerControlError> {
    // The handle reserves this request's permit before sending the command, so
    // permit availability cannot reveal whether the loop owns an earlier request.
    if active_generation.is_some() {
        let _send_outcome = start_sender.send(Err(GenerationStartError::CapacityUnavailable));
        tracing::error!(
            "received a Generate command while a generation is active; this indicates a queue bug"
        );
        return Ok(());
    }
    // REST validates model IDs, but direct WorkerHandle callers share this
    // boundary and therefore still need the empty-worker guard.
    let loaded_model_id = health_snapshot
        .read()
        .ok()
        .and_then(|snapshot| snapshot.ready_model_id.clone());
    let requested_model = &generation_command.model;
    if loaded_model_id.as_deref() != Some(requested_model) {
        let requested_model_policy = model_policy_catalog.get(requested_model);
        if requested_model_policy.is_none() && loaded_model_id.is_none() {
            tracing::warn!(
                requested_model = %requested_model,
                loaded_model = ?loaded_model_id,
                "rejected generation for an unmapped model"
            );
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            return Ok(());
        }
        if let Some(model_policy) = requested_model_policy {
            tracing::info!(
                requested_model = %requested_model,
                loaded_model = ?loaded_model_id,
                model_directory = %model_policy.model_directory.display(),
                "loading model to match request"
            );
            if let Err(swap_error) = worker_process
                .swap_model(
                    model_policy.model_directory.to_string_lossy().into_owned(),
                    model_policy.worker_model_configuration.clone(),
                )
                .await
            {
                tracing::error!(error = %swap_error, "SwapModel command failed");
                contain_worker_failure(
                    worker_process,
                    health_snapshot,
                    active_generation,
                    swap_error,
                )
                .await;
                *is_ready = false;
                let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
                return Ok(());
            }
            let model_swap_outcome = timeout(
                model_load_timeout,
                wait_for_model_swap(
                    worker_process,
                    health_snapshot,
                    is_ready,
                    model_load_deadline,
                    active_generation,
                    performance_log,
                ),
            )
            .await
            .map_err(|_| WorkerControlError::ModelLoadTimeout {
                model_load_timeout_millis: model_load_timeout.as_millis(),
            })
            .and_then(|model_swap_outcome| model_swap_outcome);
            match model_swap_outcome {
                Ok(ModelSwapWaitOutcome::Loaded) => {}
                Ok(ModelSwapWaitOutcome::Rejected {
                    model_load_failure_reason,
                }) => {
                    let _send_outcome =
                        start_sender.send(Err(GenerationStartError::ModelLoadFailed {
                            model_load_failure_reason,
                        }));
                    return Ok(());
                }
                Err(swap_error) => {
                    tracing::error!(error = %swap_error, "model swap failed during wait for ModelSwapped");
                    contain_worker_failure(
                        worker_process,
                        health_snapshot,
                        active_generation,
                        swap_error,
                    )
                    .await;
                    *is_ready = false;
                    let _send_outcome =
                        start_sender.send(Err(GenerationStartError::WorkerUnavailable));
                    return Ok(());
                }
            }
            tracing::info!(
                requested_model = %requested_model,
                "model swap completed successfully"
            );
        }
    }
    let request_id = generation_command.request_id;
    let request_max_output_tokens = generation_command.settings.max_output_tokens;
    tracing::info!(
        request_id = request_id.value(),
        max_output_tokens = request_max_output_tokens,
        "starting worker generation"
    );
    match worker_process.start_generation(generation_command).await {
        Ok(()) => {}
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
            let _send_outcome = start_sender.send(Err(GenerationStartError::RequestTooLarge {
                actual_ipc_message_bytes,
                maximum_ipc_message_bytes,
            }));
            return Ok(());
        }
        Err(start_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_worker_failure(
                worker_process,
                health_snapshot,
                active_generation,
                start_error,
            )
            .await;
            *is_ready = false;
            return Ok(());
        }
    }
    *active_generation = Some(ActiveGeneration {
        _active_generation_permit: active_generation_permit,
        generated_token_count: 0,
        generation_started_at: None,
        generation_preparation_started_at: None,
        generation_preparation_elapsed_millis: None,
        first_decode_forward_elapsed_millis: None,
        time_to_first_output_millis: None,
        final_complete_expert_layer_count: None,
        final_complete_expert_payload_bytes: None,
        final_partial_expert_layer_count: None,
        final_partial_expert_payload_bytes: None,
        latest_generation_progress_token_count: 0,
        max_output_tokens: request_max_output_tokens,
        next_sequence_number: 0,
        next_tool_call_index: 0,
        request_started_at: Instant::now(),
        prefill_elapsed_millis: 0,
        maximum_mlx_peak_memory_bytes: None,
        last_mlx_active_memory_bytes: None,
        request_id,
        stream_event_sender,
    });
    clear_active_request_progress(health_snapshot);
    publish_activity(health_snapshot, WorkerActivity::PromptProcessing);
    if start_sender.send(Ok(())).is_err() {
        cancel_active_generation(
            worker_process,
            health_snapshot,
            active_generation,
            cancellation_acknowledgement_timeout,
        )
        .await;
    }
    Ok(())
}
