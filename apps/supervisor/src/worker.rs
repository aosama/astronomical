use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_ipc_protocol::ProtocolError;
use tokio::sync::mpsc;
use tokio::time::{Instant, MissedTickBehavior, interval, timeout};

use crate::worker_memory_limit::{
    MlxMemoryLimitUpdateOutcome, apply_mlx_memory_limit, contain_mlx_memory_limit_failure,
};
use crate::worker_model_swap::{ModelSwapWaitOutcome, wait_for_model_swap};
use crate::{
    ChatGenerationStreamErrorCode, GenerationPerformanceLog, GenerationStartError, WorkerActivity,
    WorkerControlError, WorkerHealthSnapshot, WorkerHealthStatus, WorkerProcess,
    WorkerTerminationOutcome,
    chat_generation_executor::{wait_for_deadline, wait_for_stream_disconnect},
    worker_containment::{
        cancel_active_generation, close_worker_if_running, contain_worker_failure,
        fail_active_generation,
    },
    worker_event_handler::handle_worker_event,
    worker_health::{
        clear_active_request_progress, publish_activity, publish_health,
        publish_pending_mlx_memory_ceiling,
    },
    worker_loop_types::{ActiveGeneration, WorkerLoopCommand},
};

// Keeping process-loop dependencies explicit is clearer than hiding them in a context object.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_worker(
    mut worker_process: WorkerProcess,
    mut command_receiver: mpsc::Receiver<WorkerLoopCommand>,
    health_snapshot: Arc<RwLock<WorkerHealthSnapshot>>,
    model_load_timeout: Duration,
    cancellation_acknowledgement_timeout: Duration,
    mut performance_log: GenerationPerformanceLog,
    mut model_directories: Arc<HashMap<String, PathBuf>>,
    mut max_output_tokens: u32,
) {
    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading),
    );
    let mut active_generation: Option<ActiveGeneration> = None;
    let mut is_ready = false;
    let mut model_load_deadline = Some(Instant::now() + model_load_timeout);
    let mut idle_memory_sampling_interval = interval(Duration::from_secs(1));
    idle_memory_sampling_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut pending_mlx_memory_ceiling_bytes = None;

    loop {
        if active_generation.is_none()
            && let Some(pending_mlx_memory_ceiling_bytes) = pending_mlx_memory_ceiling_bytes.take()
        {
            publish_pending_mlx_memory_ceiling(&health_snapshot, None);
            if let Err(memory_limit_error) = apply_mlx_memory_limit(
                &mut worker_process,
                pending_mlx_memory_ceiling_bytes,
                model_load_timeout,
                &health_snapshot,
                &mut is_ready,
                &mut model_load_deadline,
                &mut active_generation,
                &mut performance_log,
            )
            .await
            {
                contain_mlx_memory_limit_failure(
                    &mut worker_process,
                    &health_snapshot,
                    &mut active_generation,
                    &mut is_ready,
                    memory_limit_error,
                )
                .await;
            }
        }
        let stream_event_sender = active_generation
            .as_ref()
            .map(|generation| generation.stream_event_sender.clone());
        let is_worker_running = worker_process.process_id().is_some();
        let has_loaded_model = health_snapshot
            .read()
            .ok()
            .is_some_and(|worker_health_snapshot| worker_health_snapshot.ready_model_id.is_some());

        tokio::select! {
            worker_loop_command = command_receiver.recv() => {
                let Some(worker_loop_command) = worker_loop_command else {
                    close_worker_if_running(&mut worker_process).await;
                    return;
                };
                match worker_loop_command {
                    WorkerLoopCommand::Generate {
                        active_generation_permit,
                        generation_command,
                        start_sender,
                        stream_event_sender,
                    } => {
                        if !is_ready || !is_worker_running {
                            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
                            continue;
                        }
                        // The FIFO queue in WorkerHandle serializes requests, so
                        // active_generation should always be None when a Generate
                        // command arrives. This check is a defensive assertion.
                        if active_generation.is_some() {
                            let _send_outcome = start_sender.send(Err(GenerationStartError::CapacityUnavailable));
                            tracing::error!("received a Generate command while a generation is active; this indicates a queue bug");
                            continue;
                        }
                        // Load a mapped model before forwarding generation. REST canonicalizes
                        // and validates model IDs; the explicit empty-worker guard also protects
                        // direct WorkerHandle users from sending Generate before any model exists.
                        let loaded_model_id = health_snapshot
                            .read()
                            .ok()
                            .and_then(|snapshot| snapshot.ready_model_id.clone());
                        let requested_model = &generation_command.model;
                        if loaded_model_id.as_deref() != Some(requested_model) {
                            let requested_model_directory = model_directories.get(requested_model);
                            if requested_model_directory.is_none() && loaded_model_id.is_none() {
                                tracing::warn!(
                                    requested_model = %requested_model,
                                    loaded_model = ?loaded_model_id,
                                    "rejected generation for an unmapped model"
                                );
                                let _send_outcome = start_sender
                                    .send(Err(GenerationStartError::WorkerUnavailable));
                                continue;
                            }
                            if let Some(model_directory) = requested_model_directory {
                                tracing::info!(
                                    requested_model = %requested_model,
                                    loaded_model = ?loaded_model_id,
                                    model_directory = %model_directory.display(),
                                    "loading model to match request"
                                );
                                if let Err(swap_error) = worker_process
                                    .swap_model(
                                        model_directory.to_string_lossy().into_owned(),
                                        max_output_tokens,
                                    )
                                    .await
                                {
                                    tracing::error!(error = %swap_error, "SwapModel command failed");
                                    contain_worker_failure(
                                        &mut worker_process,
                                        &health_snapshot,
                                        &mut active_generation,
                                        swap_error,
                                    ).await;
                                    is_ready = false;
                                    let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
                                    continue;
                                }
                                // Drain events until we receive ModelSwapped or an error.
                                let model_swap_outcome = timeout(
                                    model_load_timeout,
                                    wait_for_model_swap(
                                        &mut worker_process,
                                        &health_snapshot,
                                        &mut is_ready,
                                        &mut model_load_deadline,
                                        &mut active_generation,
                                        &mut performance_log,
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
                                        let _send_outcome = start_sender
                                            .send(Err(GenerationStartError::ModelLoadFailed {
                                                model_load_failure_reason,
                                            }));
                                        continue;
                                    }
                                    Err(swap_error) => {
                                        tracing::error!(error = %swap_error, "model swap failed during wait for ModelSwapped");
                                        contain_worker_failure(
                                            &mut worker_process,
                                            &health_snapshot,
                                            &mut active_generation,
                                            swap_error,
                                        ).await;
                                        is_ready = false;
                                        let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
                                        continue;
                                    }
                                }
                                tracing::info!(
                                    requested_model = %requested_model,
                                    "model swap completed successfully"
                                );
                            }
                        }
                        let request_id = generation_command.request_id;
                        let max_output_tokens = generation_command.settings.max_output_tokens;
                        tracing::info!(request_id = request_id.value(), max_output_tokens,
                            "starting worker generation");
                        match worker_process.start_generation(generation_command).await {
                            Ok(()) => {}
                            Err(WorkerControlError::Protocol(
                                ProtocolError::OutgoingMessageTooLarge {
                                    actual_message_bytes: actual_ipc_message_bytes,
                                    maximum_message_bytes: maximum_ipc_message_bytes,
                                },
                            )) => {
                                tracing::warn!(
                                    request_id = request_id.value(),
                                    actual_ipc_message_bytes,
                                    maximum_ipc_message_bytes,
                                    "rejected generation command that exceeds the IPC frame limit"
                                );
                                let _send_outcome = start_sender.send(Err(
                                    GenerationStartError::RequestTooLarge {
                                        actual_ipc_message_bytes,
                                        maximum_ipc_message_bytes,
                                    },
                                ));
                                continue;
                            }
                            Err(start_error) => {
                                let _send_outcome = start_sender
                                    .send(Err(GenerationStartError::WorkerUnavailable));
                                contain_worker_failure(
                                    &mut worker_process,
                                    &health_snapshot,
                                    &mut active_generation,
                                    start_error,
                                )
                                .await;
                                is_ready = false;
                                continue;
                            }
                        }
                        active_generation = Some(ActiveGeneration {
                            _active_generation_permit: active_generation_permit,
                            generated_token_count: 0,
                            generation_started_at: None,
                            latest_generation_progress_token_count: 0,
                            max_output_tokens,
                            next_sequence_number: 0,
                            next_tool_call_index: 0,
                            request_started_at: Instant::now(),
                            prefill_elapsed_millis: 0,
                            last_mlx_peak_memory_bytes: None,
                            last_mlx_active_memory_bytes: None,
                            request_id,
                            stream_event_sender,
                        });
                        clear_active_request_progress(&health_snapshot);
                        publish_activity(&health_snapshot, WorkerActivity::PromptProcessing);
                        if start_sender.send(Ok(())).is_err() {
                            cancel_active_generation(
                                &mut worker_process,
                                &health_snapshot,
                                &mut active_generation,
                                cancellation_acknowledgement_timeout,
                            ).await;
                        }
                    }
                    WorkerLoopCommand::Shutdown { shutdown_sender } => {
                        publish_health(
                            &health_snapshot,
                            WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
                        );
                        fail_active_generation(
                            &mut active_generation,
                            ChatGenerationStreamErrorCode::WorkerUnavailable,
                        );
                        publish_activity(&health_snapshot, WorkerActivity::Idle);
                        clear_active_request_progress(&health_snapshot);
                        let shutdown_outcome = if worker_process.process_id().is_some() {
                            worker_process.close().await
                        } else {
                            Ok(WorkerTerminationOutcome::Graceful {
                                process_exit_successful: true,
                            })
                        };
                        let _send_outcome = shutdown_sender.send(shutdown_outcome);
                        return;
                    }
                    WorkerLoopCommand::RestartWorker {
                        worker_executable_path,
                        model_directories: replacement_model_directories,
                        max_output_tokens: replacement_max_output_tokens,
                        worker_startup_configuration,
                        restart_sender,
                    } => {
                        if active_generation.is_some() {
                            let _send_outcome = restart_sender.send(Err(
                                WorkerControlError::GenerationBusy,
                            ));
                            continue;
                        }

                        publish_health(
                            &health_snapshot,
                            WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading),
                        );
                        is_ready = false;
                        model_load_deadline = None;

                        if worker_process.process_id().is_some()
                            && let Err(worker_close_error) = worker_process.close().await
                        {
                            publish_health(
                                &health_snapshot,
                                WorkerHealthSnapshot::unavailable(
                                    WorkerHealthStatus::Unavailable,
                                ),
                            );
                            let _send_outcome = restart_sender.send(Err(worker_close_error));
                            continue;
                        }

                        let replacement_worker_launch_result = match worker_startup_configuration {
                            Some(worker_startup_configuration) => {
                                WorkerProcess::launch_with_startup_configuration(
                                    &worker_executable_path,
                                    worker_startup_configuration,
                                )
                                .await
                            }
                            None => WorkerProcess::launch(&worker_executable_path).await,
                        };
                        match replacement_worker_launch_result {
                            Ok(replacement_worker_process) => {
                                worker_process = replacement_worker_process;
                                model_directories = replacement_model_directories;
                                max_output_tokens = replacement_max_output_tokens;
                                model_load_deadline =
                                    Some(Instant::now() + model_load_timeout);
                                let _send_outcome = restart_sender.send(Ok(()));
                            }
                            Err(worker_launch_error) => {
                                publish_health(
                                    &health_snapshot,
                                    WorkerHealthSnapshot::unavailable(
                                        WorkerHealthStatus::Unavailable,
                                    ),
                                );
                                let _send_outcome =
                                    restart_sender.send(Err(worker_launch_error));
                            }
                        }
                    }
                    WorkerLoopCommand::UpdateMlxMemoryLimit {
                        effective_mlx_memory_ceiling_bytes,
                        update_sender,
                    } => {
                        if active_generation.is_some() {
                            pending_mlx_memory_ceiling_bytes =
                                Some(effective_mlx_memory_ceiling_bytes);
                            publish_pending_mlx_memory_ceiling(
                                &health_snapshot,
                                pending_mlx_memory_ceiling_bytes,
                            );
                            let _send_outcome = update_sender
                                .send(Ok(MlxMemoryLimitUpdateOutcome::Queued));
                            continue;
                        }
                        let update_outcome = apply_mlx_memory_limit(
                            &mut worker_process,
                            effective_mlx_memory_ceiling_bytes,
                            model_load_timeout,
                            &health_snapshot,
                            &mut is_ready,
                            &mut model_load_deadline,
                            &mut active_generation,
                            &mut performance_log,
                        )
                        .await;
                        let _send_outcome = update_sender.send(update_outcome);
                    }
                }
            }
            worker_event_outcome = worker_process.next_event(), if is_worker_running => {
                match worker_event_outcome {
                    Ok(Some(worker_event)) => {
                        if let Err(event_handling_error) = handle_worker_event(
                            worker_event,
                            &health_snapshot,
                            &mut is_ready,
                            &mut model_load_deadline,
                            &mut active_generation,
                            &mut performance_log,
                        ) {
                            if matches!(
                                &event_handling_error,
                                WorkerControlError::StreamBackpressure
                            ) {
                                tracing::warn!("HTTP stream stopped consuming output; cancelling request");
                                cancel_active_generation(
                                    &mut worker_process,
                                    &health_snapshot,
                                    &mut active_generation,
                                    cancellation_acknowledgement_timeout,
                                ).await;
                            } else {
                                contain_worker_failure(
                                    &mut worker_process,
                                    &health_snapshot,
                                    &mut active_generation,
                                    event_handling_error,
                                ).await;
                                is_ready = false;
                            }
                        }
                        if active_generation.is_none()
                            && let Some(pending_mlx_memory_ceiling_bytes) =
                                pending_mlx_memory_ceiling_bytes.take()
                        {
                            publish_pending_mlx_memory_ceiling(&health_snapshot, None);
                            if let Err(memory_limit_error) = apply_mlx_memory_limit(
                                &mut worker_process,
                                pending_mlx_memory_ceiling_bytes,
                                model_load_timeout,
                                &health_snapshot,
                                &mut is_ready,
                                &mut model_load_deadline,
                                &mut active_generation,
                                &mut performance_log,
                            )
                            .await
                            {
                                contain_mlx_memory_limit_failure(
                                    &mut worker_process,
                                    &health_snapshot,
                                    &mut active_generation,
                                    &mut is_ready,
                                    memory_limit_error,
                                ).await;
                            }
                        }
                    }
                    Ok(None) => {
                        contain_worker_failure(
                            &mut worker_process,
                            &health_snapshot,
                            &mut active_generation,
                            WorkerControlError::WorkerEventStreamClosed,
                        ).await;
                        is_ready = false;
                    }
                    Err(worker_event_error) => {
                        contain_worker_failure(
                            &mut worker_process,
                            &health_snapshot,
                            &mut active_generation,
                            worker_event_error,
                        ).await;
                        is_ready = false;
                    }
                }
            }
            () = wait_for_deadline(model_load_deadline), if is_worker_running && !is_ready => {
                contain_worker_failure(
                    &mut worker_process,
                    &health_snapshot,
                    &mut active_generation,
                    WorkerControlError::ModelLoadTimeout {
                        model_load_timeout_millis: model_load_timeout.as_millis(),
                    },
                ).await;
                model_load_deadline = None;
            }
            () = wait_for_stream_disconnect(stream_event_sender), if active_generation.is_some() => {
                cancel_active_generation(
                    &mut worker_process,
                    &health_snapshot,
                    &mut active_generation,
                    cancellation_acknowledgement_timeout,
                ).await;
            }
            _ = idle_memory_sampling_interval.tick(), if is_worker_running && is_ready && active_generation.is_none() && has_loaded_model => {
                if let Err(memory_sample_error) = worker_process.sample_mlx_memory().await {
                    tracing::warn!(error = %memory_sample_error, "could not request idle MLX memory telemetry");
                }
            }
        }
    }
}
