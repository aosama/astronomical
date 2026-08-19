//! The main worker event loop: receives commands and events, coordinates
//! generation lifecycle, model swaps, memory limits, and idle telemetry.
//!
//! The loop owns one `WorkerProcess` and one optional `ActiveGeneration`.
//! Commands arrive through `WorkerLoopCommand`; events arrive through the
//! worker's IPC stream. Generation-scoped events must match the active
//! request or are treated as protocol violations.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::sync::{Semaphore, mpsc};
use tokio::time::{Instant, MissedTickBehavior, interval};

use crate::worker_cache_clear::{
    apply_pending_prompt_cache_clear_if_idle, handle_prompt_cache_clear_command,
};
use crate::worker_generate::handle_generate_command;
use crate::worker_memory_limit::{
    MlxMemoryLimitUpdateOutcome, apply_mlx_memory_limit, contain_mlx_memory_limit_failure,
};
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
    active_generation_permits: Arc<Semaphore>,
    generation_queue_permits: Arc<Semaphore>,
) {
    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading),
    );
    let mut active_generation: Option<ActiveGeneration> = None;
    let mut is_ready = false;
    let mut model_load_deadline = Some(Instant::now() + model_load_timeout);
    let mut idle_control_interval = interval(Duration::from_secs(1));
    idle_control_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut pending_mlx_memory_ceiling_bytes = None;
    let mut pending_prompt_cache_clear: Option<crate::PendingPromptCacheClear> = None;

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
        apply_pending_prompt_cache_clear_if_idle(
            &mut pending_prompt_cache_clear,
            &mut worker_process,
            &health_snapshot,
            &mut active_generation,
            &mut is_ready,
            &mut model_load_deadline,
            &mut performance_log,
            &active_generation_permits,
            &generation_queue_permits,
        )
        .await;
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
                        if let Err(control_error) = handle_generate_command(
                            &mut worker_process,
                            active_generation_permit,
                            generation_command,
                            start_sender,
                            stream_event_sender,
                            &health_snapshot,
                            &mut is_ready,
                            &mut model_load_deadline,
                            &mut active_generation,
                            &mut performance_log,
                            &model_directories,
                            max_output_tokens,
                            model_load_timeout,
                            cancellation_acknowledgement_timeout,
                        )
                        .await
                        {
                            contain_worker_failure(
                                &mut worker_process,
                                &health_snapshot,
                                &mut active_generation,
                                control_error,
                            )
                            .await;
                            is_ready = false;
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
                    WorkerLoopCommand::ClearPromptCache {
                        model_id,
                        clear_sender,
                    } => {
                        handle_prompt_cache_clear_command(
                            model_id,
                            clear_sender,
                            &mut pending_prompt_cache_clear,
                            &mut worker_process,
                            &health_snapshot,
                            &mut active_generation,
                            &mut is_ready,
                            &mut model_load_deadline,
                            &mut performance_log,
                            &active_generation_permits,
                            &generation_queue_permits,
                        )
                        .await;
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
            _ = idle_control_interval.tick(), if is_worker_running
                && is_ready
                && active_generation.is_none()
                && (has_loaded_model || pending_prompt_cache_clear.is_some()) => {
                if has_loaded_model
                    && let Err(memory_sample_error) = worker_process.sample_mlx_memory().await
                {
                    tracing::warn!(error = %memory_sample_error, "could not request idle MLX memory telemetry");
                }
            }
        }
    }
}
