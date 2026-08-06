use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::WorkerEvent;
use tokio::time::Instant;

use crate::worker_event_handler::{handle_worker_event, protocol_violation};
use crate::worker_health::publish_health;
use crate::worker_loop_types::ActiveGeneration;
use crate::{GenerationPerformanceLog, WorkerControlError, WorkerHealthSnapshot, WorkerProcess};

/// Result of waiting for a worker model-swap acknowledgement.
pub(super) enum ModelSwapWaitOutcome {
    Loaded,
    Rejected { model_load_failure_reason: String },
}

/// Drains worker events until the model swap succeeds or is rejected.
pub(super) async fn wait_for_model_swap(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_generation: &mut Option<ActiveGeneration>,
    performance_log: &mut GenerationPerformanceLog,
) -> Result<ModelSwapWaitOutcome, WorkerControlError> {
    loop {
        let worker_event_outcome = worker_process.next_event().await;
        match worker_event_outcome {
            Ok(Some(worker_event)) => match &worker_event {
                WorkerEvent::ModelSwapped { .. } => {
                    handle_worker_event(
                        worker_event,
                        health_snapshot,
                        is_ready,
                        model_load_deadline,
                        active_generation,
                        performance_log,
                    )?;
                    return Ok(ModelSwapWaitOutcome::Loaded);
                }
                WorkerEvent::ModelSwapFailed {
                    loaded_model_remains_ready,
                    model_load_failure_reason,
                } => {
                    if !loaded_model_remains_ready {
                        let mlx_memory_ceilings = health_snapshot
                            .read()
                            .map(|worker_health_snapshot| {
                                (
                                    worker_health_snapshot.machine_mlx_memory_ceiling_bytes,
                                    worker_health_snapshot.mlx_memory_ceiling_bytes,
                                    worker_health_snapshot.minimum_mlx_memory_ceiling_bytes,
                                )
                            })
                            .unwrap_or((0, 0, 1));
                        publish_health(
                            health_snapshot,
                            WorkerHealthSnapshot::ready_without_model_with_memory_limits(
                                mlx_memory_ceilings.0,
                                mlx_memory_ceilings.1,
                                mlx_memory_ceilings.2,
                            ),
                        );
                    }
                    tracing::warn!(
                        loaded_model_remains_ready,
                        model_load_failure_reason,
                        "worker rejected the requested model while remaining responsive"
                    );
                    return Ok(ModelSwapWaitOutcome::Rejected {
                        model_load_failure_reason: model_load_failure_reason.clone(),
                    });
                }
                WorkerEvent::PersistentPromptCacheStats { .. } => {}
                _ => {
                    return Err(protocol_violation(
                        "unexpected worker event during model swap",
                    ));
                }
            },
            Ok(None) => return Err(WorkerControlError::WorkerEventStreamClosed),
            Err(worker_event_error) => return Err(worker_event_error),
        }
    }
}
