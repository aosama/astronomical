//! Stop-first worker lifecycle for live resident-to-streaming memory demotion.
//!
//! Lowering the ceiling below a loaded resident model cannot be performed in
//! place, and constructing the streaming replacement while the resident model
//! is still alive would briefly require both model runtimes in one process, so
//! the memory could never be released deterministically. This module therefore
//! enforces one invariant by construction: the replacement worker process is
//! launched only after the trusted worker has exited and been reaped.
//!
//! Ordinary configuration reloads keep their candidate-first transactional
//! replacement (`WorkerReplacement`); that overlap is exactly what a memory
//! demotion must avoid, so this owner deliberately trades a brief model-reload
//! interval for a hard process memory boundary.

use std::{collections::HashMap, sync::Arc, time::Duration};

use astronomical_ipc_protocol::{
    WorkerEvent, WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};
use tokio::time::{Instant, timeout};

use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, RuntimeModelPolicy, WorkerControlError,
    WorkerHealthSnapshot, WorkerHealthStatus, WorkerProcess, WorkerTerminationOutcome,
    worker::worker_containment::fail_active_generation,
    worker::worker_event_handler::handle_worker_event, worker::worker_health::publish_health,
    worker::worker_loop_types::ActiveWorkerRequest,
};

const MAXIMUM_DEFERRED_DEMOTION_PROCESS_EVENTS: usize = 64;

/// Inputs for one stop-first memory-demotion restart of the trusted worker.
pub(super) struct MemoryDemotionRestart {
    candidate_executable_path: std::path::PathBuf,
    candidate_model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
    candidate_startup_configuration: WorkerStartupConfiguration,
    expected_configuration_generation: String,
    acknowledgement_timeout: Duration,
}

impl MemoryDemotionRestart {
    pub(super) fn new(
        candidate_executable_path: std::path::PathBuf,
        candidate_model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
        candidate_startup_configuration: WorkerStartupConfiguration,
        expected_configuration_generation: String,
        acknowledgement_timeout: Duration,
    ) -> Self {
        Self {
            candidate_executable_path,
            candidate_model_policy_catalog,
            candidate_startup_configuration,
            expected_configuration_generation,
            acknowledgement_timeout,
        }
    }

    /// Stops the trusted worker, launches its replacement, and restores the
    /// previously selected model.
    ///
    /// The generation admission permit is held by the caller for the entire
    /// transition, so no request can start between shutdown and restored-model
    /// readiness.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute(
        self,
        trusted_worker_process: &mut WorkerProcess,
        trusted_model_policy_catalog: &mut Arc<HashMap<String, RuntimeModelPolicy>>,
        health_snapshot: &Arc<std::sync::RwLock<WorkerHealthSnapshot>>,
        is_ready: &mut bool,
        model_load_deadline: &mut Option<Instant>,
        active_generation: &mut Option<ActiveWorkerRequest>,
        performance_log: &mut GenerationPerformanceLog,
        completion_log: &mut CompletionAttributionLog,
    ) -> Result<WorkerRuntimeFeatureConfiguration, WorkerControlError> {
        let demotion_started_at = Instant::now();
        let previously_selected_model_id = health_snapshot
            .read()
            .ok()
            .and_then(|worker_health_snapshot| worker_health_snapshot.ready_model_id.clone());
        tracing::info!(
            expected_configuration_generation = %self.expected_configuration_generation,
            previously_selected_model_id = ?previously_selected_model_id,
            "starting stop-first memory-demotion worker replacement"
        );
        self.stop_trusted_worker(
            trusted_worker_process,
            health_snapshot,
            is_ready,
            model_load_deadline,
            active_generation,
        )
        .await?;
        let shutdown_elapsed_millis = demotion_started_at.elapsed().as_millis();

        // A failed launch must leave the same truthful unavailable state as a
        // rejected acknowledgement: the trusted worker is already gone, so
        // pretending a model is still loading would hide that nothing serves.
        let mut candidate_worker_process = match self.launch_candidate().await {
            Ok(candidate_worker_process) => candidate_worker_process,
            Err(candidate_launch_error) => {
                publish_health(
                    health_snapshot,
                    WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
                );
                tracing::error!(
                    error = %candidate_launch_error,
                    "memory-demotion replacement candidate could not launch"
                );
                return Err(candidate_launch_error);
            }
        };
        let launch_elapsed_millis = demotion_started_at.elapsed().as_millis();

        let candidate_acknowledgement = match timeout(
            self.acknowledgement_timeout,
            read_candidate_acknowledgement(
                &mut candidate_worker_process,
                &self.expected_configuration_generation,
                &self.candidate_model_policy_catalog,
            ),
        )
        .await
        {
            Ok(Ok(candidate_acknowledgement)) => candidate_acknowledgement,
            Ok(Err(candidate_error)) => {
                let contained_error =
                    close_failed_candidate(candidate_worker_process, candidate_error).await;
                publish_health(
                    health_snapshot,
                    WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
                );
                tracing::error!(
                    error = %contained_error,
                    "memory-demotion replacement rejected its candidate acknowledgement"
                );
                return Err(contained_error);
            }
            Err(_) => {
                let contained_error = close_failed_candidate(
                    candidate_worker_process,
                    WorkerControlError::CandidateAcknowledgementTimeout {
                        acknowledgement_timeout_millis: self.acknowledgement_timeout.as_millis(),
                    },
                )
                .await;
                publish_health(
                    health_snapshot,
                    WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
                );
                tracing::error!(
                    error = %contained_error,
                    "memory-demotion replacement timed out awaiting candidate acknowledgement"
                );
                return Err(contained_error);
            }
        };
        let handshake_elapsed_millis = demotion_started_at.elapsed().as_millis();

        *trusted_worker_process = candidate_worker_process;
        let candidate_model_policy_catalog = Arc::clone(&self.candidate_model_policy_catalog);
        *trusted_model_policy_catalog = Arc::clone(&candidate_model_policy_catalog);
        *is_ready = false;
        *model_load_deadline = None;
        publish_restored_candidate(
            candidate_acknowledgement,
            health_snapshot,
            is_ready,
            model_load_deadline,
            performance_log,
            completion_log,
        )?;
        let restored_runtime_configuration = health_snapshot
            .read()
            .ok()
            .and_then(|worker_health_snapshot| {
                worker_health_snapshot
                    .worker_runtime_feature_configuration
                    .clone()
            })
            .ok_or(WorkerControlError::WorkerProtocolViolation {
                description: "memory-demotion restart published no runtime configuration",
            })?;
        if let Some(previously_selected_model_id) = previously_selected_model_id {
            self.restore_selected_model(
                trusted_worker_process,
                previously_selected_model_id,
                candidate_model_policy_catalog,
                health_snapshot,
                is_ready,
                model_load_deadline,
                active_generation,
                performance_log,
                completion_log,
                &restored_runtime_configuration,
            )
            .await?;
        }
        tracing::info!(
            shutdown_elapsed_millis,
            launch_elapsed_millis,
            handshake_elapsed_millis,
            model_restoration_elapsed_millis =
                demotion_started_at.elapsed().as_millis() - handshake_elapsed_millis,
            total_elapsed_millis = demotion_started_at.elapsed().as_millis(),
            "committed stop-first memory-demotion worker replacement"
        );
        Ok(restored_runtime_configuration)
    }

    /// Closes and reaps the trusted worker before any candidate can exist.
    #[allow(clippy::too_many_arguments)]
    async fn stop_trusted_worker(
        &self,
        trusted_worker_process: &mut WorkerProcess,
        health_snapshot: &Arc<std::sync::RwLock<WorkerHealthSnapshot>>,
        is_ready: &mut bool,
        model_load_deadline: &mut Option<Instant>,
        active_generation: &mut Option<ActiveWorkerRequest>,
    ) -> Result<(), WorkerControlError> {
        publish_health(
            health_snapshot,
            WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading),
        );
        fail_active_generation(
            active_generation,
            crate::ChatGenerationStreamErrorCode::WorkerUnavailable,
        );
        *is_ready = false;
        *model_load_deadline = None;
        let trusted_close_result = if trusted_worker_process.process_id().is_some() {
            trusted_worker_process.close().await
        } else {
            Ok(WorkerTerminationOutcome::Graceful {
                process_exit_successful: true,
            })
        };
        if let Err(trusted_close_error) = trusted_close_result {
            let trusted_containment_error = trusted_worker_process.force_terminate().await.err();
            let contained_error = match trusted_containment_error {
                Some(containment_error) => WorkerControlError::OperationAndCleanupFailed {
                    operation: Box::new(trusted_close_error),
                    cleanup: Box::new(containment_error),
                },
                None => trusted_close_error,
            };
            publish_health(
                health_snapshot,
                WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
            );
            tracing::error!(
                error = %contained_error,
                "memory demotion could not close and reap the trusted worker"
            );
            return Err(contained_error);
        }
        tracing::info!(
            "closed and reaped the trusted worker before the memory-demotion candidate launch"
        );
        Ok(())
    }

    async fn launch_candidate(&self) -> Result<WorkerProcess, WorkerControlError> {
        WorkerProcess::launch_with_startup_configuration(
            &self.candidate_executable_path,
            self.candidate_startup_configuration.clone(),
        )
        .await
    }

    /// Reloads the previously selected model through the existing swap path.
    ///
    /// The policy identity comes from the trusted catalog, never from user
    /// input attached to the memory request. Readiness published by
    /// `wait_for_model_swap` is the genuine loaded-model acknowledgement.
    #[allow(clippy::too_many_arguments)]
    async fn restore_selected_model(
        &self,
        replacement_worker_process: &mut WorkerProcess,
        previously_selected_model_id: String,
        candidate_model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
        health_snapshot: &Arc<std::sync::RwLock<WorkerHealthSnapshot>>,
        is_ready: &mut bool,
        model_load_deadline: &mut Option<Instant>,
        active_generation: &mut Option<ActiveWorkerRequest>,
        performance_log: &mut GenerationPerformanceLog,
        completion_log: &mut CompletionAttributionLog,
        restored_runtime_configuration: &WorkerRuntimeFeatureConfiguration,
    ) -> Result<(), WorkerControlError> {
        let Some(model_policy) = candidate_model_policy_catalog.get(&previously_selected_model_id)
        else {
            return Err(WorkerControlError::WorkerProtocolViolation {
                description: "memory demotion could not restore an unmapped selected model",
            });
        };
        replacement_worker_process
            .swap_model(
                model_policy.model_directory.to_string_lossy().into_owned(),
                model_policy.worker_model_configuration.clone(),
            )
            .await?;
        let swap_wait_outcome = timeout(
            self.acknowledgement_timeout,
            crate::worker::worker_model_swap::wait_for_model_swap(
                replacement_worker_process,
                health_snapshot,
                is_ready,
                model_load_deadline,
                active_generation,
                performance_log,
                completion_log,
                Some(
                    restored_runtime_configuration
                        .configuration_generation
                        .as_str(),
                ),
                &model_policy
                    .worker_model_configuration
                    .runtime_configuration(),
            ),
        )
        .await
        .map_err(|_| WorkerControlError::ModelLoadTimeout {
            model_load_timeout_millis: self.acknowledgement_timeout.as_millis(),
        })
        .and_then(|swap_wait_outcome| swap_wait_outcome)?;
        match swap_wait_outcome {
            crate::worker::worker_model_swap::ModelSwapWaitOutcome::Loaded => Ok(()),
            crate::worker::worker_model_swap::ModelSwapWaitOutcome::Rejected {
                model_load_failure_reason,
            } => Err(WorkerControlError::WorkerProtocolViolation {
                description: "memory demotion restored-worker model swap was rejected",
            })
            .map_err(|restoration_error| {
                tracing::error!(
                    model_load_failure_reason,
                    "memory demotion could not restore the previously selected model"
                );
                restoration_error
            }),
        }
    }
}

struct DemotionCandidateAcknowledgement {
    readiness_event: WorkerEvent,
    runtime_feature_configuration: WorkerRuntimeFeatureConfiguration,
    deferred_process_events: Vec<WorkerEvent>,
}

async fn read_candidate_acknowledgement(
    candidate_worker_process: &mut WorkerProcess,
    expected_configuration_generation: &str,
    candidate_model_policy_catalog: &HashMap<String, RuntimeModelPolicy>,
) -> Result<DemotionCandidateAcknowledgement, WorkerControlError> {
    let mut readiness_event = None;
    let mut runtime_feature_configuration = None;
    let mut deferred_process_events = Vec::new();
    loop {
        let candidate_event = candidate_worker_process
            .next_event()
            .await?
            .ok_or(WorkerControlError::WorkerEventStreamClosed)?;
        match candidate_event {
            event @ (WorkerEvent::Idle { .. } | WorkerEvent::Ready { .. }) => {
                if readiness_event.replace(event).is_some() {
                    return Err(WorkerControlError::CandidateProtocolViolation {
                        description: "demotion candidate emitted duplicate initial readiness",
                    });
                }
            }
            WorkerEvent::RuntimeFeatureConfigurationApplied {
                worker_runtime_feature_configuration,
            } => {
                if runtime_feature_configuration.is_some() {
                    return Err(WorkerControlError::CandidateProtocolViolation {
                        description: "demotion candidate emitted duplicate runtime configuration",
                    });
                }
                if worker_runtime_feature_configuration.configuration_generation
                    != expected_configuration_generation
                {
                    return Err(WorkerControlError::CandidateConfigurationGenerationMismatch);
                }
                runtime_feature_configuration = Some(worker_runtime_feature_configuration);
            }
            event @ (WorkerEvent::MlxMemorySample { .. }
            | WorkerEvent::ExpertMemoryModeChanged { .. }
            | WorkerEvent::PersistentPromptCacheStats { .. }) => {
                if deferred_process_events.len() >= MAXIMUM_DEFERRED_DEMOTION_PROCESS_EVENTS {
                    return Err(WorkerControlError::CandidateProtocolViolation {
                        description: "demotion candidate emitted too many process events before acknowledgement",
                    });
                }
                deferred_process_events.push(event);
            }
            unexpected_event => {
                return Err(WorkerControlError::UnexpectedCandidateEvent {
                    unexpected_worker_event_summary: unexpected_event.diagnostic_summary(),
                });
            }
        }
        if let (Some(readiness_event), Some(runtime_feature_configuration)) = (
            readiness_event.clone(),
            runtime_feature_configuration.clone(),
        ) {
            validate_demotion_candidate_model_binding(
                &readiness_event,
                &runtime_feature_configuration,
                candidate_model_policy_catalog,
            )?;
            return Ok(DemotionCandidateAcknowledgement {
                readiness_event,
                runtime_feature_configuration,
                deferred_process_events,
            });
        }
    }
}

fn validate_demotion_candidate_model_binding(
    readiness_event: &WorkerEvent,
    runtime_configuration: &WorkerRuntimeFeatureConfiguration,
    candidate_model_policy_catalog: &HashMap<String, RuntimeModelPolicy>,
) -> Result<(), WorkerControlError> {
    match readiness_event {
        WorkerEvent::Idle { .. } if runtime_configuration.loaded_model.is_none() => Ok(()),
        WorkerEvent::Ready { model_id, .. } => {
            let Some(loaded_model) = runtime_configuration.loaded_model.as_ref() else {
                return Err(WorkerControlError::CandidateProtocolViolation {
                    description: "ready demotion candidate did not acknowledge its loaded model policy",
                });
            };
            let Some(candidate_policy) = candidate_model_policy_catalog.get(model_id) else {
                return Err(WorkerControlError::CandidateProtocolViolation {
                    description: "ready demotion candidate model is absent from the candidate catalog",
                });
            };
            if loaded_model
                != &candidate_policy
                    .worker_model_configuration
                    .runtime_configuration()
            {
                return Err(WorkerControlError::CandidateProtocolViolation {
                    description: "ready demotion candidate model disagrees with its acknowledged policy",
                });
            }
            Ok(())
        }
        WorkerEvent::Idle { .. } => Err(WorkerControlError::CandidateProtocolViolation {
            description: "idle demotion candidate acknowledged an unexpected loaded model",
        }),
        _ => Err(WorkerControlError::CandidateProtocolViolation {
            description: "demotion candidate readiness event is unsupported",
        }),
    }
}

fn publish_restored_candidate(
    candidate_acknowledgement: DemotionCandidateAcknowledgement,
    health_snapshot: &Arc<std::sync::RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<(), WorkerControlError> {
    handle_worker_event(
        candidate_acknowledgement.readiness_event,
        health_snapshot,
        is_ready,
        model_load_deadline,
        &mut None,
        performance_log,
        completion_log,
    )?;
    handle_worker_event(
        WorkerEvent::RuntimeFeatureConfigurationApplied {
            worker_runtime_feature_configuration: candidate_acknowledgement
                .runtime_feature_configuration,
        },
        health_snapshot,
        is_ready,
        model_load_deadline,
        &mut None,
        performance_log,
        completion_log,
    )?;
    for deferred_process_event in candidate_acknowledgement.deferred_process_events {
        handle_worker_event(
            deferred_process_event,
            health_snapshot,
            is_ready,
            model_load_deadline,
            &mut None,
            performance_log,
            completion_log,
        )?;
    }
    Ok(())
}

async fn close_failed_candidate(
    mut candidate_worker_process: WorkerProcess,
    candidate_error: WorkerControlError,
) -> WorkerControlError {
    match candidate_worker_process.close().await {
        Ok(_) => candidate_error,
        Err(cleanup_error) => WorkerControlError::OperationAndCleanupFailed {
            operation: Box::new(candidate_error),
            cleanup: Box::new(cleanup_error),
        },
    }
}
