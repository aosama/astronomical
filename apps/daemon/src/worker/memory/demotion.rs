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

use crate::worker::loop_types::WorkerLoopState;
use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, RuntimeModelPolicy, WorkerControlError,
    WorkerHealthSnapshot, WorkerHealthStatus, WorkerProcess, WorkerTerminationOutcome,
    worker::event_handler::handle_worker_event, worker::health::publish_health,
    worker::loop_types::ActiveWorkerRequest, worker::process::containment::fail_active_generation,
};

const MAXIMUM_DEFERRED_DEMOTION_PROCESS_EVENTS: usize = 64;

/// Inputs for one stop-first memory-demotion restart of the trusted worker.
pub(in crate::worker) struct MemoryDemotionRestart {
    candidate_executable_path: std::path::PathBuf,
    candidate_model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
    candidate_startup_configuration: WorkerStartupConfiguration,
    expected_configuration_generation: String,
    acknowledgement_timeout: Duration,
}

impl MemoryDemotionRestart {
    pub(in crate::worker) fn new(
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
    pub(in crate::worker) async fn execute(
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
        let candidate_worker_process = self
            .launch_candidate_or_publish_unavailable(health_snapshot)
            .await?;
        let launch_elapsed_millis = demotion_started_at.elapsed().as_millis();
        let acknowledged_candidate = self
            .handshake_candidate_or_publish_unavailable(candidate_worker_process, health_snapshot)
            .await?;
        let handshake_elapsed_millis = demotion_started_at.elapsed().as_millis();
        let candidate_model_policy_catalog = Arc::clone(&self.candidate_model_policy_catalog);
        let mut replacement_state = DemotionReplacementState {
            trusted_worker_process,
            trusted_model_policy_catalog,
            health_snapshot,
            is_ready,
            model_load_deadline,
            active_generation,
            performance_log,
            completion_log,
        };
        let restored_runtime_configuration = commit_candidate_replacement(
            &mut replacement_state,
            acknowledged_candidate.candidate_worker_process,
            acknowledged_candidate.acknowledgement,
            &candidate_model_policy_catalog,
        )?;
        if let Some(previously_selected_model_id) = previously_selected_model_id {
            self.restore_selected_model(
                &mut replacement_state,
                previously_selected_model_id,
                candidate_model_policy_catalog,
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

    /// Launches the candidate, or leaves the same truthful unavailable state
    /// as a rejected acknowledgement: the trusted worker is already gone, so
    /// pretending a model is still loading would hide that nothing serves.
    async fn launch_candidate_or_publish_unavailable(
        &self,
        health_snapshot: &Arc<std::sync::RwLock<WorkerHealthSnapshot>>,
    ) -> Result<WorkerProcess, WorkerControlError> {
        match self.launch_candidate().await {
            Ok(candidate_worker_process) => Ok(candidate_worker_process),
            Err(candidate_launch_error) => {
                publish_health(
                    health_snapshot,
                    WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
                );
                tracing::error!(
                    error = %candidate_launch_error,
                    "memory-demotion replacement candidate could not launch"
                );
                Err(candidate_launch_error)
            }
        }
    }

    /// Performs the candidate acknowledgement handshake. Any failure is
    /// contained by closing and reaping the candidate and republishing the
    /// truthful unavailable state; on success the still-open candidate process
    /// is returned with its acknowledgement, and it becomes the trusted worker
    /// only at commit time.
    async fn handshake_candidate_or_publish_unavailable(
        &self,
        mut candidate_worker_process: WorkerProcess,
        health_snapshot: &Arc<std::sync::RwLock<WorkerHealthSnapshot>>,
    ) -> Result<AcknowledgedCandidate, WorkerControlError> {
        let handshake_outcome = timeout(
            self.acknowledgement_timeout,
            read_candidate_acknowledgement(
                &mut candidate_worker_process,
                &self.expected_configuration_generation,
                &self.candidate_model_policy_catalog,
            ),
        )
        .await;
        let candidate_acknowledgement = match handshake_outcome {
            Ok(Ok(candidate_acknowledgement)) => candidate_acknowledgement,
            Ok(Err(candidate_error)) => {
                return Err(contain_failed_candidate_and_publish_unavailable(
                    candidate_worker_process,
                    candidate_error,
                    "memory-demotion replacement rejected its candidate acknowledgement",
                    health_snapshot,
                )
                .await);
            }
            Err(_) => {
                return Err(contain_failed_candidate_and_publish_unavailable(
                    candidate_worker_process,
                    WorkerControlError::CandidateAcknowledgementTimeout {
                        acknowledgement_timeout_millis: self.acknowledgement_timeout.as_millis(),
                    },
                    "memory-demotion replacement timed out awaiting candidate acknowledgement",
                    health_snapshot,
                )
                .await);
            }
        };
        Ok(AcknowledgedCandidate {
            candidate_worker_process,
            acknowledgement: candidate_acknowledgement,
        })
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
    async fn restore_selected_model(
        &self,
        replacement_state: &mut DemotionReplacementState<'_>,
        previously_selected_model_id: String,
        candidate_model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
        restored_runtime_configuration: &WorkerRuntimeFeatureConfiguration,
    ) -> Result<(), WorkerControlError> {
        let Some(model_policy) = candidate_model_policy_catalog.get(&previously_selected_model_id)
        else {
            return Err(WorkerControlError::WorkerProtocolViolation {
                description: "memory demotion could not restore an unmapped selected model",
            });
        };
        replacement_state
            .trusted_worker_process
            .swap_model(
                model_policy.model_directory.to_string_lossy().into_owned(),
                model_policy.worker_model_configuration.clone(),
            )
            .await?;
        let swap_wait_outcome = timeout(
            self.acknowledgement_timeout,
            crate::worker::process::model_swap::wait_for_model_swap(
                &mut WorkerLoopState {
                    worker_process: &mut *replacement_state.trusted_worker_process,
                    health_snapshot: replacement_state.health_snapshot,
                    is_ready: &mut *replacement_state.is_ready,
                    model_load_deadline: &mut *replacement_state.model_load_deadline,
                    active_request: &mut *replacement_state.active_generation,
                    performance_log: &mut *replacement_state.performance_log,
                    completion_log: &mut *replacement_state.completion_log,
                    // The restore path waits on the acknowledgement budget,
                    // not the initial-launch load budget.
                    model_load_timeout: self.acknowledgement_timeout,
                },
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
            crate::worker::process::model_swap::ModelSwapWaitOutcome::Loaded => Ok(()),
            crate::worker::process::model_swap::ModelSwapWaitOutcome::Rejected {
                model_load_failure_reason,
            } => Err(WorkerControlError::WorkerProtocolViolation {
                description: "memory demotion restored-worker model swap was rejected",
            })
            .inspect_err(|_| {
                tracing::error!(
                    model_load_failure_reason,
                    "memory demotion could not restore the previously selected model"
                );
            }),
        }
    }
}

/// The loop-state cluster one memory-demotion restart installs and restores:
/// the process being replaced, the catalog being swapped, and the readiness,
/// deadline, active-request, and telemetry state the commit and model
/// restoration phases mutate.
struct DemotionReplacementState<'a> {
    trusted_worker_process: &'a mut WorkerProcess,
    trusted_model_policy_catalog: &'a mut Arc<HashMap<String, RuntimeModelPolicy>>,
    health_snapshot: &'a Arc<std::sync::RwLock<WorkerHealthSnapshot>>,
    is_ready: &'a mut bool,
    model_load_deadline: &'a mut Option<Instant>,
    active_generation: &'a mut Option<ActiveWorkerRequest>,
    performance_log: &'a mut GenerationPerformanceLog,
    completion_log: &'a mut CompletionAttributionLog,
}

/// A candidate that completed its acknowledgement handshake with its process
/// still open, ready to be installed as the trusted worker.
struct AcknowledgedCandidate {
    candidate_worker_process: WorkerProcess,
    acknowledgement: DemotionCandidateAcknowledgement,
}

/// Installs the acknowledged candidate as the trusted worker and republishes
/// its readiness through the ordinary event path so daemon-visible state is
/// exactly what a fresh worker start would publish. Returns the runtime
/// configuration the candidate acknowledged, which the model restoration and
/// the restart answer both need.
fn commit_candidate_replacement(
    replacement_state: &mut DemotionReplacementState<'_>,
    candidate_worker_process: WorkerProcess,
    candidate_acknowledgement: DemotionCandidateAcknowledgement,
    candidate_model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
) -> Result<WorkerRuntimeFeatureConfiguration, WorkerControlError> {
    *replacement_state.trusted_worker_process = candidate_worker_process;
    *replacement_state.trusted_model_policy_catalog = Arc::clone(candidate_model_policy_catalog);
    *replacement_state.is_ready = false;
    *replacement_state.model_load_deadline = None;
    publish_restored_candidate(
        candidate_acknowledgement,
        replacement_state.health_snapshot,
        replacement_state.is_ready,
        replacement_state.model_load_deadline,
        replacement_state.performance_log,
        replacement_state.completion_log,
    )?;
    replacement_state
        .health_snapshot
        .read()
        .ok()
        .and_then(|worker_health_snapshot| {
            worker_health_snapshot
                .worker_runtime_feature_configuration
                .clone()
        })
        .ok_or(WorkerControlError::WorkerProtocolViolation {
            description: "memory-demotion restart published no runtime configuration",
        })
}

/// Closes and reaps a failed candidate, republishes the truthful unavailable
/// state, and logs the failure with the phase-specific description; returns
/// the contained error for the caller to propagate.
async fn contain_failed_candidate_and_publish_unavailable(
    candidate_worker_process: WorkerProcess,
    candidate_error: WorkerControlError,
    failure_description: &'static str,
    health_snapshot: &Arc<std::sync::RwLock<WorkerHealthSnapshot>>,
) -> WorkerControlError {
    let contained_error = close_failed_candidate(candidate_worker_process, candidate_error).await;
    publish_health(
        health_snapshot,
        WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
    );
    tracing::error!(error = %contained_error, failure_description);
    contained_error
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
