//! Validates and commits one candidate worker without exposing partial startup state.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use astronomical_ipc_protocol::{
    WorkerEvent, WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};
use tokio::time::{Instant, timeout};

use crate::{
    GenerationPerformanceLog, RuntimeModelPolicy, WorkerControlError, WorkerHealthSnapshot,
    WorkerHealthStatus, WorkerProcess, worker_event_handler::handle_worker_event,
    worker_health::publish_health, worker_loop_types::ActiveGeneration,
};

const MAXIMUM_DEFERRED_CANDIDATE_PROCESS_EVENTS: usize = 64;

/// Owns the candidate settings and the transaction that replaces one trusted worker.
pub(crate) struct WorkerReplacement {
    candidate_executable_path: PathBuf,
    candidate_model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
    candidate_startup_configuration: Option<WorkerStartupConfiguration>,
    expected_configuration_generation: String,
    acknowledgement_timeout: Duration,
}

impl WorkerReplacement {
    pub(crate) fn new(
        candidate_executable_path: PathBuf,
        candidate_model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
        candidate_startup_configuration: Option<WorkerStartupConfiguration>,
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn execute(
        self,
        trusted_worker_process: &mut WorkerProcess,
        trusted_model_policy_catalog: &mut Arc<HashMap<String, RuntimeModelPolicy>>,
        health_snapshot: &Arc<std::sync::RwLock<WorkerHealthSnapshot>>,
        is_ready: &mut bool,
        model_load_deadline: &mut Option<Instant>,
        active_generation: &mut Option<ActiveGeneration>,
        performance_log: &mut GenerationPerformanceLog,
    ) -> Result<WorkerRuntimeFeatureConfiguration, WorkerControlError> {
        let replacement_started_at = Instant::now();
        tracing::info!(
            expected_configuration_generation = %self.expected_configuration_generation,
            "starting transactional worker replacement"
        );
        let candidate_launch_result = match self.candidate_startup_configuration {
            Some(candidate_startup_configuration) => {
                WorkerProcess::launch_with_startup_configuration(
                    &self.candidate_executable_path,
                    candidate_startup_configuration,
                )
                .await
            }
            None => WorkerProcess::launch(&self.candidate_executable_path).await,
        };
        let mut candidate_worker_process = match candidate_launch_result {
            Ok(candidate_worker_process) => candidate_worker_process,
            Err(candidate_launch_error) => {
                tracing::warn!(
                    replacement_elapsed_millis = replacement_started_at.elapsed().as_millis(),
                    error = %candidate_launch_error,
                    "transactional worker replacement failed during candidate launch"
                );
                return Err(candidate_launch_error);
            }
        };
        let acknowledgement_started_at = Instant::now();
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
                    reject_candidate(candidate_worker_process, candidate_error).await;
                tracing::warn!(
                    replacement_elapsed_millis = replacement_started_at.elapsed().as_millis(),
                    error = %contained_error,
                    "transactional worker replacement rejected candidate acknowledgement"
                );
                return Err(contained_error);
            }
            Err(_) => {
                let contained_error = reject_candidate(
                    candidate_worker_process,
                    WorkerControlError::CandidateAcknowledgementTimeout {
                        acknowledgement_timeout_millis: self.acknowledgement_timeout.as_millis(),
                    },
                )
                .await;
                tracing::warn!(
                    replacement_elapsed_millis = replacement_started_at.elapsed().as_millis(),
                    error = %contained_error,
                    "transactional worker replacement timed out"
                );
                return Err(contained_error);
            }
        };
        tracing::info!(
            candidate_acknowledgement_elapsed_millis =
                acknowledgement_started_at.elapsed().as_millis(),
            "candidate worker acknowledged readiness and runtime configuration"
        );

        let trusted_close_started_at = Instant::now();
        if let Err(trusted_close_error) = trusted_worker_process.close().await {
            let trusted_containment_error = trusted_worker_process.force_terminate().await.err();
            let candidate_cleanup_error = match candidate_worker_process.close().await {
                Ok(_) => None,
                Err(candidate_cleanup_error) => {
                    continue_candidate_reaping(candidate_worker_process);
                    Some(candidate_cleanup_error)
                }
            };
            *is_ready = false;
            // A failed force-termination keeps the trusted child under loop ownership;
            // the deadline path retries containment and replacement commands stay blocked.
            *model_load_deadline = trusted_containment_error
                .as_ref()
                .map(|_| Instant::now() + self.acknowledgement_timeout);
            publish_health(
                health_snapshot,
                WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
            );
            let contained_error = combine_replacement_cleanup_errors(
                trusted_close_error,
                trusted_containment_error,
                candidate_cleanup_error,
            );
            tracing::error!(
                replacement_elapsed_millis = replacement_started_at.elapsed().as_millis(),
                error = %contained_error,
                "transactional worker replacement could not contain the trusted worker"
            );
            return Err(contained_error);
        }
        tracing::info!(
            trusted_worker_close_elapsed_millis = trusted_close_started_at.elapsed().as_millis(),
            "closed trusted worker before replacement commit"
        );

        *trusted_worker_process = candidate_worker_process;
        *trusted_model_policy_catalog = self.candidate_model_policy_catalog;
        *is_ready = false;
        *model_load_deadline = None;
        let publish_result = publish_candidate_acknowledgement(
            candidate_acknowledgement.clone(),
            health_snapshot,
            is_ready,
            model_load_deadline,
            active_generation,
            performance_log,
        );
        if let Err(publish_error) = publish_result {
            let cleanup_error = trusted_worker_process.force_terminate().await.err();
            *is_ready = false;
            publish_health(
                health_snapshot,
                WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
            );
            let contained_error = match cleanup_error {
                Some(cleanup_error) => WorkerControlError::OperationAndCleanupFailed {
                    operation: Box::new(publish_error),
                    cleanup: Box::new(cleanup_error),
                },
                None => publish_error,
            };
            tracing::error!(
                replacement_elapsed_millis = replacement_started_at.elapsed().as_millis(),
                error = %contained_error,
                "transactional worker replacement could not publish committed state"
            );
            return Err(contained_error);
        }
        tracing::info!(
            replacement_elapsed_millis = replacement_started_at.elapsed().as_millis(),
            "committed transactional worker replacement"
        );
        Ok(candidate_acknowledgement.runtime_feature_configuration)
    }
}

#[derive(Clone)]
struct CandidateAcknowledgement {
    readiness_event: WorkerEvent,
    runtime_feature_configuration: WorkerRuntimeFeatureConfiguration,
    deferred_process_events: Vec<WorkerEvent>,
}

async fn read_candidate_acknowledgement(
    candidate_worker_process: &mut WorkerProcess,
    expected_configuration_generation: &str,
    candidate_model_policy_catalog: &HashMap<String, RuntimeModelPolicy>,
) -> Result<CandidateAcknowledgement, WorkerControlError> {
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
                        description: "candidate emitted duplicate initial readiness",
                    });
                }
            }
            WorkerEvent::RuntimeFeatureConfigurationApplied {
                worker_runtime_feature_configuration,
            } => {
                if runtime_feature_configuration.is_some() {
                    return Err(WorkerControlError::CandidateProtocolViolation {
                        description: "candidate emitted duplicate runtime configuration",
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
                if deferred_process_events.len() >= MAXIMUM_DEFERRED_CANDIDATE_PROCESS_EVENTS {
                    return Err(WorkerControlError::CandidateProtocolViolation {
                        description: "candidate emitted too many process events before acknowledgement",
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
            validate_candidate_model_binding(
                &readiness_event,
                &runtime_feature_configuration,
                candidate_model_policy_catalog,
            )?;
            return Ok(CandidateAcknowledgement {
                readiness_event,
                runtime_feature_configuration,
                deferred_process_events,
            });
        }
    }
}

fn validate_candidate_model_binding(
    readiness_event: &WorkerEvent,
    runtime_configuration: &WorkerRuntimeFeatureConfiguration,
    candidate_model_policy_catalog: &HashMap<String, RuntimeModelPolicy>,
) -> Result<(), WorkerControlError> {
    match readiness_event {
        WorkerEvent::Idle { .. } if runtime_configuration.loaded_model.is_none() => Ok(()),
        WorkerEvent::Ready { model_id, .. } => {
            let Some(loaded_model) = runtime_configuration.loaded_model.as_ref() else {
                return Err(WorkerControlError::CandidateProtocolViolation {
                    description: "ready candidate did not acknowledge its loaded model policy",
                });
            };
            let Some(candidate_policy) = candidate_model_policy_catalog.get(model_id) else {
                return Err(WorkerControlError::CandidateProtocolViolation {
                    description: "ready candidate model is absent from the candidate catalog",
                });
            };
            if loaded_model
                != &candidate_policy
                    .worker_model_configuration
                    .runtime_configuration()
            {
                return Err(WorkerControlError::CandidateProtocolViolation {
                    description: "ready candidate model disagrees with its acknowledged policy",
                });
            }
            Ok(())
        }
        WorkerEvent::Idle { .. } => Err(WorkerControlError::CandidateProtocolViolation {
            description: "idle candidate acknowledged an unexpected loaded model",
        }),
        _ => Err(WorkerControlError::CandidateProtocolViolation {
            description: "candidate readiness event is unsupported",
        }),
    }
}

fn publish_candidate_acknowledgement(
    candidate_acknowledgement: CandidateAcknowledgement,
    health_snapshot: &Arc<std::sync::RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_generation: &mut Option<ActiveGeneration>,
    performance_log: &mut GenerationPerformanceLog,
) -> Result<(), WorkerControlError> {
    handle_worker_event(
        candidate_acknowledgement.readiness_event,
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_generation,
        performance_log,
    )?;
    handle_worker_event(
        WorkerEvent::RuntimeFeatureConfigurationApplied {
            worker_runtime_feature_configuration: candidate_acknowledgement
                .runtime_feature_configuration,
        },
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_generation,
        performance_log,
    )?;
    for deferred_process_event in candidate_acknowledgement.deferred_process_events {
        handle_worker_event(
            deferred_process_event,
            health_snapshot,
            is_ready,
            model_load_deadline,
            active_generation,
            performance_log,
        )?;
    }
    Ok(())
}

async fn reject_candidate(
    mut candidate_worker_process: WorkerProcess,
    candidate_error: WorkerControlError,
) -> WorkerControlError {
    match candidate_worker_process.close().await {
        Ok(_) => candidate_error,
        Err(cleanup_error) => {
            continue_candidate_reaping(candidate_worker_process);
            WorkerControlError::OperationAndCleanupFailed {
                operation: Box::new(candidate_error),
                cleanup: Box::new(cleanup_error),
            }
        }
    }
}

fn continue_candidate_reaping(mut candidate_worker_process: WorkerProcess) {
    // Process ownership remains in this task after the request-level cleanup
    // deadline, preventing an unreaped candidate from becoming a zombie.
    tokio::spawn(async move {
        let mut reap_attempt = 0_u64;
        loop {
            reap_attempt = reap_attempt.saturating_add(1);
            match candidate_worker_process.force_terminate().await {
                Ok(_) => return,
                Err(reap_error) => {
                    if reap_attempt <= 3 || reap_attempt % 60 == 0 {
                        tracing::error!(
                            attempt = reap_attempt,
                            error = %reap_error,
                            "candidate worker reaping attempt failed"
                        );
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
    });
}

fn combine_replacement_cleanup_errors(
    trusted_close_error: WorkerControlError,
    trusted_containment_error: Option<WorkerControlError>,
    candidate_cleanup_error: Option<WorkerControlError>,
) -> WorkerControlError {
    let operation_error = match trusted_containment_error {
        Some(containment_error) => WorkerControlError::OperationAndCleanupFailed {
            operation: Box::new(trusted_close_error),
            cleanup: Box::new(containment_error),
        },
        None => trusted_close_error,
    };
    match candidate_cleanup_error {
        Some(cleanup_error) => WorkerControlError::OperationAndCleanupFailed {
            operation: Box::new(operation_error),
            cleanup: Box::new(cleanup_error),
        },
        None => operation_error,
    }
}
