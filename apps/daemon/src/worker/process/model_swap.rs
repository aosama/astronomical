use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{
    WorkerEvent, WorkerLoadedModelRuntimeConfiguration, WorkerRuntimeFeatureConfiguration,
};

use crate::worker::event_handler;
use crate::worker::health;
use crate::worker::loop_types::WorkerLoopState;
use crate::{WorkerControlError, WorkerHealthSnapshot};

/// Result of waiting for a worker model-swap acknowledgement.
pub(in crate::worker) enum ModelSwapWaitOutcome {
    Loaded,
    Rejected { model_load_failure_reason: String },
}

/// The swap acknowledgement pieces staged while draining queued events.
struct StagedModelSwapAcknowledgement {
    model_swap_event: Option<WorkerEvent>,
    runtime_configuration: Option<WorkerRuntimeFeatureConfiguration>,
}

impl StagedModelSwapAcknowledgement {
    fn new() -> Self {
        Self {
            model_swap_event: None,
            runtime_configuration: None,
        }
    }
}

/// The swap contract the caller expects the worker to acknowledge.
struct ExpectedModelSwap<'a> {
    configuration_generation: Option<&'a str>,
    model_runtime_configuration: &'a WorkerLoadedModelRuntimeConfiguration,
}

/// What one drained worker event means for the pending swap wait.
enum ModelSwapEventReaction {
    /// Not a swap terminator; keep draining queued events.
    AwaitNextWorkerEvent,
    /// The worker rejected the swap; the wait ends with this outcome.
    SwapFailed(ModelSwapWaitOutcome),
}

/// Drains worker events until the model swap succeeds or is rejected.
///
/// Process-scoped telemetry and configuration events may have been queued
/// before `SwapModel`. They remain valid while the swap acknowledgement is
/// pending and must update daemon health rather than terminate the worker.
pub(in crate::worker) async fn wait_for_model_swap(
    worker_loop_state: &mut WorkerLoopState<'_>,
    expected_configuration_generation: Option<&str>,
    expected_model_runtime_configuration: &WorkerLoadedModelRuntimeConfiguration,
) -> Result<ModelSwapWaitOutcome, WorkerControlError> {
    // The swap wait never applies the load budget itself (callers wrap it in
    // their own timeout), so the state's timeout field stays untouched here.
    let expected_model_swap = ExpectedModelSwap {
        configuration_generation: expected_configuration_generation,
        model_runtime_configuration: expected_model_runtime_configuration,
    };
    let mut staged_model_swap = StagedModelSwapAcknowledgement::new();
    loop {
        let worker_event = match await_next_worker_event(worker_loop_state).await {
            Ok(Some(worker_event)) => worker_event,
            Ok(None) => return Err(WorkerControlError::WorkerEventStreamClosed),
            Err(worker_event_error) => return Err(worker_event_error),
        };
        let worker_event_reaction = react_to_worker_event_while_awaiting_model_swap(
            worker_loop_state,
            &mut staged_model_swap,
            worker_event,
            &expected_model_swap,
        )?;
        if let ModelSwapEventReaction::SwapFailed(model_swap_wait_outcome) = worker_event_reaction {
            return Ok(model_swap_wait_outcome);
        }
        if let Some(model_swap_wait_outcome) = finish_staged_model_swap(
            worker_loop_state,
            &mut staged_model_swap,
            &expected_model_swap,
        )? {
            return Ok(model_swap_wait_outcome);
        }
    }
}

/// Polls the next worker event; `None` means the event stream closed.
async fn await_next_worker_event(
    worker_loop_state: &mut WorkerLoopState<'_>,
) -> Result<Option<WorkerEvent>, WorkerControlError> {
    worker_loop_state.worker_process.next_event().await
}

/// Classifies one drained worker event against the pending swap: stage the
/// model acknowledgement, apply a rejection, or route the event onward.
fn react_to_worker_event_while_awaiting_model_swap(
    worker_loop_state: &mut WorkerLoopState<'_>,
    staged_model_swap: &mut StagedModelSwapAcknowledgement,
    worker_event: WorkerEvent,
    expected_model_swap: &ExpectedModelSwap<'_>,
) -> Result<ModelSwapEventReaction, WorkerControlError> {
    match worker_event {
        model_swapped_event @ WorkerEvent::ModelSwapped { .. } => {
            stage_model_swap_acknowledgement(
                staged_model_swap,
                model_swapped_event,
                expected_model_swap,
            )?;
            Ok(ModelSwapEventReaction::AwaitNextWorkerEvent)
        }
        WorkerEvent::ModelSwapFailed {
            loaded_model_remains_ready,
            model_load_failure_reason,
        } => Ok(ModelSwapEventReaction::SwapFailed(reject_model_swap(
            worker_loop_state,
            loaded_model_remains_ready,
            model_load_failure_reason,
        ))),
        interleaved_worker_event => {
            process_interleaved_worker_event(
                worker_loop_state,
                staged_model_swap,
                interleaved_worker_event,
                expected_model_swap,
            )?;
            Ok(ModelSwapEventReaction::AwaitNextWorkerEvent)
        }
    }
}

/// Stages a `ModelSwapped` acknowledgement after validating it; a second
/// acknowledgement while one is staged is a protocol violation.
fn stage_model_swap_acknowledgement(
    staged_model_swap: &mut StagedModelSwapAcknowledgement,
    model_swapped_event: WorkerEvent,
    expected_model_swap: &ExpectedModelSwap<'_>,
) -> Result<(), WorkerControlError> {
    if staged_model_swap.model_swap_event.is_some() {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "duplicate model swap acknowledgement",
        });
    }
    validate_model_swap_event(
        &model_swapped_event,
        expected_model_swap.model_runtime_configuration,
    )?;
    staged_model_swap.model_swap_event = Some(model_swapped_event);
    Ok(())
}

/// Applies a swap rejection: a failed first load leaves a healthy
/// model-less worker, while a failed replacement keeps the prior model's
/// health snapshot intact.
fn reject_model_swap(
    worker_loop_state: &mut WorkerLoopState<'_>,
    loaded_model_remains_ready: bool,
    model_load_failure_reason: String,
) -> ModelSwapWaitOutcome {
    // A failed first load leaves a healthy model-less worker. A
    // failed replacement may leave the prior model ready; its
    // existing health snapshot must remain intact in that case.
    if !loaded_model_remains_ready {
        let WorkerLoopState {
            health_snapshot, ..
        } = worker_loop_state;
        let health_snapshot = *health_snapshot;
        let mlx_memory_ceilings = health_snapshot
            .read()
            .map(|worker_health_snapshot| {
                (
                    worker_health_snapshot.machine_mlx_memory_ceiling_bytes,
                    worker_health_snapshot.mlx_memory_ceiling_bytes,
                    worker_health_snapshot.minimum_mlx_memory_ceiling_bytes,
                    worker_health_snapshot.wired_residency_budget_bytes,
                )
            })
            .unwrap_or((0, 0, 1, None));
        health::publish_health(
            health_snapshot,
            WorkerHealthSnapshot::ready_without_model_with_memory_limits(
                mlx_memory_ceilings.0,
                mlx_memory_ceilings.1,
                mlx_memory_ceilings.2,
                mlx_memory_ceilings.3,
            ),
        );
    }
    tracing::warn!(
        loaded_model_remains_ready,
        model_load_failure_reason,
        "worker rejected the requested model while remaining responsive"
    );
    ModelSwapWaitOutcome::Rejected {
        model_load_failure_reason,
    }
}

/// Routes one non-swap worker event while the swap acknowledgement is
/// pending: stage a matching runtime-policy acknowledgement, or hand the
/// event to the central handler.
fn process_interleaved_worker_event(
    worker_loop_state: &mut WorkerLoopState<'_>,
    staged_model_swap: &mut StagedModelSwapAcknowledgement,
    interleaved_worker_event: WorkerEvent,
    expected_model_swap: &ExpectedModelSwap<'_>,
) -> Result<(), WorkerControlError> {
    // Do not enumerate supposedly harmless events here. The
    // central handler accepts process-scoped updates and rejects
    // generation events whose active-request contract is absent.
    let interleaved_worker_event_summary = interleaved_worker_event.diagnostic_summary();
    let is_staged_runtime_acknowledgement = stage_runtime_policy_acknowledgement(
        staged_model_swap,
        &interleaved_worker_event,
        expected_model_swap,
    )?;
    if !is_staged_runtime_acknowledgement {
        dispatch_worker_event(worker_loop_state, interleaved_worker_event)?;
        tracing::debug!(
            worker_event = interleaved_worker_event_summary,
            "processed interleaved worker event while awaiting model swap"
        );
    }
    Ok(())
}

/// Stages the `RuntimeFeatureConfigurationApplied` acknowledgement that must
/// pair with the swap when the swap carries a configuration generation.
/// Returns whether the event was consumed as that acknowledgement; every
/// other event is left for the central handler.
fn stage_runtime_policy_acknowledgement(
    staged_model_swap: &mut StagedModelSwapAcknowledgement,
    worker_event: &WorkerEvent,
    expected_model_swap: &ExpectedModelSwap<'_>,
) -> Result<bool, WorkerControlError> {
    let WorkerEvent::RuntimeFeatureConfigurationApplied {
        worker_runtime_feature_configuration,
    } = worker_event
    else {
        return Ok(false);
    };
    let Some(expected_configuration_generation) = expected_model_swap.configuration_generation
    else {
        return Ok(false);
    };
    if worker_runtime_feature_configuration.configuration_generation
        != expected_configuration_generation
        || worker_runtime_feature_configuration.loaded_model.as_ref()
            != Some(expected_model_swap.model_runtime_configuration)
    {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "model swap runtime policy acknowledgement mismatch",
        });
    }
    if staged_model_swap
        .runtime_configuration
        .replace(worker_runtime_feature_configuration.clone())
        .is_some()
    {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "duplicate model swap runtime policy acknowledgement",
        });
    }
    Ok(true)
}

/// Hands one drained worker event to the central handler so both event
/// loops enforce the same protocol rules on daemon-owned state.
fn dispatch_worker_event(
    worker_loop_state: &mut WorkerLoopState<'_>,
    interleaved_worker_event: WorkerEvent,
) -> Result<(), WorkerControlError> {
    let WorkerLoopState {
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_request,
        performance_log,
        completion_log,
        ..
    } = worker_loop_state;
    let health_snapshot = *health_snapshot;
    let is_ready = &mut **is_ready;
    let model_load_deadline = &mut **model_load_deadline;
    let active_request = &mut **active_request;
    let performance_log = &mut **performance_log;
    let completion_log = &mut **completion_log;
    event_handler::handle_worker_event(
        interleaved_worker_event,
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_request,
        performance_log,
        completion_log,
    )
}

/// Publishes the swap once both the `ModelSwapped` acknowledgement and (when
/// the swap carries a configuration generation) its runtime-policy pair are
/// staged. Returns `None` while the wait must keep draining events.
fn finish_staged_model_swap(
    worker_loop_state: &mut WorkerLoopState<'_>,
    staged_model_swap: &mut StagedModelSwapAcknowledgement,
    expected_model_swap: &ExpectedModelSwap<'_>,
) -> Result<Option<ModelSwapWaitOutcome>, WorkerControlError> {
    let policy_is_ready = expected_model_swap.configuration_generation.is_none()
        || staged_model_swap.runtime_configuration.is_some();
    if staged_model_swap.model_swap_event.is_some() && policy_is_ready {
        let WorkerLoopState {
            health_snapshot, ..
        } = worker_loop_state;
        let health_snapshot = *health_snapshot;
        publish_staged_model_swap(
            health_snapshot,
            staged_model_swap.model_swap_event.take().ok_or(
                WorkerControlError::WorkerProtocolViolation {
                    description: "model swap acknowledgement staging failed",
                },
            )?,
            staged_model_swap.runtime_configuration.take(),
        )?;
        return Ok(Some(ModelSwapWaitOutcome::Loaded));
    }
    Ok(None)
}

fn validate_model_swap_event(
    model_swap_event: &WorkerEvent,
    expected_runtime_configuration: &WorkerLoadedModelRuntimeConfiguration,
) -> Result<(), WorkerControlError> {
    let WorkerEvent::ModelSwapped {
        model_id,
        capabilities,
        ..
    } = model_swap_event
    else {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "non-model event staged as model swap acknowledgement",
        });
    };
    if model_id != expected_runtime_configuration.model_id() {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "model swap identity acknowledgement mismatch",
        });
    }
    let capabilities_match_policy = match expected_runtime_configuration {
        WorkerLoadedModelRuntimeConfiguration::Autoregressive(configuration) => {
            capabilities.chat.as_ref().is_some_and(|chat_capabilities| {
                chat_capabilities.context_window == configuration.maximum_context_tokens
                    && chat_capabilities.max_output_tokens == configuration.maximum_output_tokens
                    && chat_capabilities.max_input_tokens
                        == configuration.maximum_context_tokens.saturating_sub(1)
            }) && capabilities.image_generation.is_none()
        }
        WorkerLoadedModelRuntimeConfiguration::Flux2Klein(_) => {
            capabilities.chat.is_none() && capabilities.image_generation.is_some()
        }
        WorkerLoadedModelRuntimeConfiguration::QwenImage21(_) => {
            capabilities.chat.is_none() && capabilities.image_generation.is_some()
        }
        WorkerLoadedModelRuntimeConfiguration::Embeddings(_) => {
            capabilities.chat.is_none()
                && capabilities.image_generation.is_none()
                && capabilities.embeddings.is_some()
        }
    };
    if !capabilities_match_policy {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "model swap capabilities acknowledgement mismatch",
        });
    }
    Ok(())
}

fn publish_staged_model_swap(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    model_swap_event: WorkerEvent,
    runtime_configuration: Option<WorkerRuntimeFeatureConfiguration>,
) -> Result<(), WorkerControlError> {
    let WorkerEvent::ModelSwapped {
        model_id,
        capabilities,
        expert_memory_mode,
        minimum_mlx_memory_ceiling_bytes,
    } = model_swap_event
    else {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "non-model event staged as model swap acknowledgement",
        });
    };
    let Ok(mut current_health_snapshot) = health_snapshot.write() else {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "worker health lock is unavailable",
        });
    };
    let mut replacement_health_snapshot = WorkerHealthSnapshot::ready_with_replacement_model(
        model_id,
        capabilities,
        minimum_mlx_memory_ceiling_bytes,
        &current_health_snapshot,
    );
    replacement_health_snapshot.expert_memory_mode = expert_memory_mode;
    if let Some(runtime_configuration) = runtime_configuration {
        replacement_health_snapshot.worker_runtime_feature_configuration =
            Some(runtime_configuration);
    }
    *current_health_snapshot = replacement_health_snapshot;
    Ok(())
}

#[cfg(test)]
#[path = "model_swap_tests.rs"]
mod model_swap_tests;
