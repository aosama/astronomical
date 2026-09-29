//! Admits one image command through the shared model-swap and active-request owner.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Duration,
};

use astronomical_ipc_protocol::{
    ImageGenerationCapabilities, ImageGenerationCommand, ImageGenerationSettings, ProtocolError,
};
use tokio::time::{Instant, timeout};

use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, GenerationStartError,
    ImageGenerationExecutionError, ImageGenerationOutput, ImageGenerationTimeouts,
    RuntimeModelPolicy, WorkerActivity, WorkerControlError, WorkerHealthSnapshot, WorkerProcess,
    worker_containment::{cancel_active_generation, contain_worker_failure},
    worker_health::{clear_active_request_progress, publish_activity},
    worker_loop_types::{ActiveImageGeneration, ActiveWorkerRequest},
    worker_model_swap::{ModelSwapWaitOutcome, wait_for_model_swap},
    worker_startup_runtime::wait_for_startup_runtime_configuration,
};

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_generate_image_command(
    worker_process: &mut WorkerProcess,
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    generation_command: ImageGenerationCommand,
    start_sender: tokio::sync::oneshot::Sender<Result<(), GenerationStartError>>,
    image_result_sender: tokio::sync::mpsc::Sender<
        Result<ImageGenerationOutput, ImageGenerationExecutionError>,
    >,
    admitted_at: Instant,
    queue_wait_elapsed: Duration,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
    model_load_timeout: Duration,
    cancellation_acknowledgement_timeout: Duration,
    image_generation_timeouts: ImageGenerationTimeouts,
) -> Result<(), WorkerControlError> {
    if active_request.is_some() {
        let _send_outcome = start_sender.send(Err(GenerationStartError::CapacityUnavailable));
        tracing::error!("received GenerateImage while another request is active");
        return Ok(());
    }
    wait_for_startup_runtime_configuration(
        worker_process,
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_request,
        performance_log,
        completion_log,
        model_load_timeout,
    )
    .await?;
    let loaded_model_id = health_snapshot
        .read()
        .ok()
        .and_then(|snapshot| snapshot.ready_model_id.clone());
    let requested_model = &generation_command.model;
    let mut swap_load_elapsed = Duration::ZERO;
    if loaded_model_id.as_deref() != Some(requested_model) {
        let Some(model_policy) = model_policy_catalog.get(requested_model) else {
            tracing::error!(
                requested_model = %requested_model,
                "no runtime model policy maps the requested image model"
            );
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            return Ok(());
        };
        let expected_configuration_generation = health_snapshot.read().ok().and_then(|snapshot| {
            snapshot
                .worker_runtime_feature_configuration
                .as_ref()
                .map(|configuration| configuration.configuration_generation.clone())
        });
        let expected_model_runtime_configuration = model_policy
            .worker_model_configuration
            .runtime_configuration();
        let swap_load_started_at = Instant::now();
        if let Err(swap_error) = worker_process
            .swap_model(
                model_policy.model_directory.to_string_lossy().into_owned(),
                model_policy.worker_model_configuration.clone(),
            )
            .await
        {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_worker_failure(worker_process, health_snapshot, active_request, swap_error)
                .await;
            *is_ready = false;
            return Ok(());
        }
        let mut swap_wait = Box::pin(timeout(
            model_load_timeout,
            wait_for_model_swap(
                worker_process,
                health_snapshot,
                is_ready,
                model_load_deadline,
                active_request,
                performance_log,
                completion_log,
                expected_configuration_generation.as_deref(),
                &expected_model_runtime_configuration,
            ),
        ));
        let (swap_outcome, client_disconnected_during_swap) = tokio::select! {
            swap_outcome = &mut swap_wait => (swap_outcome, false),
            () = image_result_sender.closed() => ((&mut swap_wait).await, true),
        };
        drop(swap_wait);
        swap_load_elapsed = swap_load_started_at.elapsed();
        let swap_outcome = swap_outcome
            .map_err(|_| WorkerControlError::ModelLoadTimeout {
                model_load_timeout_millis: model_load_timeout.as_millis(),
            })
            .and_then(|outcome| outcome);
        match swap_outcome {
            Ok(ModelSwapWaitOutcome::Loaded) => {}
            Ok(ModelSwapWaitOutcome::Rejected {
                model_load_failure_reason,
            }) => {
                let _send_outcome = start_sender.send(Err(GenerationStartError::ModelLoadFailed {
                    model_load_failure_reason,
                }));
                return Ok(());
            }
            Err(swap_error) => {
                let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
                contain_worker_failure(worker_process, health_snapshot, active_request, swap_error)
                    .await;
                *is_ready = false;
                return Ok(());
            }
        }
        if client_disconnected_during_swap {
            tracing::info!(
                requested_model = %requested_model,
                "client disconnected during the image model swap"
            );
            return Ok(());
        }
    }
    if image_result_sender.is_closed() {
        tracing::info!(
            requested_model = %requested_model,
            "client disconnected before the image generation could start"
        );
        return Ok(());
    }
    let envelope_rejection_reason = health_snapshot
        .read()
        .ok()
        .and_then(|snapshot| {
            snapshot
                .ready_model_capabilities
                .as_ref()
                .and_then(|capabilities| capabilities.image_generation.as_ref())
                .map(|capabilities| {
                    image_envelope_rejection_reason(&generation_command.settings, capabilities)
                })
        })
        .unwrap_or_else(|| {
            // A missing advertisement rejects exactly like the previous boolean check:
            // without an advertised envelope this request cannot be proven servable.
            Some("the loaded model does not advertise image-generation capabilities".to_owned())
        });
    if let Some(rejection_reason) = envelope_rejection_reason {
        tracing::error!(
            requested_model = %generation_command.model,
            reason = %rejection_reason,
            "the loaded image model capabilities reject this admitted request"
        );
        let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
        return Ok(());
    }

    let request_id = generation_command.request_id;
    let model_id = generation_command.model.clone();
    let settings = generation_command.settings;
    let execution_started_at = Instant::now();
    match worker_process
        .start_image_generation(generation_command)
        .await
    {
        Ok(()) => {}
        Err(WorkerControlError::Protocol(ProtocolError::OutgoingMessageTooLarge {
            actual_message_bytes,
            maximum_message_bytes,
        })) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::RequestTooLarge {
                actual_ipc_message_bytes: actual_message_bytes,
                maximum_ipc_message_bytes: maximum_message_bytes,
            }));
            return Ok(());
        }
        Err(start_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_worker_failure(worker_process, health_snapshot, active_request, start_error)
                .await;
            *is_ready = false;
            return Ok(());
        }
    }
    *active_request = Some(ActiveWorkerRequest::Image(ActiveImageGeneration {
        _active_generation_permit: active_generation_permit,
        request_id,
        model_id,
        settings,
        admitted_at,
        queue_wait_elapsed,
        swap_load_elapsed,
        execution_started_at,
        execution_deadline: execution_started_at + image_generation_timeouts.execution_timeout,
        progress_stall_deadline: execution_started_at
            + image_generation_timeouts.progress_stall_timeout,
        progress_stall_timeout: image_generation_timeouts.progress_stall_timeout,
        latest_phase: None,
        latest_completed_steps: 0,
        latest_elapsed_millis: 0,
        terminal_received_at: None,
        image_result_sender,
        terminal_outcome: None,
    }));
    clear_active_request_progress(health_snapshot);
    publish_activity(health_snapshot, WorkerActivity::ImageGeneration);
    if start_sender.send(Ok(())).is_err() {
        cancel_active_generation(
            worker_process,
            health_snapshot,
            active_request,
            cancellation_acknowledgement_timeout,
            model_load_timeout,
            is_ready,
        )
        .await;
    }
    Ok(())
}

/// Names the first advertised capability an image request violates, if any.
///
/// The public endpoint already validates dimensions against the discovery-time envelope,
/// so a rejection here means the loaded advertisement disagrees with what was admitted —
/// logged at error level because that disagreement is a supervisor/worker contract break.
fn image_envelope_rejection_reason(
    settings: &ImageGenerationSettings,
    capabilities: &ImageGenerationCapabilities,
) -> Option<String> {
    for (parameter_name, actual_pixels, minimum_pixels, maximum_pixels) in [
        (
            "width",
            settings.width_pixels,
            capabilities.minimum_width_pixels,
            capabilities.maximum_width_pixels,
        ),
        (
            "height",
            settings.height_pixels,
            capabilities.minimum_height_pixels,
            capabilities.maximum_height_pixels,
        ),
    ] {
        if actual_pixels < minimum_pixels || actual_pixels > maximum_pixels {
            return Some(format!(
                "{parameter_name} {actual_pixels} is outside the {minimum_pixels}..={maximum_pixels} pixel envelope"
            ));
        }
    }
    if capabilities.dimension_multiple_pixels == 0
        || !settings
            .width_pixels
            .is_multiple_of(capabilities.dimension_multiple_pixels)
        || !settings
            .height_pixels
            .is_multiple_of(capabilities.dimension_multiple_pixels)
    {
        return Some(format!(
            "dimensions {}x{} are not multiples of {} pixels",
            settings.width_pixels, settings.height_pixels, capabilities.dimension_multiple_pixels
        ));
    }
    if settings.steps > capabilities.maximum_steps {
        return Some(format!(
            "steps {} exceed the advertised maximum of {}",
            settings.steps, capabilities.maximum_steps
        ));
    }
    if settings.guidance_thousandths > capabilities.maximum_guidance_thousandths {
        return Some(format!(
            "guidance {} thousandths exceed the advertised maximum of {}",
            settings.guidance_thousandths, capabilities.maximum_guidance_thousandths
        ));
    }
    if !capabilities
        .output_mime_types
        .iter()
        .any(|mime_type| mime_type == "image/png")
    {
        return Some("the loaded model does not advertise image/png output".to_owned());
    }
    None
}
