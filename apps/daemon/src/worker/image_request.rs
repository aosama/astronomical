//! Admits one image command through the shared model-swap and active-request owner.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Duration,
};

use ipc_protocol::{
    ImageGenerationCapabilities, ImageGenerationCommand, ImageGenerationSettings, ProtocolError,
};
use tokio::time::Instant;

use crate::worker::loop_types::WorkerLoopState;
use crate::worker::request_start::{
    GenerationStartSender, contain_failed_worker, loaded_model_id,
    published_configuration_generation,
};
use crate::{
    GenerationStartError, ImageGenerationExecutionError, ImageGenerationOutput,
    ImageGenerationTimeouts, RuntimeModelPolicy, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot, WorkerProcess,
    worker::health::{clear_active_request_progress, publish_activity},
    worker::loop_types::{ActiveImageGeneration, ActiveWorkerRequest},
    worker::process::containment::cancel_active_generation,
    worker::process::model_swap::{ModelSwapWaitOutcome, wait_for_model_swap_within_budget},
    worker::process::startup_runtime::wait_for_startup_runtime_configuration,
};

/// Everything the loop knows about one image command when it admits it.
pub(super) struct ImageAdmission {
    pub active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    pub generation_command: ImageGenerationCommand,
    pub start_sender: GenerationStartSender,
    pub image_result_sender: ImageResultSender,
    pub admitted_at: Instant,
    pub queue_wait_elapsed: Duration,
}

/// Where the worker's image outcome, or its failure, is delivered.
type ImageResultSender =
    tokio::sync::mpsc::Sender<Result<ImageGenerationOutput, ImageGenerationExecutionError>>;

pub(super) async fn handle_generate_image_command(
    worker_loop_state: &mut WorkerLoopState<'_>,
    image_admission: ImageAdmission,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
    cancellation_acknowledgement_timeout: Duration,
    image_generation_timeouts: ImageGenerationTimeouts,
) -> Result<(), WorkerControlError> {
    let ImageAdmission {
        active_generation_permit,
        generation_command,
        start_sender,
        image_result_sender,
        admitted_at,
        queue_wait_elapsed,
    } = image_admission;
    if worker_loop_state.active_request.is_some() {
        let _send_outcome = start_sender.send(Err(GenerationStartError::CapacityUnavailable));
        tracing::error!("received GenerateImage while another request is active");
        return Ok(());
    }
    let Some((swap_load_elapsed, start_sender)) = ready_worker_for_image_request(
        worker_loop_state,
        &generation_command,
        &image_result_sender,
        start_sender,
        model_policy_catalog,
    )
    .await?
    else {
        return Ok(());
    };
    let request_id = generation_command.request_id;
    let model_id = generation_command.model.clone();
    let settings = generation_command.settings;
    let generation_start_outcome =
        start_image_generation_or_reject(worker_loop_state.worker_process, generation_command)
            .await?;
    let Some(start_sender) =
        settle_image_generation_start(worker_loop_state, generation_start_outcome, start_sender)
            .await
    else {
        return Ok(());
    };
    *worker_loop_state.active_request = Some(assemble_active_image_generation(
        active_generation_permit,
        request_id,
        model_id,
        settings,
        admitted_at,
        queue_wait_elapsed,
        swap_load_elapsed,
        Instant::now(),
        image_generation_timeouts,
        image_result_sender,
    ));
    acknowledge_or_cancel_image_start(
        worker_loop_state,
        start_sender,
        cancellation_acknowledgement_timeout,
    )
    .await;
    Ok(())
}

/// Gets the worker ready for the image request: waits out its startup
/// configuration, makes the requested model the loaded one, and runs the last
/// checks before the start. Hands the swap's measured load time and the start
/// sender back when the request goes on; otherwise the start sender has been
/// answered, or dropped for a client that left.
async fn ready_worker_for_image_request(
    worker_loop_state: &mut WorkerLoopState<'_>,
    generation_command: &ImageGenerationCommand,
    image_result_sender: &ImageResultSender,
    start_sender: GenerationStartSender,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
) -> Result<Option<(Duration, GenerationStartSender)>, WorkerControlError> {
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
    let model_selection_outcome = ensure_image_model_loaded(
        worker_loop_state,
        &generation_command.model,
        model_policy_catalog,
        image_result_sender,
    )
    .await?;
    let Some((swap_load_elapsed, start_sender)) =
        settle_image_model_selection(model_selection_outcome, start_sender)
    else {
        return Ok(None);
    };
    let Some(start_sender) = admit_image_request_before_start(
        image_result_sender,
        worker_loop_state.health_snapshot,
        generation_command,
        start_sender,
    ) else {
        return Ok(None);
    };
    Ok(Some((swap_load_elapsed, start_sender)))
}

/// Ends the request when image model selection stopped it, answering the
/// start sender, and hands the sender back with the swap's measured load
/// time when the request goes on. A client that disconnected during the swap
/// gets no answer.
fn settle_image_model_selection(
    model_selection_outcome: ImageModelSelectionOutcome,
    start_sender: GenerationStartSender,
) -> Option<(Duration, GenerationStartSender)> {
    match model_selection_outcome {
        ImageModelSelectionOutcome::Proceed { swap_load_elapsed } => {
            Some((swap_load_elapsed, start_sender))
        }
        ImageModelSelectionOutcome::Rejected(start_failure) => {
            let _send_outcome = start_sender.send(Err(start_failure));
            None
        }
        ImageModelSelectionOutcome::SwapFailed => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            None
        }
        ImageModelSelectionOutcome::ClientDisconnected => None,
    }
}

/// The last checks before the worker is asked to start the image: the client
/// must still be waiting, and the loaded model's advertised envelope must
/// admit the request. Hands the start sender back when the request goes on;
/// a client that has gone gets no answer.
fn admit_image_request_before_start(
    image_result_sender: &ImageResultSender,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    generation_command: &ImageGenerationCommand,
    start_sender: GenerationStartSender,
) -> Option<GenerationStartSender> {
    if image_result_sender.is_closed() {
        tracing::info!(
            requested_model = %generation_command.model,
            "client disconnected before the image generation could start"
        );
        return None;
    }
    if let Err(start_failure) = ensure_advertised_image_envelope(
        health_snapshot,
        &generation_command.model,
        &generation_command.settings,
    ) {
        let _send_outcome = start_sender.send(Err(start_failure));
        return None;
    }
    Some(start_sender)
}

/// Ends the request when the worker did not start the image, answering the
/// start sender, and hands the sender back when the generation is running. A
/// failed start answers before the worker is contained.
async fn settle_image_generation_start(
    worker_loop_state: &mut WorkerLoopState<'_>,
    generation_start_outcome: GenerationImageStartOutcome,
    start_sender: GenerationStartSender,
) -> Option<GenerationStartSender> {
    match generation_start_outcome {
        GenerationImageStartOutcome::Started => Some(start_sender),
        GenerationImageStartOutcome::Rejected(start_failure) => {
            let _send_outcome = start_sender.send(Err(start_failure));
            None
        }
        GenerationImageStartOutcome::Failed(start_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_failed_worker(worker_loop_state, start_error).await;
            None
        }
    }
}

/// Publishes the image-generation activity and answers the start sender; a
/// failed acknowledgement (the client already hung up) cancels the running
/// generation.
async fn acknowledge_or_cancel_image_start(
    worker_loop_state: &mut WorkerLoopState<'_>,
    start_sender: GenerationStartSender,
    cancellation_acknowledgement_timeout: Duration,
) {
    clear_active_request_progress(worker_loop_state.health_snapshot);
    publish_activity(
        worker_loop_state.health_snapshot,
        WorkerActivity::ImageGeneration,
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

/// The image model-selection outcome. `Proceed` carries the swap's measured
/// load duration; `SwapFailed` is already contained inside the unit and the
/// caller only answers the start sender.
enum ImageModelSelectionOutcome {
    Proceed { swap_load_elapsed: Duration },
    Rejected(GenerationStartError),
    SwapFailed,
    ClientDisconnected,
}

/// Ensures the requested image model is the loaded one, swapping through the
/// policy catalog when it is not. The swap's wait races the client
/// disconnecting: a disconnect ends the request without a failure answer.
async fn ensure_image_model_loaded(
    worker_loop_state: &mut WorkerLoopState<'_>,
    requested_model: &str,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
    image_result_sender: &ImageResultSender,
) -> Result<ImageModelSelectionOutcome, WorkerControlError> {
    let loaded_model_id = loaded_model_id(worker_loop_state.health_snapshot);
    if loaded_model_id.as_deref() == Some(requested_model) {
        return Ok(ImageModelSelectionOutcome::Proceed {
            swap_load_elapsed: Duration::ZERO,
        });
    }
    let Some(model_policy) = model_policy_catalog.get(requested_model) else {
        tracing::error!(
            requested_model = %requested_model,
            "no runtime model policy maps the requested image model"
        );
        return Ok(ImageModelSelectionOutcome::Rejected(
            GenerationStartError::WorkerUnavailable,
        ));
    };
    swap_to_image_model(
        worker_loop_state,
        requested_model,
        model_policy,
        image_result_sender,
    )
    .await
}

/// Swaps the worker to the requested image model and waits for it inside the
/// load budget. The wait races the client disconnecting; the swap is still
/// waited out so the worker's state stays consistent, and the disconnect is
/// reported after it.
async fn swap_to_image_model(
    worker_loop_state: &mut WorkerLoopState<'_>,
    requested_model: &str,
    model_policy: &RuntimeModelPolicy,
    image_result_sender: &ImageResultSender,
) -> Result<ImageModelSelectionOutcome, WorkerControlError> {
    let expected_configuration_generation =
        published_configuration_generation(worker_loop_state.health_snapshot);
    let expected_model_runtime_configuration = model_policy
        .worker_model_configuration
        .runtime_configuration();
    let swap_load_started_at = Instant::now();
    if let Err(swap_error) = worker_loop_state
        .worker_process
        .swap_model(
            model_policy.model_directory.to_string_lossy().into_owned(),
            model_policy.worker_model_configuration.clone(),
        )
        .await
    {
        contain_failed_worker(worker_loop_state, swap_error).await;
        return Ok(ImageModelSelectionOutcome::SwapFailed);
    }
    let mut swap_wait = Box::pin(wait_for_model_swap_within_budget(
        worker_loop_state,
        expected_configuration_generation.as_deref(),
        &expected_model_runtime_configuration,
    ));
    let (swap_outcome, client_disconnected_during_swap) = tokio::select! {
        swap_outcome = &mut swap_wait => (swap_outcome, false),
        () = image_result_sender.closed() => ((&mut swap_wait).await, true),
    };
    drop(swap_wait);
    let swap_load_elapsed = swap_load_started_at.elapsed();
    match swap_outcome {
        Ok(ModelSwapWaitOutcome::Loaded) => {}
        Ok(ModelSwapWaitOutcome::Rejected {
            model_load_failure_reason,
        }) => {
            return Ok(ImageModelSelectionOutcome::Rejected(
                GenerationStartError::ModelLoadFailed {
                    model_load_failure_reason,
                },
            ));
        }
        Err(swap_error) => {
            contain_failed_worker(worker_loop_state, swap_error).await;
            return Ok(ImageModelSelectionOutcome::SwapFailed);
        }
    }
    if client_disconnected_during_swap {
        tracing::info!(
            requested_model = %requested_model,
            "client disconnected during the image model swap"
        );
        return Ok(ImageModelSelectionOutcome::ClientDisconnected);
    }
    Ok(ImageModelSelectionOutcome::Proceed { swap_load_elapsed })
}

/// Ensures the loaded model's advertised image envelope admits the request;
/// a violation (or a missing advertisement) logs the named reason and
/// answers with a worker-unavailable start failure.
fn ensure_advertised_image_envelope(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    requested_model: &str,
    settings: &ImageGenerationSettings,
) -> Result<(), GenerationStartError> {
    let rejection_reason = health_snapshot
        .read()
        .ok()
        .and_then(|snapshot| {
            snapshot
                .ready_model_capabilities
                .as_ref()
                .and_then(|capabilities| capabilities.image_generation.as_ref())
                .map(|capabilities| image_envelope_rejection_reason(settings, capabilities))
        })
        .unwrap_or_else(|| {
            Some("the loaded model does not advertise image-generation capabilities".to_owned())
        });
    let Some(rejection_reason) = rejection_reason else {
        return Ok(());
    };
    tracing::error!(
        requested_model = %requested_model,
        reason = %rejection_reason,
        "the loaded image model capabilities reject this admitted request"
    );
    Err(GenerationStartError::WorkerUnavailable)
}

/// The image generation-start outcome: running on the worker, or the start
/// failure the caller answers with (the failed case carries the process
/// error for containment after the answer).
enum GenerationImageStartOutcome {
    Started,
    Rejected(GenerationStartError),
    Failed(WorkerControlError),
}

/// Starts the image generation on the worker process. The generic failure
/// case contains the worker failure and clears readiness before returning
/// `Failed`, exactly as the pre-split flow ordered it.
async fn start_image_generation_or_reject(
    worker_process: &mut WorkerProcess,
    generation_command: ImageGenerationCommand,
) -> Result<GenerationImageStartOutcome, WorkerControlError> {
    match worker_process
        .start_image_generation(generation_command)
        .await
    {
        Ok(()) => Ok(GenerationImageStartOutcome::Started),
        Err(WorkerControlError::Protocol(ProtocolError::OutgoingMessageTooLarge {
            actual_message_bytes,
            maximum_message_bytes,
        })) => Ok(GenerationImageStartOutcome::Rejected(
            GenerationStartError::RequestTooLarge {
                actual_ipc_message_bytes: actual_message_bytes,
                maximum_ipc_message_bytes: maximum_message_bytes,
            },
        )),
        Err(start_error) => Ok(GenerationImageStartOutcome::Failed(start_error)),
    }
}

/// Pure: assembles the active-request state one image generation owns until
/// it resolves.
#[allow(clippy::too_many_arguments)]
fn assemble_active_image_generation(
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    request_id: ipc_protocol::RequestId,
    model_id: String,
    settings: ImageGenerationSettings,
    admitted_at: Instant,
    queue_wait_elapsed: Duration,
    swap_load_elapsed: Duration,
    execution_started_at: Instant,
    image_generation_timeouts: ImageGenerationTimeouts,
    image_result_sender: tokio::sync::mpsc::Sender<
        Result<ImageGenerationOutput, ImageGenerationExecutionError>,
    >,
) -> ActiveWorkerRequest {
    ActiveWorkerRequest::Image(ActiveImageGeneration {
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
    })
}

/// Names the first advertised capability an image request violates, if any.
///
/// The public endpoint already validates dimensions against the discovery-time envelope,
/// so a rejection here means the loaded advertisement disagrees with what was admitted —
/// logged at error level because that disagreement is a daemon/worker contract break.
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

#[cfg(test)]
#[path = "image_request_tests.rs"]
mod image_request_tests;
