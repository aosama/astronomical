//! Unit contracts for the pure steps of image admission: what each model
//! selection outcome answers, and the last checks before the worker is asked
//! to start. Swapping models and starting the worker need a live process and
//! run under the daemon integration journeys.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use ipc_protocol::{
    ImageGenerationCapabilities, ImageGenerationCommand, ImageGenerationSettings, RequestId,
    WorkerModelCapabilities,
};
use tokio::sync::{mpsc, oneshot};

use super::{
    ImageModelSelectionOutcome, ImageResultSender, admit_image_request_before_start,
    settle_image_model_selection,
};
use crate::{GenerationStartError, WorkerHealthSnapshot};

type StartReceiver = oneshot::Receiver<Result<(), GenerationStartError>>;

fn start_channel() -> (
    oneshot::Sender<Result<(), GenerationStartError>>,
    StartReceiver,
) {
    oneshot::channel()
}

fn image_result_channel() -> (
    ImageResultSender,
    mpsc::Receiver<Result<crate::ImageGenerationOutput, crate::ImageGenerationExecutionError>>,
) {
    mpsc::channel(1)
}

fn shakespeare_image_command(width_pixels: u32) -> ImageGenerationCommand {
    ImageGenerationCommand {
        request_id: RequestId::new(9),
        model: "image-model".to_owned(),
        prompt: "Shall I compare thee to a summer's day?".to_owned(),
        settings: ImageGenerationSettings {
            width_pixels,
            height_pixels: 512,
            steps: 4,
            guidance_thousandths: 1_000,
            seed: 7,
        },
    }
}

fn health_snapshot_advertising_images(
    image_generation: Option<ImageGenerationCapabilities>,
) -> Arc<RwLock<WorkerHealthSnapshot>> {
    Arc::new(RwLock::new(WorkerHealthSnapshot::ready_with_model(
        "image-model".to_owned(),
        WorkerModelCapabilities {
            chat: None,
            image_generation,
            embeddings: None,
            decisions: None,
            speech: None,
        },
    )))
}

fn advertised_image_envelope() -> ImageGenerationCapabilities {
    ImageGenerationCapabilities {
        minimum_width_pixels: 256,
        maximum_width_pixels: 1_024,
        minimum_height_pixels: 256,
        maximum_height_pixels: 1_024,
        dimension_multiple_pixels: 16,
        maximum_steps: 8,
        maximum_guidance_thousandths: 8_000,
        output_mime_types: vec!["image/png".to_owned()],
    }
}

#[test]
fn should_hand_the_start_sender_back_with_the_swap_load_time_when_selection_proceeds() {
    let (start_sender, mut start_receiver) = start_channel();

    let settled_selection = settle_image_model_selection(
        ImageModelSelectionOutcome::Proceed {
            swap_load_elapsed: Duration::from_millis(1_500),
        },
        start_sender,
    );

    let (swap_load_elapsed, _returned_start_sender) =
        settled_selection.expect("a proceeding selection should keep the request going");
    assert_eq!(swap_load_elapsed, Duration::from_millis(1_500));
    assert!(
        start_receiver.try_recv().is_err(),
        "a request that goes on must not be answered yet"
    );
}

#[test]
fn should_answer_a_rejected_selection_with_its_start_failure() {
    let (start_sender, mut start_receiver) = start_channel();

    let settled_selection = settle_image_model_selection(
        ImageModelSelectionOutcome::Rejected(GenerationStartError::ModelLoadFailed {
            model_load_failure_reason: "the weights are unreadable".to_owned(),
        }),
        start_sender,
    );

    assert!(settled_selection.is_none());
    assert_eq!(
        start_receiver.try_recv(),
        Ok(Err(GenerationStartError::ModelLoadFailed {
            model_load_failure_reason: "the weights are unreadable".to_owned()
        }))
    );
}

#[test]
fn should_answer_a_failed_swap_as_the_worker_being_unavailable() {
    let (start_sender, mut start_receiver) = start_channel();

    let settled_selection =
        settle_image_model_selection(ImageModelSelectionOutcome::SwapFailed, start_sender);

    assert!(settled_selection.is_none());
    assert_eq!(
        start_receiver.try_recv(),
        Ok(Err(GenerationStartError::WorkerUnavailable))
    );
}

#[test]
fn should_give_a_client_that_disconnected_during_the_swap_no_answer() {
    let (start_sender, mut start_receiver) = start_channel();

    let settled_selection =
        settle_image_model_selection(ImageModelSelectionOutcome::ClientDisconnected, start_sender);

    assert!(settled_selection.is_none());
    assert_eq!(
        start_receiver.try_recv(),
        Err(oneshot::error::TryRecvError::Closed),
        "the dropped start sender is the only signal the caller gets"
    );
}

#[test]
fn should_let_an_image_inside_the_advertised_envelope_go_on_to_start() {
    let (start_sender, _start_receiver) = start_channel();
    let (image_result_sender, _image_result_receiver) = image_result_channel();
    let health_snapshot = health_snapshot_advertising_images(Some(advertised_image_envelope()));

    let admitted_start_sender = admit_image_request_before_start(
        &image_result_sender,
        &health_snapshot,
        &shakespeare_image_command(512),
        start_sender,
    );

    assert!(admitted_start_sender.is_some());
}

#[test]
fn should_answer_an_image_outside_the_advertised_envelope_as_the_worker_being_unavailable() {
    let (start_sender, mut start_receiver) = start_channel();
    let (image_result_sender, _image_result_receiver) = image_result_channel();
    let health_snapshot = health_snapshot_advertising_images(Some(advertised_image_envelope()));

    let admitted_start_sender = admit_image_request_before_start(
        &image_result_sender,
        &health_snapshot,
        &shakespeare_image_command(2_048),
        start_sender,
    );

    assert!(admitted_start_sender.is_none());
    assert_eq!(
        start_receiver.try_recv(),
        Ok(Err(GenerationStartError::WorkerUnavailable))
    );
}

#[test]
fn should_answer_a_loaded_model_that_advertises_no_image_capabilities_as_unavailable() {
    let (start_sender, mut start_receiver) = start_channel();
    let (image_result_sender, _image_result_receiver) = image_result_channel();
    let health_snapshot = health_snapshot_advertising_images(None);

    let admitted_start_sender = admit_image_request_before_start(
        &image_result_sender,
        &health_snapshot,
        &shakespeare_image_command(512),
        start_sender,
    );

    assert!(admitted_start_sender.is_none());
    assert_eq!(
        start_receiver.try_recv(),
        Ok(Err(GenerationStartError::WorkerUnavailable))
    );
}

#[test]
fn should_give_a_client_that_hung_up_before_the_start_no_answer() {
    let (start_sender, mut start_receiver) = start_channel();
    let (image_result_sender, image_result_receiver) = image_result_channel();
    drop(image_result_receiver);
    let health_snapshot = health_snapshot_advertising_images(Some(advertised_image_envelope()));

    let admitted_start_sender = admit_image_request_before_start(
        &image_result_sender,
        &health_snapshot,
        &shakespeare_image_command(512),
        start_sender,
    );

    assert!(admitted_start_sender.is_none());
    assert_eq!(
        start_receiver.try_recv(),
        Err(oneshot::error::TryRecvError::Closed)
    );
}
