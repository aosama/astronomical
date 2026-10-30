//! Unit contracts for the pure image-event helpers: progress validation,
//! completed-result validation, and finalization duration accounting.

use std::sync::Arc;

use astronomical_ipc_protocol::{
    GeneratedImage, ImageGenerationFailureReason, ImageGenerationPhase,
    ImageGenerationResultMetadata, ImageGenerationSettings, RequestId,
};
use tokio::sync::{Semaphore, mpsc};

use super::{
    WorkerControlError, image_finalization_durations, validate_image_completed_result,
    validate_image_generation_progress,
};
use crate::worker::loop_types::ActiveImageGeneration;

fn acquirable_permit() -> tokio::sync::OwnedSemaphorePermit {
    Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable")
}

fn admitted_settings() -> ImageGenerationSettings {
    ImageGenerationSettings {
        width_pixels: 512,
        height_pixels: 512,
        steps: 4,
        guidance_thousandths: 4_500,
        seed: 7,
    }
}

fn active_image() -> ActiveImageGeneration {
    let (image_result_sender, _image_result_receiver) = mpsc::channel(1);
    ActiveImageGeneration {
        _active_generation_permit: acquirable_permit(),
        request_id: RequestId::new(2),
        model_id: String::from("qwen-image"),
        settings: admitted_settings(),
        admitted_at: tokio::time::Instant::now(),
        queue_wait_elapsed: std::time::Duration::from_millis(20),
        swap_load_elapsed: std::time::Duration::from_millis(30),
        execution_started_at: tokio::time::Instant::now(),
        execution_deadline: tokio::time::Instant::now(),
        progress_stall_deadline: tokio::time::Instant::now(),
        progress_stall_timeout: std::time::Duration::ZERO,
        latest_phase: None,
        latest_completed_steps: 0,
        latest_elapsed_millis: 0,
        terminal_received_at: None,
        image_result_sender,
        terminal_outcome: None,
    }
}

fn assert_protocol_violation<T: std::fmt::Debug>(
    validation_outcome: Result<T, WorkerControlError>,
    expected_description: &'static str,
) {
    match validation_outcome {
        Err(WorkerControlError::WorkerProtocolViolation { description }) => {
            assert_eq!(description, expected_description);
        }
        other => panic!("expected a worker protocol violation, got {other:?}"),
    }
}

#[test]
fn should_accept_the_first_conforming_progress_event() {
    let active_image = active_image();

    let validation_outcome = validate_image_generation_progress(
        &active_image,
        ImageGenerationPhase::Denoising,
        2,
        4,
        150,
    );

    assert!(
        validation_outcome.is_ok(),
        "fresh conforming progress must be accepted: {validation_outcome:?}"
    );
}

#[test]
fn should_reject_progress_whose_total_steps_disagree_with_the_admitted_settings() {
    let active_image = active_image();

    let validation_outcome = validate_image_generation_progress(
        &active_image,
        ImageGenerationPhase::Denoising,
        2,
        8,
        150,
    );

    assert_protocol_violation(validation_outcome, "invalid image generation progress");
}

#[test]
fn should_reject_progress_after_a_terminal_outcome() {
    let mut active_image = active_image();
    active_image.terminal_outcome = Some(Err(ImageGenerationFailureReason::EngineBusy));

    let validation_outcome = validate_image_generation_progress(
        &active_image,
        ImageGenerationPhase::Denoising,
        2,
        4,
        150,
    );

    assert_protocol_violation(validation_outcome, "invalid image generation progress");
}

#[test]
fn should_reject_progress_reporting_more_completed_steps_than_the_event_total() {
    let active_image = active_image();

    let validation_outcome = validate_image_generation_progress(
        &active_image,
        ImageGenerationPhase::Denoising,
        5,
        4,
        150,
    );

    assert_protocol_violation(validation_outcome, "invalid image generation progress");
}

#[test]
fn should_accept_progress_repeating_the_latest_observed_step_and_time() {
    let mut active_image = active_image();
    active_image.latest_phase = Some(ImageGenerationPhase::Denoising);
    active_image.latest_completed_steps = 2;
    active_image.latest_elapsed_millis = 150;

    let validation_outcome = validate_image_generation_progress(
        &active_image,
        ImageGenerationPhase::Denoising,
        2,
        4,
        150,
    );

    assert!(
        validation_outcome.is_ok(),
        "an exactly repeated observation is not a regression: {validation_outcome:?}"
    );
}

#[test]
fn should_reject_progress_ranked_below_the_latest_observed_phase() {
    let mut active_image = active_image();
    active_image.latest_phase = Some(ImageGenerationPhase::Decoding);

    let validation_outcome = validate_image_generation_progress(
        &active_image,
        ImageGenerationPhase::Denoising,
        3,
        4,
        150,
    );

    assert_protocol_violation(validation_outcome, "invalid image generation progress");
}

fn matching_result_metadata(elapsed_millis: u64) -> ImageGenerationResultMetadata {
    ImageGenerationResultMetadata {
        width_pixels: 512,
        height_pixels: 512,
        steps: 4,
        guidance_thousandths: 4_500,
        seed: 7,
        elapsed_millis,
    }
}

fn png_encoded_image() -> GeneratedImage {
    GeneratedImage {
        mime_type: String::from("image/png"),
        encoded_bytes: vec![1, 2, 3],
    }
}

#[test]
fn should_accept_a_completed_image_matching_the_admitted_settings() {
    let active_image = active_image();

    let validation_outcome = validate_image_completed_result(
        &active_image,
        &png_encoded_image(),
        &matching_result_metadata(500),
    );

    assert!(
        validation_outcome.is_ok(),
        "a conforming completed image must be accepted: {validation_outcome:?}"
    );
}

#[test]
fn should_reject_a_completed_image_with_a_non_png_mime_type() {
    let active_image = active_image();
    let jpeg_encoded_image = GeneratedImage {
        mime_type: String::from("image/jpeg"),
        encoded_bytes: vec![1, 2, 3],
    };

    let validation_outcome = validate_image_completed_result(
        &active_image,
        &jpeg_encoded_image,
        &matching_result_metadata(500),
    );

    assert_protocol_violation(validation_outcome, "invalid completed image result");
}

#[test]
fn should_reject_a_completed_image_after_a_terminal_outcome() {
    let mut active_image = active_image();
    active_image.terminal_outcome = Some(Err(ImageGenerationFailureReason::EngineBusy));

    let validation_outcome = validate_image_completed_result(
        &active_image,
        &png_encoded_image(),
        &matching_result_metadata(500),
    );

    assert_protocol_violation(validation_outcome, "invalid completed image result");
}

#[test]
fn should_reject_a_completed_image_with_empty_encoded_bytes() {
    let active_image = active_image();
    let empty_encoded_image = GeneratedImage {
        mime_type: String::from("image/png"),
        encoded_bytes: Vec::new(),
    };

    let validation_outcome = validate_image_completed_result(
        &active_image,
        &empty_encoded_image,
        &matching_result_metadata(500),
    );

    assert_protocol_violation(validation_outcome, "invalid completed image result");
}

#[test]
fn should_accept_completed_elapsed_time_equal_to_the_latest_observed_progress() {
    let mut active_image = active_image();
    active_image.latest_elapsed_millis = 500;

    let validation_outcome = validate_image_completed_result(
        &active_image,
        &png_encoded_image(),
        &matching_result_metadata(500),
    );

    assert!(
        validation_outcome.is_ok(),
        "completion time exactly equal to the latest progress is not a regression: \
         {validation_outcome:?}"
    );
}

#[test]
fn should_measure_every_finalization_window_from_the_admitted_request_timestamps() {
    let mut active_image = active_image();
    let admitted_at = tokio::time::Instant::now();
    active_image.admitted_at = admitted_at;
    active_image.execution_started_at = admitted_at + std::time::Duration::from_millis(300);
    active_image.terminal_received_at = Some(admitted_at + std::time::Duration::from_millis(800));
    let finalized_at = admitted_at + std::time::Duration::from_millis(1_000);

    let finalization_durations = image_finalization_durations(&active_image, finalized_at)
        .expect("a terminal-acknowledged image must produce finalization durations");

    assert_eq!(
        finalization_durations.total_elapsed_millis, 1_000,
        "total must span admission to finalization"
    );
    assert_eq!(finalization_durations.queue_wait_elapsed_millis, 20);
    assert_eq!(finalization_durations.swap_load_elapsed_millis, 30);
    assert_eq!(
        finalization_durations.execution_elapsed_millis, 500,
        "execution must span execution start to terminal receipt"
    );
    assert_eq!(
        finalization_durations.finalization_elapsed_millis, 200,
        "cleanup must span terminal receipt to finalization"
    );
}

#[test]
fn should_reject_finalization_without_a_recorded_terminal_receipt() {
    let mut active_image = active_image();
    active_image.terminal_received_at = None;
    let finalized_at = tokio::time::Instant::now();

    let finalization_outcome = image_finalization_durations(&active_image, finalized_at);

    assert_protocol_violation(
        finalization_outcome,
        "image finalization lost terminal timing",
    );
}

#[test]
fn should_report_zero_execution_and_cleanup_windows_for_an_instant_terminal_receipt() {
    let mut active_image = active_image();
    let admitted_at = tokio::time::Instant::now();
    active_image.admitted_at = admitted_at;
    active_image.execution_started_at = admitted_at + std::time::Duration::from_millis(100);
    active_image.terminal_received_at = Some(admitted_at + std::time::Duration::from_millis(100));
    let finalized_at = admitted_at + std::time::Duration::from_millis(100);

    let finalization_durations = image_finalization_durations(&active_image, finalized_at)
        .expect("a terminal-acknowledged image must produce finalization durations");

    assert_eq!(finalization_durations.execution_elapsed_millis, 0);
    assert_eq!(finalization_durations.finalization_elapsed_millis, 0);
    assert_eq!(finalization_durations.total_elapsed_millis, 100);
}
