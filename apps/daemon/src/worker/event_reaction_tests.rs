//! Unit contracts for the worker loop's non-command branch reactions: the
//! event failure classification and the image-deadline terminal send.

use astronomical_ipc_protocol::RequestId;
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc};

use crate::ImageGenerationExecutionError;
use crate::worker::control_error::WorkerControlError;
use crate::worker::event_reaction::{
    WorkerEventReaction, send_image_deadline_exceeded, worker_event_reaction_to_failure,
};
use crate::worker::loop_types::{ActiveImageGeneration, ActiveWorkerRequest};

#[test]
fn should_cancel_the_active_request_on_stream_backpressure() {
    let reaction = worker_event_reaction_to_failure(&WorkerControlError::StreamBackpressure);
    assert_eq!(reaction, WorkerEventReaction::CancelActiveGeneration);
}

#[test]
fn should_contain_the_worker_when_the_event_stream_closes() {
    let reaction = worker_event_reaction_to_failure(&WorkerControlError::WorkerEventStreamClosed);
    assert_eq!(reaction, WorkerEventReaction::ContainWorkerFailure);
}

#[test]
fn should_contain_the_worker_on_a_generation_busy_control_error() {
    // Any non-backpressure control error is a worker failure — one more
    // negative sample to pin the classification to exactly one variant.
    let reaction = worker_event_reaction_to_failure(&WorkerControlError::GenerationBusy);
    assert_eq!(reaction, WorkerEventReaction::ContainWorkerFailure);
}

fn active_image_request(
    image_result_sender: mpsc::Sender<
        Result<crate::ImageGenerationOutput, ImageGenerationExecutionError>,
    >,
) -> Option<ActiveWorkerRequest> {
    let active_generation_permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable");
    Some(ActiveWorkerRequest::Image(ActiveImageGeneration {
        _active_generation_permit: active_generation_permit,
        request_id: RequestId::new(1),
        model_id: String::from("qwen-image"),
        settings: astronomical_ipc_protocol::ImageGenerationSettings {
            width_pixels: 512,
            height_pixels: 512,
            steps: 4,
            guidance_thousandths: 4_500,
            seed: 7,
        },
        admitted_at: tokio::time::Instant::now(),
        queue_wait_elapsed: std::time::Duration::ZERO,
        swap_load_elapsed: std::time::Duration::ZERO,
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
    }))
}

#[tokio::test]
async fn should_deliver_deadline_exceeded_to_the_active_image_request() {
    let (image_result_sender, mut image_result_receiver) = mpsc::channel(1);
    let active_request = active_image_request(image_result_sender);

    send_image_deadline_exceeded(&active_request);

    let received_outcome = image_result_receiver
        .try_recv()
        .expect("the deadline send must reach the active image request");
    assert!(matches!(
        received_outcome,
        Err(ImageGenerationExecutionError::DeadlineExceeded)
    ));
}

#[tokio::test]
async fn should_not_send_anything_when_no_request_is_active() {
    let no_active_request: Option<ActiveWorkerRequest> = None;

    send_image_deadline_exceeded(&no_active_request);
    // Nothing to observe beyond the absence of a panic — the function's
    // contract for an idle loop is exactly "no send, no failure".
}

#[tokio::test]
async fn should_not_send_when_the_active_request_is_not_an_image_request() {
    // A chat request carries no image result channel; the send must be a
    // no-op rather than a protocol violation.
    let (stream_event_sender, mut stream_event_receiver) =
        mpsc::channel::<crate::ChatGenerationStreamEvent>(1);
    let active_generation_permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable");
    let active_chat_request = Some(ActiveWorkerRequest::Chat(
        crate::worker::loop_types::ActiveGeneration {
            _active_generation_permit: active_generation_permit,
            generated_token_count: 0,
            generation_started_at: None,
            generation_preparation_started_at: None,
            generation_preparation_elapsed_millis: None,
            first_decode_forward_elapsed_millis: None,
            time_to_first_output_millis: None,
            final_resident_expert_count: None,
            final_resident_expert_payload_bytes: None,
            latest_generation_progress_token_count: 0,
            max_output_tokens: 200,
            next_sequence_number: 0,
            next_tool_call_index: 0,
            completed_tool_calls: Vec::new(),
            request_started_at: tokio::time::Instant::now(),
            prefill_elapsed_millis: 0,
            maximum_mlx_peak_memory_bytes: None,
            last_mlx_active_memory_bytes: None,
            request_id: RequestId::new(2),
            stream_event_sender,
        },
    ));

    send_image_deadline_exceeded(&active_chat_request);

    assert!(
        stream_event_receiver.try_recv().is_err(),
        "the image-deadline send must not touch the chat stream channel"
    );
}
