//! Unit contracts for the loop-carried types: the active request's per-modality
//! channel extractors the worker loop's `select!` branches depend on.

use std::sync::Arc;

use astronomical_ipc_protocol::{ImageGenerationSettings, RequestId};
use tokio::sync::{Semaphore, mpsc};

use crate::worker::loop_types::{
    ActiveEmbeddingsGeneration, ActiveGeneration, ActiveImageGeneration, ActiveWorkerRequest,
};
use crate::{ChatGenerationStreamEvent, EmbeddingsExecutionError, ImageGenerationExecutionError};

fn acquirable_permit() -> tokio::sync::OwnedSemaphorePermit {
    Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable")
}

fn active_chat_request(
    stream_event_sender: mpsc::Sender<ChatGenerationStreamEvent>,
) -> ActiveWorkerRequest {
    ActiveWorkerRequest::Chat(ActiveGeneration {
        _active_generation_permit: acquirable_permit(),
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
        request_id: RequestId::new(1),
        stream_event_sender,
    })
}

fn active_image_request(
    image_result_sender: mpsc::Sender<
        Result<crate::ImageGenerationOutput, ImageGenerationExecutionError>,
    >,
) -> ActiveWorkerRequest {
    ActiveWorkerRequest::Image(ActiveImageGeneration {
        _active_generation_permit: acquirable_permit(),
        request_id: RequestId::new(2),
        model_id: String::from("qwen-image"),
        settings: ImageGenerationSettings {
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
    })
}

fn active_embeddings_request(
    embeddings_result_sender: mpsc::Sender<
        Result<crate::EmbeddingsOutput, EmbeddingsExecutionError>,
    >,
) -> ActiveWorkerRequest {
    ActiveWorkerRequest::Embeddings(ActiveEmbeddingsGeneration {
        _active_generation_permit: acquirable_permit(),
        request_id: RequestId::new(3),
        terminal_received_at: None,
        execution_deadline: tokio::time::Instant::now(),
        embeddings_result_sender,
        terminal_outcome: None,
    })
}

#[test]
fn should_return_the_chat_stream_channel_only_for_an_active_chat_request() {
    let (stream_event_sender, mut stream_event_receiver) = mpsc::channel(1);
    let active_request = active_chat_request(stream_event_sender);

    let extracted_chat_sender = active_request.chat_stream_event_sender();
    assert!(
        extracted_chat_sender.is_some(),
        "a chat request must expose its stream-event channel"
    );
    assert!(
        active_request.image_result_sender().is_none(),
        "a chat request has no image result channel"
    );
    assert!(
        active_request.embeddings_result_sender().is_none(),
        "a chat request has no embeddings result channel"
    );

    // The extracted sender must address the very same channel the request
    // owns: an event sent through the clone arrives on the original receiver.
    let send_outcome = extracted_chat_sender.expect("checked above").try_send(
        ChatGenerationStreamEvent::TextFragment(String::from("roundtrip")),
    );
    assert!(send_outcome.is_ok());
    let received_event = stream_event_receiver
        .try_recv()
        .expect("the extracted clone addresses the original channel");
    assert_eq!(
        received_event,
        ChatGenerationStreamEvent::TextFragment(String::from("roundtrip"))
    );
}

#[test]
fn should_return_the_image_result_channel_only_for_an_active_image_request() {
    let (image_result_sender, mut image_result_receiver) = mpsc::channel(1);
    let active_request = active_image_request(image_result_sender);

    let extracted_image_sender = active_request.image_result_sender();
    assert!(
        extracted_image_sender.is_some(),
        "an image request must expose its result channel"
    );
    assert!(
        active_request.chat_stream_event_sender().is_none(),
        "an image request has no chat stream channel"
    );
    assert!(
        active_request.embeddings_result_sender().is_none(),
        "an image request has no embeddings result channel"
    );

    let send_outcome = extracted_image_sender
        .expect("checked above")
        .try_send(Err(ImageGenerationExecutionError::DeadlineExceeded));
    assert!(send_outcome.is_ok());
    let received_outcome = image_result_receiver
        .try_recv()
        .expect("the extracted clone addresses the original channel");
    assert!(matches!(
        received_outcome,
        Err(ImageGenerationExecutionError::DeadlineExceeded)
    ));
}

#[test]
fn should_return_the_embeddings_result_channel_only_for_an_active_embeddings_request() {
    let (embeddings_result_sender, mut embeddings_result_receiver) = mpsc::channel(1);
    let active_request = active_embeddings_request(embeddings_result_sender);

    let extracted_embeddings_sender = active_request.embeddings_result_sender();
    assert!(
        extracted_embeddings_sender.is_some(),
        "an embeddings request must expose its result channel"
    );
    assert!(
        active_request.chat_stream_event_sender().is_none(),
        "an embeddings request has no chat stream channel"
    );
    assert!(
        active_request.image_result_sender().is_none(),
        "an embeddings request has no image result channel"
    );

    let send_outcome = extracted_embeddings_sender
        .expect("checked above")
        .try_send(Err(EmbeddingsExecutionError::WorkerUnavailable));
    assert!(send_outcome.is_ok());
    let received_outcome = embeddings_result_receiver
        .try_recv()
        .expect("the extracted clone addresses the original channel");
    assert!(matches!(
        received_outcome,
        Err(EmbeddingsExecutionError::WorkerUnavailable)
    ));
}
