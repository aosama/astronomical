//! Acceptance coverage for the admission every request kind shares at the
//! worker loop, driven through the real worker process protocol: a speech
//! request swaps its model in, is refused when the resident model cannot
//! speak, and a refusal leaves the worker ready for the next request.

use std::time::Duration;

use astronomical_daemon::{
    ChatGenerationExecutor, GenerationStartError, ImageGenerationExecutor, SpeechExecutionError,
    SpeechOutput, WorkerActivity, WorkerHandle,
};
use ipc_protocol::{RequestId, SpeechCommand};
use tokio::sync::mpsc;
use tokio::time::timeout;

use super::worker_model_swap::{
    SPEECH_MODEL_ID, TELEMETRY_BEFORE_SWAP_MODEL_ID, launch_idle_worker_fixture,
};

/// Two scripted segments of four 16-bit samples each.
const SCRIPTED_SPEECH_PCM16_BYTE_COUNT: usize = 16;
const SPEECH_RESULT_TIMEOUT: Duration = Duration::from_secs(2);

// Positive: a speech request naming an installed speech model swaps it in,
// runs to completion, and leaves the worker idle with that model resident.
#[tokio::test]
async fn should_swap_in_a_speech_model_and_speak_to_completion() {
    let worker_handle = launch_idle_worker_fixture().await;

    let mut speech_results = worker_handle
        .start_speech_generation(speech_command(SPEECH_MODEL_ID, 1, 1.0))
        .await
        .expect("the speech request should be admitted");
    let spoken_output = receive_speech_output(&mut speech_results).await;

    assert_eq!(
        spoken_output.pcm16_bytes.len(),
        SCRIPTED_SPEECH_PCM16_BYTE_COUNT
    );
    assert_eq!(spoken_output.result_metadata.segment_count, 2);
    let health_snapshot = worker_handle.worker_health_snapshot();
    assert_eq!(
        health_snapshot.ready_model_id.as_deref(),
        Some(SPEECH_MODEL_ID)
    );
    assert_eq!(health_snapshot.activity, WorkerActivity::Idle);
    shut_down(worker_handle).await;
}

// Negative: a resident model without a speech surface refuses the request
// before dispatch, and the refusal does not take the worker down.
#[tokio::test]
async fn should_refuse_speech_when_the_resident_model_cannot_speak() {
    let worker_handle = launch_idle_worker_fixture().await;

    let refusal = worker_handle
        .start_speech_generation(speech_command(TELEMETRY_BEFORE_SWAP_MODEL_ID, 2, 1.0))
        .await
        .expect_err("a chat model should not be asked to speak");

    assert_eq!(refusal, GenerationStartError::WorkerUnavailable);
    let mut speech_results = worker_handle
        .start_speech_generation(speech_command(SPEECH_MODEL_ID, 3, 1.0))
        .await
        .expect("the worker should stay ready after a refusal");
    receive_speech_output(&mut speech_results).await;
    shut_down(worker_handle).await;
}

// Edge: a model the catalog does not know is refused without a swap, and the
// slowest pace the contract accepts is admitted under its larger audio bound.
#[tokio::test]
async fn should_refuse_an_unknown_model_and_admit_the_slowest_pace() {
    let worker_handle = launch_idle_worker_fixture().await;

    let refusal = worker_handle
        .start_speech_generation(speech_command("astronomical/not-installed", 4, 1.0))
        .await
        .expect_err("an unknown model should be refused");

    assert_eq!(refusal, GenerationStartError::WorkerUnavailable);
    let mut speech_results = worker_handle
        .start_speech_generation(speech_command(
            SPEECH_MODEL_ID,
            5,
            speech_contract::SLOWEST_SPEED,
        ))
        .await
        .expect("the slowest pace should be admitted");
    let spoken_output = receive_speech_output(&mut speech_results).await;
    assert_eq!(
        spoken_output.pcm16_bytes.len(),
        SCRIPTED_SPEECH_PCM16_BYTE_COUNT
    );
    shut_down(worker_handle).await;
}

fn speech_command(model_id: &str, request_id: u64, speed: f32) -> SpeechCommand {
    SpeechCommand {
        request_id: RequestId::new(request_id),
        model: model_id.to_owned(),
        text: "Now is the winter of our discontent".to_owned(),
        voice_name: None,
        speed,
        seed: 7,
    }
}

async fn receive_speech_output(
    speech_results: &mut mpsc::Receiver<Result<SpeechOutput, SpeechExecutionError>>,
) -> SpeechOutput {
    timeout(SPEECH_RESULT_TIMEOUT, speech_results.recv())
        .await
        .expect("the speech result should arrive before the timeout")
        .expect("the speech result channel should carry one result")
        .expect("the scripted speech should succeed")
}

async fn shut_down(worker_handle: WorkerHandle) {
    worker_handle
        .shutdown()
        .await
        .expect("worker should shut down");
}
