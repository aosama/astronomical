//! Daemon IPC speech journeys: routing SpeakGenerate requests to the worker
//! executor, streaming the audio in bounded frames, and refusing what cannot
//! be spoken before dispatch.

use std::sync::Arc;

use astronomical_daemon::{SpeechExecutionError, SpeechOutput};
use ipc_protocol::{
    DaemonRequest, DaemonResponse, MAXIMUM_SPEECH_AUDIO_FRAME_BYTES, SpeakGenerateRequest,
    SpeechResultMetadata,
};
use speech_contract::SpeechFailure;
use tokio::time::timeout;

use crate::integration::daemon_ipc_exchange::connect_and_send_request;
use crate::integration::daemon_ipc_service_fixtures::{
    shut_down_and_remove_state_directory, start_service_with_installed_models,
};
use crate::support::daemon_ipc::{
    HANDSHAKE_TEST_TIMEOUT, StubGenerationExecutor, fresh_instance_state_directory,
    ipc_chat_discovered_model, ipc_speech_discovered_model, ready_stub_executor_with_speech,
};

const SPEECH_MODEL_ID: &str = "Kokoro-82M-bf16";
const CHAT_MODEL_ID: &str = "test-local-chatter";
const ROMEO_LINE: &str = "But soft, what light through yonder window breaks?";

fn speak_request(voice_name: Option<&str>) -> DaemonRequest {
    DaemonRequest::SpeakGenerate(SpeakGenerateRequest {
        model: None,
        text: ROMEO_LINE.to_owned(),
        voice_name: voice_name.map(str::to_owned),
        speed: 1.0,
        seed: Some(42),
    })
}

fn spoken_output(sample_count: usize) -> SpeechOutput {
    SpeechOutput {
        pcm16_bytes: vec![0x10; sample_count * 2],
        result_metadata: SpeechResultMetadata {
            sample_rate_hertz: 24_000,
            sample_count: sample_count as u64,
            segment_count: 1,
            cut_short_segment_count: 0,
            voice_name: "af_heart".to_owned(),
            seed: 42,
        },
    }
}

/// Sends one speak request and reads every frame until the terminal one.
async fn speak_once(
    test_name: &str,
    stub_executor: Arc<StubGenerationExecutor>,
    discovered_models: Vec<config::DiscoveredModel>,
    speak_request: DaemonRequest,
) -> Vec<DaemonResponse> {
    let state_directory = fresh_instance_state_directory(test_name);
    let daemon_ipc_service = start_service_with_installed_models(
        &state_directory,
        stub_executor,
        discovered_models,
        &[CHAT_MODEL_ID],
    )
    .await;
    let mut daemon_client =
        connect_and_send_request(daemon_ipc_service.socket_path(), &speak_request).await;
    let mut speech_frames = Vec::new();
    while let Ok(Ok(Some(daemon_response))) =
        timeout(HANDSHAKE_TEST_TIMEOUT, daemon_client.next_response()).await
    {
        speech_frames.push(daemon_response);
    }
    shut_down_and_remove_state_directory(daemon_ipc_service, &state_directory).await;
    speech_frames
}

fn installed_models() -> Vec<config::DiscoveredModel> {
    vec![
        ipc_chat_discovered_model(CHAT_MODEL_ID),
        ipc_speech_discovered_model(SPEECH_MODEL_ID),
    ]
}

// Positive and edge: audio larger than one frame arrives in bounded frames,
// in order, before the terminal frame names the model that spoke it.
#[tokio::test]
async fn should_stream_long_audio_in_bounded_frames_then_complete() {
    let sample_count = MAXIMUM_SPEECH_AUDIO_FRAME_BYTES; // two frames of bytes, plus a part
    let stub_executor =
        ready_stub_executor_with_speech(CHAT_MODEL_ID, Ok(spoken_output(sample_count)));
    let speech_frames = speak_once(
        "sp-frames",
        Arc::clone(&stub_executor),
        installed_models(),
        speak_request(None),
    )
    .await;

    let (terminal_frame, audio_frames) = speech_frames
        .split_last()
        .expect("the daemon should answer");
    assert!(matches!(
        terminal_frame,
        DaemonResponse::SpeechCompleted { model, .. } if model == SPEECH_MODEL_ID
    ));
    assert_eq!(audio_frames.len(), 2);
    let mut joined_audio_bytes = 0;
    for audio_frame in audio_frames {
        let DaemonResponse::SpeechAudio { pcm16_bytes } = audio_frame else {
            panic!("only audio frames precede the terminal frame");
        };
        assert!(pcm16_bytes.len() <= MAXIMUM_SPEECH_AUDIO_FRAME_BYTES);
        joined_audio_bytes += pcm16_bytes.len();
    }
    assert_eq!(joined_audio_bytes, sample_count * 2);
    assert_eq!(stub_executor.logged_speech_commands()[0].seed, 42);
}

// Negative: a voice the model lacks is refused before the worker is asked.
#[tokio::test]
async fn should_refuse_an_unknown_voice_before_dispatch() {
    let stub_executor = ready_stub_executor_with_speech(CHAT_MODEL_ID, Ok(spoken_output(2)));
    let speech_frames = speak_once(
        "sp-voice",
        Arc::clone(&stub_executor),
        installed_models(),
        speak_request(Some("alloy")),
    )
    .await;

    assert!(matches!(
        speech_frames.as_slice(),
        [DaemonResponse::SpeechFailed {
            reason: SpeechFailure::InvalidRequest { .. }
        }]
    ));
    assert!(stub_executor.logged_speech_commands().is_empty());
}

// Negative and edge: with no speech model installed the daemon declines;
// a worker failure after admission is reported as the request's failure.
#[tokio::test]
async fn should_decline_without_a_speech_model_and_report_worker_failures() {
    let speech_frames = speak_once(
        "sp-none",
        ready_stub_executor_with_speech(CHAT_MODEL_ID, Ok(spoken_output(2))),
        vec![ipc_chat_discovered_model(CHAT_MODEL_ID)],
        speak_request(None),
    )
    .await;
    assert!(matches!(
        speech_frames.as_slice(),
        [DaemonResponse::GenerationRejected { reason }] if reason.contains("no speech model")
    ));

    let failed_frames = speak_once(
        "sp-fail",
        ready_stub_executor_with_speech(
            CHAT_MODEL_ID,
            Err(SpeechExecutionError::WorkerFailure(
                SpeechFailure::fatal_execution("the model failed"),
            )),
        ),
        installed_models(),
        speak_request(None),
    )
    .await;
    assert!(matches!(
        failed_frames.as_slice(),
        [DaemonResponse::SpeechFailed {
            reason: SpeechFailure::FatalExecution { .. }
        }]
    ));
}
