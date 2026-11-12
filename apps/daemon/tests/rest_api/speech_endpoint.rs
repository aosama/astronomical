//! Acceptance coverage for the speech HTTP journey, `POST /v1/audio/speech`.

use std::path::PathBuf;

use astronomical_config::{
    DiscoveredModel, ModelCapabilities, ModelFamily, ModelLicense, SpeechModelCapabilities,
};
use astronomical_daemon::SpeechExecutionError;
use astronomical_speech_contract::SpeechFailure;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt;

use crate::support::ScriptedExecutor;

const KOKORO_MODEL_ID: &str = "Kokoro-82M-bf16";
const PHONON_MODEL_ID: &str = "phonon-31m-en";
const ROMEO_LINE: &str = "But soft, what light through yonder window breaks?";

fn speech_model(
    model_id: &str,
    model_family: ModelFamily,
    supports_speed: bool,
) -> DiscoveredModel {
    let default_voice_name = if supports_speed {
        "af_heart"
    } else {
        "default"
    };
    DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: None,
        model_family,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/speech"),
        capabilities: ModelCapabilities::Speech(SpeechModelCapabilities {
            model_family,
            sample_rate_hertz: 24_000,
            voice_names: vec![default_voice_name.to_owned()],
            default_voice_name: default_voice_name.to_owned(),
            supports_speed,
        }),
        license: Some(ModelLicense::Apache20),
        model_size_bytes: 1,
    }
}

fn installed_speech_models() -> Vec<DiscoveredModel> {
    vec![
        speech_model(PHONON_MODEL_ID, ModelFamily::Phonon, false),
        speech_model(KOKORO_MODEL_ID, ModelFamily::Kokoro, true),
    ]
}

fn speech_request(request_document: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/audio/speech")
        .header("content-type", "application/json")
        .body(Body::from(request_document.to_string()))
        .expect("the speech request should be valid")
}

async fn response_bytes(response: axum::response::Response) -> Vec<u8> {
    to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("the bounded response body should be readable")
        .to_vec()
}

// Positive: with only `input`, the default model speaks and a WAV comes back
// with how it was spoken in headers.
#[tokio::test]
async fn should_speak_input_alone_as_wav_with_the_default_model() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_speech_commands();
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        installed_speech_models(),
    );

    let response = application
        .oneshot(speech_request(serde_json::json!({ "input": ROMEO_LINE })))
        .await
        .expect("the application should answer");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "audio/wav");
    assert_eq!(response.headers()["x-astronomical-seed"], "42");
    assert_eq!(response.headers()["x-astronomical-sample-rate"], "24000");
    let wav_bytes = response_bytes(response).await;
    assert_eq!(&wav_bytes[..4], b"RIFF");
    assert_eq!(wav_bytes.len(), 44 + 4);
    let commands = received_commands
        .lock()
        .expect("the speech command log should remain available");
    assert_eq!(commands.len(), 1);
    assert_eq!(commands[0].model, KOKORO_MODEL_ID);
    assert_eq!(commands[0].text, ROMEO_LINE);
    assert_eq!(commands[0].speed, 1.0);
}

// Positive: a named model, voice, pace, seed and PCM format reach the worker
// and come back as bare samples.
#[tokio::test]
async fn should_pass_every_field_through_and_answer_pcm() {
    let executor = ScriptedExecutor::ready(Vec::new());
    let received_commands = executor.received_speech_commands();
    let application = astronomical_daemon::build_application_with_discovered_models(
        executor,
        installed_speech_models(),
    );

    let response = application
        .oneshot(speech_request(serde_json::json!({
            "model": KOKORO_MODEL_ID, "input": ROMEO_LINE, "voice": "af_heart",
            "speed": 1.5, "response_format": "pcm", "seed": 7,
        })))
        .await
        .expect("the application should answer");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "audio/pcm");
    assert_eq!(response_bytes(response).await, [0x00, 0x00, 0x00, 0x40]);
    let commands = received_commands
        .lock()
        .expect("the speech command log should remain available");
    assert_eq!(commands[0].voice_name.as_deref(), Some("af_heart"));
    assert_eq!(commands[0].speed, 1.5);
    assert_eq!(commands[0].seed, 7);
}

// Negative: an encoder format, an unknown voice and a pace Phonon cannot
// change are refused before any worker command exists.
#[tokio::test]
async fn should_refuse_what_cannot_be_spoken_before_reaching_the_worker() {
    let refused_documents = [
        serde_json::json!({ "input": ROMEO_LINE, "response_format": "mp3" }),
        serde_json::json!({ "input": ROMEO_LINE, "voice": "alloy" }),
        serde_json::json!({ "model": PHONON_MODEL_ID, "input": ROMEO_LINE, "speed": 1.5 }),
        serde_json::json!({ "input": "" }),
    ];
    for refused_document in refused_documents {
        let executor = ScriptedExecutor::ready(Vec::new());
        let received_commands = executor.received_speech_commands();
        let application = astronomical_daemon::build_application_with_discovered_models(
            executor,
            installed_speech_models(),
        );
        let response = application
            .oneshot(speech_request(refused_document.clone()))
            .await
            .expect("the application should answer");
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "{refused_document}"
        );
        assert!(
            received_commands
                .lock()
                .expect("the speech command log should remain available")
                .is_empty(),
            "{refused_document}"
        );
    }
}

// Negative: with no speech model installed the caller is told to download one.
#[tokio::test]
async fn should_say_when_no_speech_model_is_installed() {
    let application = astronomical_daemon::build_application_with_discovered_models(
        ScriptedExecutor::ready(Vec::new()),
        Vec::new(),
    );

    let response = application
        .oneshot(speech_request(serde_json::json!({ "input": ROMEO_LINE })))
        .await
        .expect("the application should answer");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error_text = String::from_utf8(response_bytes(response).await).expect("UTF-8");
    assert!(error_text.contains("no_speech_model"), "{error_text}");
}

// Edge: the worker's own failures map to the caller's and the server's codes.
#[tokio::test]
async fn should_map_worker_failures_to_their_status_codes() {
    let failure_statuses = [
        (
            SpeechExecutionError::WorkerFailure(SpeechFailure::invalid_request("no voice")),
            StatusCode::BAD_REQUEST,
        ),
        (
            SpeechExecutionError::WorkerFailure(SpeechFailure::fatal_execution("model failed")),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            SpeechExecutionError::DeadlineExceeded,
            StatusCode::GATEWAY_TIMEOUT,
        ),
    ];
    for (speech_error, expected_status) in failure_statuses {
        let mut executor = ScriptedExecutor::ready(Vec::new());
        executor.speech_outcome = Err(speech_error.clone());
        let application = astronomical_daemon::build_application_with_discovered_models(
            executor,
            installed_speech_models(),
        );
        let response = application
            .oneshot(speech_request(serde_json::json!({ "input": ROMEO_LINE })))
            .await
            .expect("the application should answer");
        assert_eq!(response.status(), expected_status, "{speech_error:?}");
    }
}
