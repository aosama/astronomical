//! Unit contracts for what the speech endpoint refuses before a model swap
//! and for the audio response it builds.

use astronomical_config::{ModelFamily, SpeechModelCapabilities};
use astronomical_ipc_protocol::SpeechResultMetadata;
use astronomical_rest_contract::OpenAiSpeechRequestParts;
use astronomical_speech_contract::AudioFormat;
use axum::http::StatusCode;

use super::{audio_response, refusal_before_swap};
use crate::SpeechOutput;

fn phonon_capabilities() -> SpeechModelCapabilities {
    SpeechModelCapabilities {
        model_family: ModelFamily::Phonon,
        sample_rate_hertz: 24_000,
        voice_names: vec!["default".to_owned()],
        default_voice_name: "default".to_owned(),
        supports_speed: false,
    }
}

fn romeo_parts(voice: Option<&str>, speed: f32) -> OpenAiSpeechRequestParts {
    OpenAiSpeechRequestParts {
        model: None,
        input: "But soft, what light through yonder window breaks?".to_owned(),
        voice: voice.map(str::to_owned),
        speed,
        response_format: AudioFormat::Wav,
        seed: None,
    }
}

fn spoken_output() -> SpeechOutput {
    SpeechOutput {
        pcm16_bytes: vec![0, 0, 0, 64],
        result_metadata: SpeechResultMetadata {
            sample_rate_hertz: 24_000,
            sample_count: 2,
            segment_count: 1,
            cut_short_segment_count: 0,
            voice_name: "default".to_owned(),
            seed: 42,
        },
    }
}

// Positive: a request the model can speak passes, as does one for a model
// discovery knows nothing more about.
#[test]
fn should_pass_a_request_the_model_can_speak() {
    assert!(refusal_before_swap(Some(&phonon_capabilities()), &romeo_parts(None, 1.0)).is_none());
    assert!(
        refusal_before_swap(
            Some(&phonon_capabilities()),
            &romeo_parts(Some("default"), 1.0)
        )
        .is_none()
    );
    assert!(refusal_before_swap(None, &romeo_parts(Some("alloy"), 2.0)).is_none());
}

// Negative: an unknown voice or a pace the model cannot change is refused
// before the model is made resident.
#[test]
fn should_refuse_an_unknown_voice_or_an_unsupported_pace() {
    let unknown_voice = refusal_before_swap(
        Some(&phonon_capabilities()),
        &romeo_parts(Some("alloy"), 1.0),
    )
    .expect("alloy is not a Phonon voice");
    assert_eq!(unknown_voice.status(), StatusCode::BAD_REQUEST);
    let faster = refusal_before_swap(Some(&phonon_capabilities()), &romeo_parts(None, 1.5))
        .expect("Phonon speaks only at its natural pace");
    assert_eq!(faster.status(), StatusCode::BAD_REQUEST);
}

// Positive: WAV wraps the samples, and the headers say how they were spoken.
#[test]
fn should_answer_with_wav_and_describe_it_in_headers() {
    let wav_response = audio_response(spoken_output(), AudioFormat::Wav);
    assert_eq!(wav_response.status(), StatusCode::OK);
    let response_headers = wav_response.headers();
    assert_eq!(response_headers["content-type"], "audio/wav");
    assert_eq!(response_headers["x-astronomical-seed"], "42");
    assert_eq!(response_headers["x-astronomical-sample-rate"], "24000");
    assert_eq!(response_headers["x-astronomical-voice"], "default");
    assert_eq!(response_headers["x-astronomical-segments-cut-short"], "0");
}

// Edge: PCM is the bare samples, with the sample rate a player needs.
#[tokio::test]
async fn should_answer_with_bare_samples_for_pcm() {
    let pcm_response = audio_response(spoken_output(), AudioFormat::Pcm);
    assert_eq!(pcm_response.headers()["content-type"], "audio/pcm");
    let body_bytes = axum::body::to_bytes(pcm_response.into_body(), 1024)
        .await
        .expect("the body is readable");
    assert_eq!(body_bytes.as_ref(), [0, 0, 0, 64]);
}
