use astronomical_speech_contract::AudioFormat;

use super::{OpenAiSpeechRequest, OpenAiSpeechValidationError};

fn parse(
    request_json: &str,
) -> Result<super::OpenAiSpeechRequestParts, OpenAiSpeechValidationError> {
    serde_json::from_str::<OpenAiSpeechRequest>(request_json)
        .expect("the request is JSON")
        .into_parts()
}

// Positive: only `input` is required; everything else has a default.
#[test]
fn should_accept_input_alone_with_defaults() {
    let request_parts = parse(r#"{"input": "But soft, what light through yonder window breaks?"}"#)
        .expect("a valid request");
    assert_eq!(request_parts.model, None);
    assert_eq!(request_parts.voice, None);
    assert_eq!(request_parts.speed, 1.0);
    assert_eq!(request_parts.response_format, AudioFormat::Wav);
    assert_eq!(request_parts.seed, None);
}

// Positive: every OpenAI field we support, plus a seed.
#[test]
fn should_read_every_supported_field() {
    let request_parts = parse(
        r#"{"model": "Kokoro-82M-bf16", "input": "It is the east, and Juliet is the sun.",
            "voice": "af_heart", "speed": 1.25, "response_format": "pcm", "seed": 42}"#,
    )
    .expect("a valid request");
    assert_eq!(request_parts.model.as_deref(), Some("Kokoro-82M-bf16"));
    assert_eq!(request_parts.voice.as_deref(), Some("af_heart"));
    assert_eq!(request_parts.speed, 1.25);
    assert_eq!(request_parts.response_format, AudioFormat::Pcm);
    assert_eq!(request_parts.seed, Some(42));
}

// Negative: formats that would need an encoder, unknown fields, blank text.
// An unknown field has no `param`; its message names it.
#[test]
fn should_refuse_what_the_endpoint_cannot_honour() {
    let refused_requests = [
        (
            r#"{"input": "Arise, fair sun", "response_format": "mp3"}"#,
            Some("response_format"),
        ),
        (
            r#"{"input": "Arise, fair sun", "voice_style": "whisper"}"#,
            None,
        ),
        (r#"{"input": "   "}"#, Some("input")),
        (
            r#"{"input": "Arise, fair sun", "voice": " "}"#,
            Some("voice"),
        ),
        (
            r#"{"input": "Arise, fair sun", "speed": 9.0}"#,
            Some("speed"),
        ),
    ];
    for (request_json, refused_parameter) in refused_requests {
        let validation_error = parse(request_json).expect_err(request_json);
        assert_eq!(
            validation_error.parameter_name(),
            refused_parameter,
            "{request_json}"
        );
    }
    let unknown_field_error = parse(r#"{"input": "Arise", "voice_style": "whisper"}"#)
        .expect_err("an unknown field is refused");
    assert!(unknown_field_error.to_string().contains("voice_style"));
}

// Edge: the longest input is accepted and one more character is not; the
// slowest and fastest paces are in range.
#[test]
fn should_hold_the_input_and_speed_limits_at_their_edges() {
    let longest_input = "a".repeat(4096);
    assert!(parse(&format!(r#"{{"input": "{longest_input}"}}"#)).is_ok());
    assert!(matches!(
        parse(&format!(r#"{{"input": "{longest_input}a"}}"#)),
        Err(OpenAiSpeechValidationError::InputTooLong {
            character_count: 4097
        })
    ));
    for edge_speed in ["0.25", "4.0"] {
        assert!(
            parse(&format!(
                r#"{{"input": "Wherefore", "speed": {edge_speed}}}"#
            ))
            .is_ok()
        );
    }
}

// Positive and negative: OpenAI's `instructions` is accepted and unused;
// `stream_format` may only ask for the audio itself.
#[test]
fn should_accept_instructions_and_only_the_audio_stream_format() {
    assert!(parse(r#"{"input": "Arise, fair sun", "instructions": "speak softly"}"#).is_ok());
    assert!(parse(r#"{"input": "Arise, fair sun", "stream_format": "audio"}"#).is_ok());
    let refused = parse(r#"{"input": "Arise, fair sun", "stream_format": "sse"}"#)
        .expect_err("server-sent events are not offered");
    assert_eq!(refused.parameter_name(), Some("stream_format"));
}
