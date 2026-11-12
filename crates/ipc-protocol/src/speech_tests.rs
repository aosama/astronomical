use astronomical_speech_contract::SpeechFailure;

use super::{SpeechCommand, SpokenAudioSegment, WorkerSpeechCapabilities};
use crate::{RequestId, WorkerModelCapabilitiesValidationError};

fn romeo_command() -> SpeechCommand {
    SpeechCommand {
        request_id: RequestId::new(7),
        model: "Kokoro-82M-bf16".to_owned(),
        text: "But soft, what light through yonder window breaks?".to_owned(),
        voice_name: Some("af_heart".to_owned()),
        speed: 1.0,
        seed: 42,
    }
}

fn kokoro_capabilities() -> WorkerSpeechCapabilities {
    WorkerSpeechCapabilities {
        sample_rate_hertz: 24_000,
        voice_names: vec!["af_heart".to_owned(), "am_michael".to_owned()],
        default_voice_name: "af_heart".to_owned(),
        supports_speed: true,
    }
}

// Positive: a well-formed command becomes the engine's request unchanged.
#[test]
fn should_turn_a_valid_command_into_its_request() {
    let speech_request = romeo_command()
        .validated_request()
        .expect("a valid command");
    assert_eq!(speech_request.text, romeo_command().text);
    assert_eq!(speech_request.voice_name.as_deref(), Some("af_heart"));
    assert_eq!(speech_request.seed, 42);
}

// Negative: the worker re-checks what the daemon already checked.
#[test]
fn should_refuse_an_empty_model_blank_text_or_an_impossible_speed() {
    let mut empty_model = romeo_command();
    empty_model.model = " ".to_owned();
    let mut blank_text = romeo_command();
    blank_text.text = "  ".to_owned();
    let mut too_fast = romeo_command();
    too_fast.speed = 9.0;
    for invalid_command in [empty_model, blank_text, too_fast] {
        assert!(matches!(
            invalid_command.validated_request(),
            Err(SpeechFailure::InvalidRequest { .. })
        ));
    }
}

// Positive: audio crosses as base64 and reads back byte for byte.
#[test]
fn should_carry_audio_as_base64_and_read_it_back() {
    let audio_segment = SpokenAudioSegment {
        pcm16_bytes: vec![0x00, 0x40, 0xff, 0x7f],
    };
    let segment_json = serde_json::to_string(&audio_segment).expect("a segment serialises");
    assert_eq!(segment_json, r#"{"pcm16_bytes":"AED/fw=="}"#);
    let read_back: SpokenAudioSegment =
        serde_json::from_str(&segment_json).expect("and reads back");
    assert_eq!(read_back, audio_segment);
}

// Edge: an odd byte count is half a sample.
#[test]
fn should_tell_whole_samples_from_a_partial_one() {
    assert!(
        SpokenAudioSegment {
            pcm16_bytes: vec![]
        }
        .has_whole_samples()
    );
    assert!(
        !SpokenAudioSegment {
            pcm16_bytes: vec![1, 2, 3]
        }
        .has_whole_samples()
    );
}

// Positive and negative: an advertisement needs a rate and its default voice.
#[test]
fn should_refuse_an_advertisement_no_request_could_use() {
    assert_eq!(kokoro_capabilities().validate(), Ok(()));
    let mut silent = kokoro_capabilities();
    silent.sample_rate_hertz = 0;
    assert_eq!(
        silent.validate(),
        Err(WorkerModelCapabilitiesValidationError::ZeroSpeechSampleRate)
    );
    let mut voiceless = kokoro_capabilities();
    voiceless.default_voice_name = "alloy".to_owned();
    assert_eq!(
        voiceless.validate(),
        Err(WorkerModelCapabilitiesValidationError::MissingSpeechVoice)
    );
}
