//! Speech-model advertisement contracts: what `/v1/models` says about a
//! model that reads text aloud.

use crate::{OpenAiModel, OpenAiModelValidationError, OpenAiSpeechModelParts};

fn kokoro_parts(default_voice: &str) -> OpenAiSpeechModelParts {
    OpenAiSpeechModelParts {
        model_id: "Kokoro-82M-bf16".to_owned(),
        created: 1_784_231_803,
        owned_by: "astronomical".to_owned(),
        voices: vec!["af_heart".to_owned(), "am_michael".to_owned()],
        default_voice: default_voice.to_owned(),
        sample_rate_hertz: 24_000,
        supports_speed: true,
    }
}

// Positive: the speech endpoint, audio output, and the voices to ask for.
#[test]
fn should_advertise_a_speech_model_with_its_voices_and_formats() {
    let model = OpenAiModel::from_speech_parts(kokoro_parts("af_heart"))
        .expect("a valid speech model should be advertised");

    let serialized = serde_json::to_string(&model).expect("the speech model should serialize");
    assert!(serialized.contains(r#""supported_endpoints":["/v1/audio/speech"]"#));
    assert!(serialized.contains(r#""output_modalities":["audio"]"#));
    assert!(serialized.contains(r#""voices":["af_heart","am_michael"]"#));
    assert!(serialized.contains(r#""default_voice":"af_heart""#));
    assert!(serialized.contains(r#""sample_rate_hertz":24000"#));
    assert!(serialized.contains(r#""response_formats":["wav","pcm"]"#));
    assert!(
        !serialized.contains("context_window") && !serialized.contains("max_input_tokens"),
        "a speech model has no token limits: {serialized}"
    );
}

// Negative: a default voice the model does not have cannot be advertised.
#[test]
fn should_refuse_a_default_voice_outside_the_voices() {
    assert_eq!(
        OpenAiModel::from_speech_parts(kokoro_parts("alloy")),
        Err(OpenAiModelValidationError::SpeechDefaultVoiceMissing)
    );
}
