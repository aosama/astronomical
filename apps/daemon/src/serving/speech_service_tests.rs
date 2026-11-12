//! Unit contracts for choosing the model that speaks a request and for the
//! command the worker receives.

use std::path::PathBuf;

use astronomical_config::{
    BUILTIN_DEFAULT_SPEECH_MODEL_ID, ChatModelCapabilities, DiscoveredModel, ModelCapabilities,
    ModelFamily, SpeechModelCapabilities,
};

use super::{
    SpeechAsked, SpeechModelResolutionError, VoiceOrPaceRefusal, build_speech_command,
    refuse_voice_or_pace_before_swap, resolve_speech_model, speech_capabilities_of,
};

fn speech_model(model_id: &str) -> DiscoveredModel {
    DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: None,
        model_family: ModelFamily::Kokoro,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/speech"),
        capabilities: ModelCapabilities::Speech(SpeechModelCapabilities {
            model_family: ModelFamily::Kokoro,
            sample_rate_hertz: 24_000,
            voice_names: vec!["af_heart".to_owned()],
            default_voice_name: "af_heart".to_owned(),
            supports_speed: true,
        }),
        license: None,
        model_size_bytes: 1,
    }
}

fn chat_model(model_id: &str) -> DiscoveredModel {
    DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: None,
        model_family: ModelFamily::Qwen3_5,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/chat"),
        capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
            context_window: 2_048,
            max_input_tokens: 1_024,
            max_output_tokens: 128,
            supports_vision: false,
            supports_reasoning: false,
            supports_tool_calls: false,
        }),
        license: None,
        model_size_bytes: 1,
    }
}

fn romeo_asked(seed: Option<u64>) -> SpeechAsked {
    SpeechAsked {
        text: "But soft, what light through yonder window breaks?".to_owned(),
        voice_name: None,
        speed: 1.0,
        seed,
    }
}

// Positive: with no name, the built-in default answers when installed.
#[test]
fn should_choose_the_built_in_default_speech_model() {
    let discovered_models = [
        speech_model("phonon-31m-en"),
        speech_model(BUILTIN_DEFAULT_SPEECH_MODEL_ID),
    ];
    assert_eq!(
        resolve_speech_model(&discovered_models, Some("chatter"), None),
        Ok(BUILTIN_DEFAULT_SPEECH_MODEL_ID.to_owned())
    );
}

// Positive: a resident speech model keeps answering without a swap.
#[test]
fn should_keep_the_resident_speech_model() {
    let discovered_models = [
        speech_model("phonon-31m-en"),
        speech_model(BUILTIN_DEFAULT_SPEECH_MODEL_ID),
    ];
    assert_eq!(
        resolve_speech_model(&discovered_models, Some("phonon-31m-en"), None),
        Ok("phonon-31m-en".to_owned())
    );
}

// Negative: a named chat model cannot speak, and nothing installed speaks.
#[test]
fn should_refuse_a_model_that_cannot_speak_and_an_install_without_one() {
    let discovered_models = [chat_model("chatter")];
    assert_eq!(
        resolve_speech_model(&discovered_models, None, Some("chatter")),
        Err(SpeechModelResolutionError::ModelCannotSpeak)
    );
    assert_eq!(
        resolve_speech_model(&discovered_models, None, None),
        Err(SpeechModelResolutionError::NoSpeechModelInstalled)
    );
}

// Edge: two non-default speech models with none resident are ambiguous.
#[test]
fn should_refuse_to_guess_between_two_speech_models() {
    let discovered_models = [speech_model("voice-b"), speech_model("voice-a")];
    assert_eq!(
        resolve_speech_model(&discovered_models, None, None),
        Err(SpeechModelResolutionError::SeveralSpeechModelsInstalled {
            installed_model_ids: vec!["voice-a".to_owned(), "voice-b".to_owned()],
        })
    );
}

// Positive and edge: a named seed is kept; an unnamed one is picked.
#[test]
fn should_keep_a_named_seed_and_pick_one_otherwise() {
    let seeded = build_speech_command(7, "Kokoro-82M-bf16", romeo_asked(Some(42)));
    assert_eq!(seeded.seed, 42);
    assert_eq!(seeded.request_id.value(), 7);
    let unseeded = build_speech_command(8, "Kokoro-82M-bf16", romeo_asked(None));
    assert_eq!(unseeded.model, "Kokoro-82M-bf16");
    assert_eq!(unseeded.voice_name, None);
}

// Positive and negative: capabilities are found only for a speech model.
#[test]
fn should_find_speech_capabilities_only_for_a_speech_model() {
    let discovered_models = [speech_model("Kokoro-82M-bf16"), chat_model("chatter")];
    assert!(speech_capabilities_of(&discovered_models, "Kokoro-82M-bf16").is_some());
    assert!(speech_capabilities_of(&discovered_models, "chatter").is_none());
    assert!(speech_capabilities_of(&discovered_models, "missing").is_none());
}

// Positive, negative and edge: a voice the model has and its natural pace
// pass; an unknown voice or a changed pace on a fixed-pace model do not; a
// model discovery knows nothing more about passes.
#[test]
fn should_refuse_only_what_the_model_cannot_speak() {
    let ModelCapabilities::Speech(mut speech_capabilities) = speech_model("kokoro").capabilities
    else {
        panic!("the fixture is a speech model");
    };
    assert_eq!(
        refuse_voice_or_pace_before_swap(Some(&speech_capabilities), Some("af_heart"), 1.5),
        Ok(())
    );
    assert!(matches!(
        refuse_voice_or_pace_before_swap(Some(&speech_capabilities), Some("alloy"), 1.0),
        Err(VoiceOrPaceRefusal::UnknownVoice { .. })
    ));
    speech_capabilities.supports_speed = false;
    assert_eq!(
        refuse_voice_or_pace_before_swap(Some(&speech_capabilities), None, 1.5),
        Err(VoiceOrPaceRefusal::FixedPace)
    );
    assert_eq!(
        refuse_voice_or_pace_before_swap(None, Some("alloy"), 2.0),
        Ok(())
    );
}
