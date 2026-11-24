use std::path::Path;

use super::{classifies_directory, kokoro_speech_capabilities};

const KOKORO_CONFIGURATION: &str = r#"{"istftnet": {"upsample_rates": [10, 6]}, "plbert": {"num_hidden_layers": 12}, "vocab": {"a": 43}}"#;

fn kokoro_folder(configuration: &str, voice_file_names: &[&str]) -> tempfile::TempDir {
    let model_directory = tempfile::tempdir().expect("a temporary directory");
    std::fs::write(model_directory.path().join("config.json"), configuration).expect("a config");
    std::fs::write(
        model_directory.path().join("kokoro-v1_0.safetensors"),
        b"weights",
    )
    .expect("weights");
    let voices_directory = model_directory.path().join("voices");
    std::fs::create_dir(&voices_directory).expect("a voices folder");
    for voice_file_name in voice_file_names {
        std::fs::write(voices_directory.join(voice_file_name), b"voice").expect("a voice");
    }
    model_directory
}

// Positive: the StyleTTS2 sections and the weights mark a Kokoro folder, and
// its American English voices are offered with af_heart as the default.
#[test]
fn should_classify_kokoro_and_offer_its_american_voices() {
    let model_directory = kokoro_folder(
        KOKORO_CONFIGURATION,
        &[
            "am_michael.safetensors",
            "af_heart.safetensors",
            "bf_emma.safetensors",
        ],
    );
    assert!(classifies_directory(model_directory.path()));
    let speech_capabilities =
        kokoro_speech_capabilities(model_directory.path()).expect("a voice to speak in");
    assert_eq!(speech_capabilities.voice_names, ["af_heart", "am_michael"]);
    assert_eq!(speech_capabilities.default_voice_name, "af_heart");
    assert_eq!(speech_capabilities.sample_rate_hertz, 24_000);
    assert!(speech_capabilities.supports_speed);
}

// Negative: a config without the StyleTTS2 sections is some other model.
#[test]
fn should_not_classify_a_folder_without_the_kokoro_sections() {
    let model_directory = kokoro_folder(r#"{"model_type": "qwen3_5", "vocab": {}}"#, &[]);
    assert!(!classifies_directory(model_directory.path()));
    assert!(!classifies_directory(Path::new("no/such/model")));
}

// Edge: without af_heart the first voice is the default, and a folder with
// only voices our phonemizer cannot serve offers nothing.
#[test]
fn should_fall_back_to_the_first_voice_and_offer_nothing_without_an_american_voice() {
    let without_heart = kokoro_folder(KOKORO_CONFIGURATION, &["am_michael.safetensors"]);
    assert_eq!(
        kokoro_speech_capabilities(without_heart.path())
            .expect("a voice")
            .default_voice_name,
        "am_michael"
    );
    let british_only = kokoro_folder(KOKORO_CONFIGURATION, &["bf_emma.safetensors"]);
    assert!(kokoro_speech_capabilities(british_only.path()).is_none());
}
