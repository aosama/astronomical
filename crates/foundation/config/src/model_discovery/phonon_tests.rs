use super::{classifies_directory, phonon_speech_capabilities};

const PHONON_CONFIGURATION: &str =
    r#"{"flow_lm": {"d_model": 512}, "mimi": {"sample_rate": 24000, "frame_rate": 12.5}}"#;

fn phonon_folder(configuration: &str, catalog_voice_file_names: &[&str]) -> tempfile::TempDir {
    let model_directory = tempfile::tempdir().expect("a temporary directory");
    std::fs::write(model_directory.path().join("config.json"), configuration).expect("a config");
    for file_name in [
        "model.safetensors",
        "tokenizer.json",
        "default-voice.safetensors",
    ] {
        std::fs::write(model_directory.path().join(file_name), b"artifact").expect("a file");
    }
    if !catalog_voice_file_names.is_empty() {
        let voices_directory = model_directory.path().join("voices");
        std::fs::create_dir(&voices_directory).expect("a voices folder");
        for voice_file_name in catalog_voice_file_names {
            std::fs::write(voices_directory.join(voice_file_name), b"voice").expect("a voice");
        }
    }
    model_directory
}

// Positive: the flow and Mimi sections mark a Phonon folder; its shipped
// voice comes first and is the default, and it speaks at its natural pace.
#[test]
fn should_classify_phonon_and_offer_its_shipped_voice_first() {
    let model_directory = phonon_folder(PHONON_CONFIGURATION, &["juliet.safetensors"]);
    assert!(classifies_directory(model_directory.path()));
    let speech_capabilities =
        phonon_speech_capabilities(model_directory.path()).expect("a voice to speak in");
    assert_eq!(speech_capabilities.voice_names, ["default", "juliet"]);
    assert_eq!(speech_capabilities.default_voice_name, "default");
    assert_eq!(speech_capabilities.sample_rate_hertz, 24_000);
    assert!(!speech_capabilities.supports_speed);
}

// Negative: without the flow model's section it is not Phonon.
#[test]
fn should_not_classify_a_folder_without_the_flow_model() {
    let model_directory = phonon_folder(r#"{"mimi": {"sample_rate": 24000}}"#, &[]);
    assert!(!classifies_directory(model_directory.path()));
}

// Edge: a zero sample rate could never be played, so the folder is refused.
#[test]
fn should_refuse_a_zero_sample_rate() {
    let model_directory = phonon_folder(r#"{"flow_lm": {}, "mimi": {"sample_rate": 0}}"#, &[]);
    assert!(!classifies_directory(model_directory.path()));
    assert!(phonon_speech_capabilities(model_directory.path()).is_none());
}
