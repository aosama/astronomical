use std::path::Path;

use super::{voice_file_path, voice_names};

#[test]
fn should_map_the_default_voice_to_the_shipped_file() {
    let default_path = voice_file_path(Path::new("model"), "default").expect("default");
    assert_eq!(default_path, Path::new("model/default-voice.safetensors"));
}

#[test]
fn should_place_a_catalog_voice_in_the_voices_folder() {
    let voice_path =
        voice_file_path(Path::new("model"), "zoey-NbpkqMVS3CJeq2j8").expect("a plain name");
    assert_eq!(
        voice_path,
        Path::new("model/voices/zoey-NbpkqMVS3CJeq2j8.safetensors")
    );
}

#[test]
fn should_list_the_default_voice_first_then_the_catalog() {
    let model_directory = tempfile::tempdir().expect("a temporary directory");
    std::fs::write(
        model_directory.path().join("default-voice.safetensors"),
        b"",
    )
    .expect("default voice");
    std::fs::create_dir(model_directory.path().join("voices")).expect("voices folder");
    for file_name in ["zoey.safetensors", "easton.safetensors"] {
        std::fs::write(model_directory.path().join("voices").join(file_name), b"")
            .expect("voice file");
    }
    assert_eq!(
        voice_names(model_directory.path()),
        ["default", "easton", "zoey"]
    );
}

// Negative: an empty artifact offers no voices.
#[test]
fn should_list_no_voices_for_an_empty_artifact() {
    let model_directory = tempfile::tempdir().expect("a temporary directory");
    assert!(voice_names(model_directory.path()).is_empty());
}
