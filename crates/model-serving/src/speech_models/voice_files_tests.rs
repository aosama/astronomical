use std::path::Path;

use super::{catalog_voice_names, catalog_voice_path, is_plain_voice_name};

#[test]
fn should_place_a_voice_file_in_its_folder() {
    let voice_path = catalog_voice_path(Path::new("voices"), "af_heart").expect("a plain name");
    assert_eq!(voice_path, Path::new("voices/af_heart.safetensors"));
}

#[test]
fn should_refuse_a_voice_name_that_names_a_path() {
    for unsafe_name in [
        "../config",
        "voices/af_heart",
        "",
        "af heart",
        "af_heart.safetensors",
    ] {
        assert!(!is_plain_voice_name(unsafe_name), "{unsafe_name:?}");
        assert!(
            catalog_voice_path(Path::new("voices"), unsafe_name).is_err(),
            "{unsafe_name:?}"
        );
    }
}

// Edge: a file a name could not request is not listed either.
#[test]
fn should_list_only_voices_a_request_can_name_sorted() {
    let voices_directory = tempfile::tempdir().expect("a temporary directory");
    for file_name in [
        "zoey-1.safetensors",
        "af_heart.safetensors",
        "my voice.safetensors",
        "af_heart.pt",
        "notes.txt",
    ] {
        std::fs::write(voices_directory.path().join(file_name), b"").expect("a voice file");
    }
    assert_eq!(
        catalog_voice_names(voices_directory.path()),
        ["af_heart", "zoey-1"]
    );
}

// Negative: a missing folder offers no voices.
#[test]
fn should_list_no_voices_for_a_missing_folder() {
    assert!(catalog_voice_names(Path::new("no/such/voices")).is_empty());
}
