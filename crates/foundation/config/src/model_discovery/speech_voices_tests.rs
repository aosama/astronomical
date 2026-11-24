use std::path::Path;

use super::catalog_voice_names;

// Positive and edge: only files a request could name are listed, sorted.
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
