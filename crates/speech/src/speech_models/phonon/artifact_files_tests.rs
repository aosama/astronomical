use std::path::Path;

use super::voice_file_path;

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

// Negative: a catalog name that is a path is refused.
#[test]
fn should_refuse_a_catalog_voice_name_that_is_a_path() {
    assert!(voice_file_path(Path::new("model"), "../model").is_err());
}
