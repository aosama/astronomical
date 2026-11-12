use std::path::Path;

use super::voice_file_path;

// Positive: a voice is a file in the voices folder.
#[test]
fn should_place_a_voice_in_the_voices_folder() {
    let voice_path = voice_file_path(Path::new("model"), "af_heart").expect("a plain name");
    assert_eq!(voice_path, Path::new("model/voices/af_heart.safetensors"));
}

// Negative: a name that is a path is refused.
#[test]
fn should_refuse_a_voice_name_that_is_a_path() {
    assert!(voice_file_path(Path::new("model"), "../config").is_err());
}
