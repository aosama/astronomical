use std::path::Path;

use super::catalog_voice_path;

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
        assert!(
            catalog_voice_path(Path::new("voices"), unsafe_name).is_err(),
            "{unsafe_name:?}"
        );
    }
}
