use super::is_plain_voice_name;

#[test]
fn should_accept_letters_digits_hyphens_and_underscores() {
    for voice_name in ["af_heart", "zoey-1", "default", "A9"] {
        assert!(is_plain_voice_name(voice_name), "{voice_name:?}");
    }
}

// Negative: anything that could name a path, or nothing at all.
#[test]
fn should_refuse_a_name_that_could_reach_outside_the_voices_folder() {
    for voice_name in [
        "../config",
        "voices/af_heart",
        "af_heart.safetensors",
        "af heart",
        "~",
    ] {
        assert!(!is_plain_voice_name(voice_name), "{voice_name:?}");
    }
}

// Edge: the empty name and a lone non-ASCII letter.
#[test]
fn should_refuse_an_empty_or_non_ascii_name() {
    assert!(!is_plain_voice_name(""));
    assert!(!is_plain_voice_name("é"));
}
