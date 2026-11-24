use super::{MAXIMUM_SPEECH_INPUT_CHARACTERS, SpeechRequest};

fn request(text: &str, speed: f32) -> SpeechRequest {
    SpeechRequest {
        text: text.to_owned(),
        voice_name: None,
        speed,
        seed: 7,
    }
}

#[test]
fn should_accept_a_line_of_shakespeare_at_natural_pace() {
    assert!(
        request("Good night, good night! Parting is such sweet sorrow.", 1.0)
            .validate()
            .is_ok()
    );
}

// Negative: nothing to say.
#[test]
fn should_refuse_blank_text() {
    assert!(request("  \n ", 1.0).validate().is_err());
}

// Boundaries: exactly the character limit, and one over.
#[test]
fn should_accept_the_character_limit_and_refuse_one_more() {
    assert!(
        request(&"a".repeat(MAXIMUM_SPEECH_INPUT_CHARACTERS), 1.0)
            .validate()
            .is_ok()
    );
    assert!(
        request(&"a".repeat(MAXIMUM_SPEECH_INPUT_CHARACTERS + 1), 1.0)
            .validate()
            .is_err()
    );
}

#[test]
fn should_refuse_a_pace_outside_the_range_or_not_a_number() {
    for speed in [0.0, 0.2, 4.5, f32::NAN, f32::INFINITY] {
        assert!(request("Romeo", speed).validate().is_err(), "{speed}");
    }
    assert!(request("Romeo", 0.25).validate().is_ok());
    assert!(request("Romeo", 4.0).validate().is_ok());
}
