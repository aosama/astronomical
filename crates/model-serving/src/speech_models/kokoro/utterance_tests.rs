use super::{KOKORO_MAXIMUM_PHONEMES, validated_token_ids};
use crate::speech_models::kokoro::configuration::KokoroConfiguration;

fn small_vocabulary_configuration() -> KokoroConfiguration {
    KokoroConfiguration::from_vocabulary_for_tests(&[('h', 50), ('ə', 83), (',', 3)])
}

#[test]
fn should_wrap_known_phonemes_in_markers() {
    let token_ids =
        validated_token_ids(&small_vocabulary_configuration(), "hə,", 1.0).expect("valid");
    assert_eq!(token_ids, [0, 50, 83, 3, 0]);
}

// Boundaries: 510 phonemes fit, 511 and 0 do not.
#[test]
fn should_accept_exactly_the_maximum_and_refuse_one_more() {
    let configuration = small_vocabulary_configuration();
    let longest = "h".repeat(KOKORO_MAXIMUM_PHONEMES);
    assert!(validated_token_ids(&configuration, &longest, 1.0).is_ok());
    assert!(validated_token_ids(&configuration, &format!("{longest}h"), 1.0).is_err());
    assert!(validated_token_ids(&configuration, "", 1.0).is_err());
}

#[test]
fn should_refuse_a_speed_that_is_not_a_positive_number() {
    let configuration = small_vocabulary_configuration();
    for speed in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            validated_token_ids(&configuration, "hə", speed).is_err(),
            "{speed}"
        );
    }
}

// Negative: phonemes the vocabulary lacks would leave only the markers.
#[test]
fn should_refuse_phonemes_that_are_all_unknown() {
    assert!(validated_token_ids(&small_vocabulary_configuration(), "❓❓", 1.0).is_err());
}
