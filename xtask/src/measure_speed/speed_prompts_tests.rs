use super::{CALIBRATED_LONG_PROMPT_TOKENS, LONG_ESSAY_CHARACTERS, characters_for};

#[test]
fn should_keep_the_calibrated_characters_for_ten_thousand_tokens() {
    assert_eq!(
        characters_for(LONG_ESSAY_CHARACTERS, CALIBRATED_LONG_PROMPT_TOKENS),
        LONG_ESSAY_CHARACTERS
    );
}

#[test]
fn should_cut_a_fifth_of_the_characters_for_two_thousand_tokens() {
    assert_eq!(characters_for(37_000, 2_000), 7_400);
}

#[test]
fn should_cut_nothing_for_no_tokens() {
    assert_eq!(characters_for(37_000, 0), 0);
}

#[test]
fn should_cut_half_the_characters_for_the_usual_five_thousand_tokens() {
    assert_eq!(
        characters_for(37_000, super::USUAL_LONG_PROMPT_TOKENS),
        18_500
    );
}
