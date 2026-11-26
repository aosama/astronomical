use super::{LONG_ESSAY_CHARACTERS, USUAL_LONG_PROMPT_TOKENS, characters_for};

#[test]
fn should_keep_the_usual_characters_for_the_usual_ten_thousand_tokens() {
    assert_eq!(
        characters_for(LONG_ESSAY_CHARACTERS, USUAL_LONG_PROMPT_TOKENS),
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
