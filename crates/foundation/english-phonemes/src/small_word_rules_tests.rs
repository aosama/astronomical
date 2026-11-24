use crate::lexicon::{Lexicon, WordContext};

fn spoken_before(word: &str, next_starts_with_vowel: Option<bool>) -> Option<String> {
    let context = WordContext {
        next_starts_with_vowel,
    };
    Lexicon::american().word_phonemes(word, context)
}

#[test]
fn should_say_the_as_thee_only_before_a_vowel() {
    assert_eq!(spoken_before("the", Some(true)).as_deref(), Some("ði"));
    assert_eq!(spoken_before("the", Some(false)).as_deref(), Some("ðə"));
    assert_eq!(spoken_before("the", None).as_deref(), Some("ðə"));
}

#[test]
fn should_weaken_to_before_a_following_word() {
    assert_eq!(spoken_before("to", Some(false)).as_deref(), Some("tə"));
    assert_eq!(spoken_before("to", Some(true)).as_deref(), Some("tʊ"));
}

#[test]
fn should_stress_in_only_before_a_pause() {
    assert_eq!(spoken_before("in", None).as_deref(), Some("ˈɪn"));
    assert_eq!(spoken_before("in", Some(false)).as_deref(), Some("ɪn"));
}

#[test]
fn should_read_dotted_abbreviations_letter_by_letter() {
    let spelled = spoken_before("U.S.", None).expect("U.S. is spelled");
    assert_eq!(spelled, Lexicon::american().spelled_letters("US").unwrap());
}

// Negative: a word no rule covers is left to the lexicon.
#[test]
fn should_leave_ordinary_words_to_the_lexicon() {
    let context = WordContext {
        next_starts_with_vowel: None,
    };
    assert_eq!(
        Lexicon::american().special_case("Juliet", None, context),
        None
    );
}
