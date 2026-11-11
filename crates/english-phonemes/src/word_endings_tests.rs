use super::{with_ed_ending, with_ing_ending, with_s_ending};
use crate::lexicon::{Lexicon, WordContext};

fn spoken_before(word: &str, next_starts_with_vowel: Option<bool>) -> Option<String> {
    let context = WordContext {
        next_starts_with_vowel,
    };
    Lexicon::american().word_phonemes(word, context)
}

#[test]
fn should_add_the_plural_sound_that_matches_the_last_sound() {
    assert_eq!(with_s_ending("kˈæt"), "kˈæts");
    assert_eq!(with_s_ending("dˈɔɡ"), "dˈɔɡz");
    assert_eq!(with_s_ending("bˈɑks"), "bˈɑksᵻz");
}

#[test]
fn should_flap_a_t_between_vowels_before_ed_and_ing() {
    assert_eq!(with_ed_ending("wˈAt"), "wˈAɾᵻd");
    assert_eq!(with_ing_ending("wˈAt"), "wˈAɾɪŋ");
    assert_eq!(with_ed_ending("wˈɔk"), "wˈɔkt");
    assert_eq!(with_ed_ending("nˈid"), "nˈidᵻd");
}

// Edge: the doubled consonant of "running" leads back to "run".

#[test]
fn should_find_the_stem_of_a_word_with_a_doubled_consonant() {
    let running = spoken_before("runnings", None);
    let stemmed = Lexicon::american().stem_ing("runing", None, WordContext::default());
    assert!(
        running.is_some() || stemmed.is_some(),
        "runnings or runing should resolve through a stem"
    );
    assert!(
        Lexicon::american()
            .stem_ing("ring", None, WordContext::default())
            .is_none()
    );
}
