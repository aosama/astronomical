use super::{Lexicon, WordContext};

fn spoken(word: &str) -> Option<String> {
    Lexicon::american().word_phonemes(word, WordContext::default())
}

#[test]
fn should_spell_known_words_from_the_gold_lexicon() {
    assert_eq!(spoken("hello").as_deref(), Some("həlˈO"));
    assert_eq!(spoken("world").as_deref(), Some("wˈɜɹld"));
}

#[test]
fn should_read_a_capitalised_word_like_its_lowercase_entry() {
    assert_eq!(spoken("Hello"), spoken("hello"));
}

#[test]
fn should_take_the_default_reading_of_a_word_with_several() {
    assert_eq!(spoken("read").as_deref(), Some("ɹˈid"));
}

#[test]
fn should_spell_an_unknown_acronym_letter_by_letter_with_final_stress() {
    let spelled = spoken("XQZ").expect("letters are always known");
    assert_eq!(
        spelled.matches('ˈ').count(),
        1,
        "one primary stress on {spelled}"
    );
    let last_letter = Lexicon::american()
        .gold_default("Z")
        .expect("letters are in the gold lexicon");
    assert!(
        spelled.ends_with(&last_letter),
        "{spelled} should end with {last_letter}"
    );
}

// Negative: no letters, nothing to spell.
#[test]
fn should_know_nothing_about_a_word_without_letters() {
    assert_eq!(Lexicon::american().spelled_letters("123"), None);
}

#[test]
fn should_read_symbols_as_words() {
    assert_eq!(
        Lexicon::american().symbol_phonemes("%"),
        Lexicon::american().lookup("percent", None, None)
    );
    assert_eq!(Lexicon::american().symbol_phonemes("#"), None);
}

#[test]
fn should_treat_a_single_ascii_letter_as_known() {
    assert!(Lexicon::american().is_known("q"));
    assert!(!Lexicon::american().is_known("naïve-ish"));
}
