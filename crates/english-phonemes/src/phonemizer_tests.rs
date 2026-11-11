use super::english_phonemes;

#[test]
fn should_phonemize_a_line_of_shakespeare_with_its_pauses() {
    let phonemes = english_phonemes("But soft, what light through yonder window breaks?");
    assert!(phonemes.contains(','), "{phonemes}");
    assert!(phonemes.ends_with('?'), "{phonemes}");
    assert_eq!(phonemes.split(' ').count(), 8, "{phonemes}");
}

#[test]
fn should_choose_the_article_form_from_the_next_word() {
    assert!(english_phonemes("the apple").starts_with("ði "));
    assert!(english_phonemes("the pear").starts_with("ðə "));
}

#[test]
fn should_write_flaps_the_way_kokoro_learned_them() {
    let phonemes = english_phonemes("water");
    assert!(!phonemes.contains('ɾ'), "{phonemes}");
}

#[test]
fn should_read_numbers_and_money_in_a_sentence() {
    let phonemes = english_phonemes("It costs $5 in 2026.");
    assert!(!phonemes.contains('$'), "{phonemes}");
    assert!(
        !phonemes.chars().any(|sound| sound.is_ascii_digit()),
        "{phonemes}"
    );
}

// Negative: nothing sayable gives nothing.
#[test]
fn should_return_an_empty_string_for_text_without_sounds() {
    assert_eq!(english_phonemes("*** // ~~"), "");
}
