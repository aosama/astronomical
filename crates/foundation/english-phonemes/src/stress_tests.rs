use super::apply_stress;

#[test]
fn should_leave_the_lexicon_spelling_alone_without_a_stress_level() {
    assert_eq!(apply_stress("həlˈO", None), "həlˈO");
}

#[test]
fn should_strip_every_mark_below_minus_one() {
    assert_eq!(apply_stress("ˌækˈʧuəli", Some(-2.0)), "ækʧuəli");
}

#[test]
fn should_demote_primary_stress_for_a_word_that_gives_way() {
    assert_eq!(apply_stress("ˌækˈʧuəli", Some(-0.5)), "ækˌʧuəli");
}

#[test]
fn should_add_secondary_stress_before_the_first_vowel_of_an_unmarked_word() {
    assert_eq!(apply_stress("ðə", Some(0.5)), "ðˌə");
}

#[test]
fn should_add_primary_stress_to_an_unmarked_capitalised_acronym() {
    assert_eq!(apply_stress("ðə", Some(2.0)), "ðˈə");
}

#[test]
fn should_promote_secondary_stress_when_the_word_is_shouted() {
    assert_eq!(apply_stress("ˌæd", Some(2.0)), "ˈæd");
}

// Edge: a run of consonants has no syllable to carry a mark.
#[test]
fn should_not_stress_a_word_without_vowels() {
    assert_eq!(apply_stress("hmm", Some(2.0)), "hmm");
}
