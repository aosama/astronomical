use super::{NumberEnding, is_number_token, spoken_number};

fn words(token: &str, is_dollar_amount: bool) -> String {
    spoken_number(token, is_dollar_amount)
        .expect("a readable number")
        .words
        .join(" ")
}

#[test]
fn should_recognise_numbers_with_separators_signs_and_endings() {
    for number_token in ["1,234", "-3.5", "21st", "1990s"] {
        assert!(is_number_token(number_token), "{number_token}");
    }
}

#[test]
fn should_refuse_tokens_that_are_not_numbers() {
    assert!(!is_number_token("abc"));
    assert!(!is_number_token("3x4"));
    assert!(!is_number_token("-"));
}

#[test]
fn should_read_a_four_digit_number_as_a_year_and_others_as_cardinals() {
    assert_eq!(words("1984", false), "nineteen eighty four");
    assert_eq!(words("1,000,000", false), "one million");
    assert_eq!(words("-7", false), "minus seven");
}

#[test]
fn should_read_ordinals_and_mark_plural_decades() {
    assert_eq!(words("21st", false), "twenty first");
    let nineties = spoken_number("1990s", false).expect("decades read");
    assert_eq!(nineties.ending, NumberEnding::Plural);
}

#[test]
fn should_read_dollars_and_cents() {
    assert_eq!(words("5.50", true), "five dollars and fifty cents");
    assert_eq!(words("1", true), "one dollar");
    assert_eq!(words("0.01", true), "one cent");
}

// Edge: one cent digit means tens of cents, as people write it.
#[test]
fn should_read_one_cent_digit_as_tens_of_cents() {
    assert_eq!(words("5.5", true), "five dollars and fifty cents");
}

// Negative: a number past 64 bits cannot be read.
#[test]
fn should_give_up_on_a_twenty_digit_number() {
    assert_eq!(spoken_number("12345678901234567890123", false), None);
}
