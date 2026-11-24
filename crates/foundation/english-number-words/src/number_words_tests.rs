use super::{cardinal_words, decimal_words, ordinal_words, year_words};

fn joined(spoken_words: Vec<String>) -> String {
    spoken_words.join(" ")
}

#[test]
fn should_spell_cardinals_without_and_as_misaki_reads_them() {
    assert_eq!(
        joined(cardinal_words(1234)),
        "one thousand two hundred thirty four"
    );
    assert_eq!(joined(cardinal_words(1_000_005)), "one million five");
}

#[test]
fn should_spell_zero_and_the_teens_as_single_words() {
    assert_eq!(joined(cardinal_words(0)), "zero");
    assert_eq!(joined(cardinal_words(13)), "thirteen");
}

#[test]
fn should_make_only_the_last_word_ordinal() {
    assert_eq!(joined(ordinal_words(21)), "twenty first");
    assert_eq!(joined(ordinal_words(12)), "twelfth");
    assert_eq!(joined(ordinal_words(40)), "fortieth");
    assert_eq!(joined(ordinal_words(100)), "one hundredth");
}

#[test]
fn should_read_years_the_way_people_say_them() {
    assert_eq!(joined(year_words(1984)), "nineteen eighty four");
    assert_eq!(joined(year_words(1905)), "nineteen oh five");
    assert_eq!(joined(year_words(1900)), "nineteen hundred");
    assert_eq!(joined(year_words(2026)), "twenty twenty six");
}

// Boundary: num2words keeps these years cardinal.
#[test]
fn should_keep_round_and_early_century_years_cardinal() {
    assert_eq!(joined(year_words(2000)), "two thousand");
    assert_eq!(joined(year_words(2005)), "two thousand five");
    assert_eq!(joined(year_words(1000)), "one thousand");
}

#[test]
fn should_read_decimals_digit_by_digit_after_the_point() {
    assert_eq!(
        decimal_words("3.14").map(joined).as_deref(),
        Some("three point one four")
    );
    assert_eq!(
        decimal_words(".5").map(joined).as_deref(),
        Some("point five")
    );
}

#[test]
fn should_refuse_text_that_is_not_a_decimal() {
    assert_eq!(decimal_words("3.x"), None);
    assert_eq!(decimal_words("314"), None);
    assert_eq!(decimal_words("3."), None);
}
