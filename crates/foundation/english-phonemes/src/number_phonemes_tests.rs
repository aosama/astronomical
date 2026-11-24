use crate::lexicon::Lexicon;

fn words_read(words: &str) -> String {
    words
        .split(' ')
        .map(|word| {
            Lexicon::american()
                .lookup(word, None, None)
                .expect("number words are in the lexicon")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn should_read_a_four_digit_number_as_a_year() {
    let reading = Lexicon::american().number_phonemes("1984", false);
    assert_eq!(reading, Some(words_read("nineteen eighty four")));
}

#[test]
fn should_read_ordinals_and_plural_decades() {
    assert_eq!(
        Lexicon::american().number_phonemes("21st", false),
        Some(words_read("twenty first"))
    );
    let nineties = Lexicon::american()
        .number_phonemes("1990s", false)
        .expect("decades read");
    assert!(nineties.ends_with('z'), "{nineties}");
}

#[test]
fn should_read_a_dollar_amount_with_its_units() {
    let reading = Lexicon::american()
        .number_phonemes("5.50", true)
        .expect("money reads");
    assert!(reading.starts_with(&words_read("five")), "{reading}");
    assert!(reading.contains(&words_read("fifty")), "{reading}");
}

// Boundary: one dollar is singular.
#[test]
fn should_keep_one_dollar_singular() {
    let reading = Lexicon::american()
        .number_phonemes("1", true)
        .expect("money reads");
    assert_eq!(
        reading,
        format!("{} {}", words_read("one"), words_read("dollar"))
    );
}

#[test]
fn should_say_minus_before_a_negative_number() {
    let reading = Lexicon::american()
        .number_phonemes("-7", false)
        .expect("negatives read");
    assert_eq!(reading, words_read("minus seven"));
}

// Negative: a number too long to read gives no phonemes.
#[test]
fn should_read_nothing_for_an_unreadable_number() {
    assert_eq!(
        Lexicon::american().number_phonemes("12345678901234567890123", false),
        None
    );
}
