//! The words a number token is said with: Misaki's `get_number` for the
//! forms running text uses (cardinals, ordinals, years, decimals, negative
//! numbers and dollar amounts).

use crate::number_words::{cardinal_words, decimal_words, ordinal_words, year_words};

const ORDINAL_SUFFIXES: [&str; 4] = ["st", "nd", "rd", "th"];

/// An ending a reader adds to the last spoken word: "1990s", "2000's".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumberEnding {
    None,
    Plural,
    PastTense,
    Progressive,
}

/// A number's words and the ending its last word takes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpokenNumber {
    pub words: Vec<String>,
    pub ending: NumberEnding,
}

/// Whether `token` is a number Misaki would read: digits with commas or a
/// point, an optional leading minus and at most one ending such as "th" or "s".
pub fn is_number_token(token: &str) -> bool {
    if !token.chars().any(|character| character.is_ascii_digit()) {
        return false;
    }
    let (number_part, _) = split_ending(token);
    number_part
        .chars()
        .enumerate()
        .all(|(position, character)| {
            character.is_ascii_digit()
                || ",.".contains(character)
                || (position == 0 && character == '-')
        })
}

/// The words for `token`; `is_dollar_amount` when a "$" stood right before
/// it. `None` when the token cannot be read (for example a number too large
/// for 64 bits).
pub fn spoken_number(token: &str, is_dollar_amount: bool) -> Option<SpokenNumber> {
    let (signed_number, ending_text) = split_ending(token);
    let (is_negative, number) = match signed_number.strip_prefix('-') {
        Some(unsigned_number) => (true, unsigned_number),
        None => (false, signed_number),
    };
    let mut words = if is_negative {
        vec!["minus".to_owned()]
    } else {
        Vec::new()
    };
    if is_dollar_amount {
        words.extend(dollar_words(number)?);
    } else {
        words.extend(plain_number_words(number, ending_text, is_negative)?);
    }
    Some(SpokenNumber {
        words,
        ending: number_ending(ending_text),
    })
}

/// Splits "1990s" into ("1990", "s") and "21st" into ("21", "st").
fn split_ending(token: &str) -> (&str, &str) {
    let ending_start = token
        .char_indices()
        .rev()
        .take_while(|(_, character)| character.is_ascii_lowercase() || *character == '\'')
        .last()
        .map_or(token.len(), |(position, _)| position);
    token.split_at(ending_start)
}

fn number_ending(ending_text: &str) -> NumberEnding {
    match ending_text {
        "s" | "'s" => NumberEnding::Plural,
        "ed" | "'d" => NumberEnding::PastTense,
        "ing" => NumberEnding::Progressive,
        _ => NumberEnding::None,
    }
}

fn plain_number_words(number: &str, ending_text: &str, is_negative: bool) -> Option<Vec<String>> {
    let is_digits_only =
        !number.is_empty() && number.chars().all(|character| character.is_ascii_digit());
    let is_ordinal = ORDINAL_SUFFIXES.contains(&ending_text);
    if is_digits_only && is_ordinal {
        return Some(ordinal_words(number.parse().ok()?));
    }
    if is_digits_only && !is_negative && number.len() == 4 {
        return Some(year_words(number.parse().ok()?));
    }
    let without_commas = number.replace(',', "");
    if without_commas.contains('.') {
        return decimal_words(&without_commas);
    }
    let whole_number: u64 = without_commas.parse().ok()?;
    Some(if is_ordinal {
        ordinal_words(whole_number)
    } else {
        cardinal_words(whole_number)
    })
}

/// "$5.50" as "five dollars and fifty cents"; a zero half is left unsaid, and
/// cents with more than two digits are read as a decimal.
fn dollar_words(amount: &str) -> Option<Vec<String>> {
    let without_commas = amount.replace(',', "");
    let (dollars_text, cents_text) = without_commas
        .split_once('.')
        .unwrap_or((&without_commas, ""));
    if cents_text.len() > 2 {
        return decimal_words(&without_commas);
    }
    let dollars = parsed_or_zero(dollars_text)?;
    // "$5.5" is five dollars and fifty cents, as people mean it.
    let cents = if cents_text.len() == 1 {
        parsed_or_zero(cents_text)? * 10
    } else {
        parsed_or_zero(cents_text)?
    };
    let mut words = Vec::new();
    if dollars > 0 || cents == 0 {
        words.extend(cardinal_words(dollars));
        words.push(if dollars == 1 { "dollar" } else { "dollars" }.to_owned());
    }
    if cents > 0 {
        if dollars > 0 {
            words.push("and".to_owned());
        }
        words.extend(cardinal_words(cents));
        words.push(if cents == 1 { "cent" } else { "cents" }.to_owned());
    }
    Some(words)
}

fn parsed_or_zero(digits: &str) -> Option<u64> {
    if digits.is_empty() {
        Some(0)
    } else {
        digits.parse().ok()
    }
}

#[cfg(test)]
#[path = "spoken_number_tests.rs"]
mod tests;
