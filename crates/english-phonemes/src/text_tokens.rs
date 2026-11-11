//! Splits running text into the pieces the phonemizer reads: words, numbers,
//! punctuation that becomes a pause, and symbols that become words. This
//! stands in for spaCy's tokenizer in Misaki; it keeps only the distinctions
//! the phonemes depend on.

use astronomical_english_number_words::{number_span_end, starts_signed_number};

/// Punctuation Kokoro reads as a pause or intonation mark, kept in the output.
pub(crate) const SPOKEN_PUNCTUATION: &str = ";:,.!?—…\"“”()";
/// The spoken punctuation that ends a phrase, after which no word "comes next"
/// (quotes and brackets do not).
pub(crate) const PAUSING_PUNCTUATION: &str = ";:,.!?—…";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TextTokenKind {
    /// Letters, with apostrophes inside ("don't") and dots inside ("U.S.").
    Word,
    /// Digits with commas, a point, a leading minus and an ending ("21st").
    Number,
    /// A mark from [`SPOKEN_PUNCTUATION`], or a dash between words.
    Punctuation,
    /// "%", "&", "+", "@" or "$", which are read as words.
    Symbol,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextToken {
    pub(crate) text: String,
    pub(crate) kind: TextTokenKind,
    /// Whether a space followed the token in the text, so the phonemes keep
    /// the word boundary.
    pub(crate) is_followed_by_space: bool,
}

/// The tokens of `text`; characters that carry no sound (markdown marks,
/// slashes, emoji) are dropped.
pub(crate) fn text_tokens(text: &str) -> Vec<TextToken> {
    let characters: Vec<char> = text.replace(['‘', '’'], "'").chars().collect();
    let mut tokens = Vec::new();
    let mut position = 0;
    while position < characters.len() {
        let character = characters[position];
        let (token_end, token_kind) = if character.is_alphabetic() {
            (word_end(&characters, position), Some(TextTokenKind::Word))
        } else if character.is_ascii_digit() || starts_signed_number(&characters, position) {
            (
                number_span_end(&characters, position),
                Some(TextTokenKind::Number),
            )
        } else if SPOKEN_PUNCTUATION.contains(character) || is_spoken_dash(&characters, position) {
            (position + 1, Some(TextTokenKind::Punctuation))
        } else if "%&+@$".contains(character) {
            (position + 1, Some(TextTokenKind::Symbol))
        } else {
            (position + 1, None)
        };
        if let Some(kind) = token_kind {
            let token_text: String = characters[position..token_end].iter().collect();
            let is_followed_by_space = characters
                .get(token_end)
                .is_some_and(|next| next.is_whitespace());
            tokens.push(TextToken {
                text: normalized_punctuation(token_text, &kind),
                kind,
                is_followed_by_space,
            });
        } else if character.is_whitespace()
            && let Some(previous_token) = tokens.last_mut()
        {
            previous_token.is_followed_by_space = true;
        }
        position = token_end;
    }
    tokens
}

fn word_end(characters: &[char], start: usize) -> usize {
    if let Some(abbreviation_end) = dotted_abbreviation_end(characters, start) {
        return abbreviation_end;
    }
    let mut token_end = start;
    while token_end < characters.len() {
        let is_joining_apostrophe = characters[token_end] == '\''
            && token_end > start
            && characters
                .get(token_end + 1)
                .is_some_and(|next| next.is_alphabetic());
        if characters[token_end].is_alphabetic() || is_joining_apostrophe {
            token_end += 1;
        } else {
            break;
        }
    }
    token_end
}

/// The token_end of "U.S." or "e.g.": two or more groups of one or two letters,
/// each closed by a dot. A dot between two whole words ("token_end.Next") is not one.
fn dotted_abbreviation_end(characters: &[char], start: usize) -> Option<usize> {
    let mut token_end = start;
    let mut letter_group_count = 0;
    loop {
        let group_length = characters[token_end..]
            .iter()
            .take_while(|character| character.is_alphabetic())
            .count();
        if !(1..=2).contains(&group_length)
            || characters.get(token_end + group_length) != Some(&'.')
        {
            break;
        }
        token_end += group_length + 1;
        letter_group_count += 1;
    }
    (letter_group_count >= 2).then_some(token_end)
}

/// A hyphen or en dash standing alone between spaces is a pause; inside a word
/// ("well-known") it only separates the parts.
fn is_spoken_dash(characters: &[char], position: usize) -> bool {
    let is_dash = matches!(characters[position], '-' | '–');
    let has_space_before = position == 0 || characters[position - 1].is_whitespace();
    let has_space_after = characters
        .get(position + 1)
        .is_none_or(|next| next.is_whitespace());
    is_dash && has_space_before && has_space_after
}

fn normalized_punctuation(token_text: String, kind: &TextTokenKind) -> String {
    if *kind == TextTokenKind::Punctuation && (token_text == "-" || token_text == "–") {
        return "—".to_owned();
    }
    token_text
}

#[cfg(test)]
#[path = "text_tokens_tests.rs"]
mod tests;
