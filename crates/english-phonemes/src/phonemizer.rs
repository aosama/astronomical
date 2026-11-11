//! Text to one phoneme string, read right to left as Misaki reads it: a small
//! word such as "the" or "to" sounds different before a vowel, so each word is
//! spelled knowing how the next one starts.

use crate::lexicon::{Lexicon, WordContext};
use crate::stress::is_vowel;
use crate::text_tokens::{PAUSING_PUNCTUATION, TextToken, TextTokenKind, text_tokens};
use astronomical_english_number_words::is_number_token;

const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";

/// Parses the lexicon now (a few hundred milliseconds, once per process), so
/// the first request does not pay for it.
pub fn prepare_english_lexicon() {
    Lexicon::american();
}

/// The phonemes Kokoro reads for `text`. Words no rule knows are spelled
/// letter by letter, and characters without a sound are left out.
pub fn english_phonemes(text: &str) -> String {
    let lexicon = Lexicon::american();
    let tokens = text_tokens(text);
    let mut token_phonemes: Vec<String> = vec![String::new(); tokens.len()];
    let mut context = WordContext::default();
    for token_index in (0..tokens.len()).rev() {
        let token = &tokens[token_index];
        let is_dollar_amount = token_index > 0 && tokens[token_index - 1].text == "$";
        let phonemes = token_reading(lexicon, token, is_dollar_amount, context);
        context = context_before(&phonemes, context);
        token_phonemes[token_index] = phonemes;
    }
    let mut joined_phonemes = String::new();
    for (token, phonemes) in tokens.iter().zip(&token_phonemes) {
        joined_phonemes.push_str(phonemes);
        if token.is_followed_by_space && !phonemes.is_empty() {
            joined_phonemes.push(' ');
        }
    }
    // Kokoro v1.0 was trained before Misaki wrote the flap and the glottal
    // stop with their own symbols; it knows them as "T" and "t".
    joined_phonemes.trim().replace('ɾ', "T").replace('ʔ', "t")
}

fn token_reading(
    lexicon: &Lexicon,
    token: &TextToken,
    is_dollar_amount: bool,
    context: WordContext,
) -> String {
    let reading = match token.kind {
        TextTokenKind::Word => lexicon.word_phonemes(&token.text, context),
        TextTokenKind::Number if is_number_token(&token.text) => lexicon
            .number_phonemes(&token.text, is_dollar_amount)
            .or_else(|| digit_by_digit(lexicon, &token.text)),
        TextTokenKind::Number => digit_by_digit(lexicon, &token.text),
        TextTokenKind::Punctuation => Some(token.text.clone()),
        // The dollar sign is said after the amount ("five dollars").
        TextTokenKind::Symbol if token.text == "$" => Some(String::new()),
        TextTokenKind::Symbol => lexicon.symbol_phonemes(&token.text),
    };
    reading.unwrap_or_default()
}

fn digit_by_digit(lexicon: &Lexicon, number_text: &str) -> Option<String> {
    let digit_readings: Vec<String> = number_text
        .chars()
        .filter(char::is_ascii_digit)
        .filter_map(|digit| lexicon.number_phonemes(&digit.to_string(), false))
        .collect();
    (!digit_readings.is_empty()).then(|| digit_readings.join(" "))
}

/// What the word before should know: whether these phonemes start with a
/// vowel sound, or `None` when they are a pause.
fn context_before(phonemes: &str, context_after: WordContext) -> WordContext {
    let first_telling_sound = phonemes.chars().find(|sound| {
        is_vowel(*sound) || CONSONANTS.contains(*sound) || PAUSING_PUNCTUATION.contains(*sound)
    });
    let next_starts_with_vowel = match first_telling_sound {
        Some(sound) if PAUSING_PUNCTUATION.contains(sound) => None,
        Some(sound) => Some(is_vowel(sound)),
        None => context_after.next_starts_with_vowel,
    };
    WordContext {
        next_starts_with_vowel,
    }
}

#[cfg(test)]
#[path = "phonemizer_tests.rs"]
mod tests;
