//! Small words whose sound depends on the next word or on how they are
//! used, which Misaki spells by rule rather than from the lexicon.
//!
//! Misaki reads a part-of-speech tag for some of these. This port has no
//! tagger, so each takes the reading it has in most sentences: "a" and "an"
//! as articles, "in" as a preposition, "I" as the pronoun.

use crate::lexicon::{Lexicon, WordContext};

impl Lexicon {
    /// The rule-spelled reading of `word`, or `None` when no rule covers it.
    pub(crate) fn special_case(
        &self,
        word: &str,
        stress_level: Option<f32>,
        context: WordContext,
    ) -> Option<String> {
        if let Some(symbol_phonemes) = self.symbol_phonemes(word) {
            return Some(symbol_phonemes);
        }
        if is_dotted_abbreviation(word) {
            return self.spelled_letters(word);
        }
        match word {
            "a" | "A" => Some("ɐ".to_owned()),
            "an" | "An" | "AN" => Some("ɐn".to_owned()),
            "am" | "Am" | "AM" => self.am_reading(word, stress_level, context),
            "I" => Some("ˌI".to_owned()),
            "to" | "To" | "TO" => self.to_reading(context),
            "in" | "In" | "IN" => Some(in_reading(context)),
            "the" | "The" | "THE" => Some(the_reading(context)),
            "vs" | "vs." | "VS" | "Vs" => self.lookup("versus", None, Some(context)),
            "used" | "Used" | "USED" => self.gold_default("used"),
            _ => None,
        }
    }

    /// The weak "ɐm" only for lowercase, unstressed "am" before another word.
    fn am_reading(
        &self,
        word: &str,
        stress_level: Option<f32>,
        context: WordContext,
    ) -> Option<String> {
        let is_weak_form = context.next_starts_with_vowel.is_some()
            && word == "am"
            && !stress_level.is_some_and(|level| level > 0.0);
        if is_weak_form {
            Some("ɐm".to_owned())
        } else {
            self.gold_default("am")
        }
    }

    /// "tə" before a consonant, "tʊ" before a vowel, the full form before a pause.
    fn to_reading(&self, context: WordContext) -> Option<String> {
        match context.next_starts_with_vowel {
            None => self.gold_default("to"),
            Some(false) => Some("tə".to_owned()),
            Some(true) => Some("tʊ".to_owned()),
        }
    }
}

/// As a preposition "in" is unstressed unless a pause follows.
fn in_reading(context: WordContext) -> String {
    if context.next_starts_with_vowel.is_none() {
        "ˈɪn"
    } else {
        "ɪn"
    }
    .to_owned()
}

/// "ði" before a vowel, "ðə" otherwise.
fn the_reading(context: WordContext) -> String {
    if context.next_starts_with_vowel == Some(true) {
        "ði"
    } else {
        "ðə"
    }
    .to_owned()
}

/// "U.S." or "e.g.": short letter groups between dots.
fn is_dotted_abbreviation(word: &str) -> bool {
    let inner = word.trim_matches('.');
    inner.contains('.')
        && word
            .chars()
            .filter(|character| *character != '.')
            .all(char::is_alphabetic)
        && word
            .split('.')
            .map(|letter_group| letter_group.chars().count())
            .max()
            .unwrap_or(0)
            < 3
}

#[cfg(test)]
#[path = "small_word_rules_tests.rs"]
mod tests;
