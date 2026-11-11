//! How one word token is read: Misaki's `Lexicon.__call__` and `get_word`.
//! A small word with a context rule comes first, then the lexicon (after
//! lowercasing a word whose case alone differs), then a known stem with an
//! -s, -ed or -ing ending, and last the word spelled letter by letter.

use crate::lexicon::{Lexicon, WordContext};

/// Misaki's `cap_stresses`: a Capitalised word gets 0.5, an ALL-CAPS word 2.
const CAPITALISED_STRESS_LEVEL: f32 = 0.5;
const ALL_CAPITALS_STRESS_LEVEL: f32 = 2.0;
/// Misaki stresses an -ing stem at 0.5 when the word carries no stress of its own.
const UNMARKED_ING_STRESS_LEVEL: f32 = 0.5;

impl Lexicon {
    /// The phonemes of one word token, or `None` when no rule knows it.
    pub(crate) fn word_phonemes(&self, word: &str, context: WordContext) -> Option<String> {
        let normalized_word = word.replace(['‘', '’'], "'");
        let stress_level = if normalized_word == normalized_word.to_lowercase() {
            None
        } else if normalized_word == normalized_word.to_uppercase() {
            Some(ALL_CAPITALS_STRESS_LEVEL)
        } else {
            Some(CAPITALISED_STRESS_LEVEL)
        };
        if let Some(special_phonemes) = self.special_case(&normalized_word, stress_level, context) {
            return Some(special_phonemes);
        }
        let lookup_word =
            self.lowercase_if_only_the_case_differs(&normalized_word, stress_level, context);
        self.known_word_phonemes(&lookup_word, stress_level, context)
            .or_else(|| self.stem_s(&lookup_word, stress_level, context))
            .or_else(|| self.stem_ed(&lookup_word, stress_level, context))
            .or_else(|| {
                self.stem_ing(
                    &lookup_word,
                    stress_level.or(Some(UNMARKED_ING_STRESS_LEVEL)),
                    context,
                )
            })
            .or_else(|| self.spelled_letters(&lookup_word))
    }

    fn known_word_phonemes(
        &self,
        word: &str,
        stress_level: Option<f32>,
        context: WordContext,
    ) -> Option<String> {
        if self.is_known(word) {
            return self.lookup(word, stress_level, Some(context));
        }
        if let Some(stem) = word.strip_suffix("s'") {
            let possessive = format!("{stem}'s");
            if self.is_known(&possessive) {
                return self.lookup(&possessive, stress_level, Some(context));
            }
        }
        let without_apostrophe = word.strip_suffix('\'')?;
        if self.is_known(without_apostrophe) {
            return self.lookup(without_apostrophe, stress_level, Some(context));
        }
        None
    }

    /// Misaki's rule for "Hello" or "HELLO" when only "hello" is listed.
    fn lowercase_if_only_the_case_differs(
        &self,
        word: &str,
        stress_level: Option<f32>,
        context: WordContext,
    ) -> String {
        let lowercase_word = word.to_lowercase();
        let letters_only = word.replace('\'', "");
        let is_tail_lowercase = word.chars().skip(1).all(|letter| !letter.is_uppercase());
        let should_lowercase = word.chars().count() > 1
            && !letters_only.is_empty()
            && letters_only.chars().all(char::is_alphabetic)
            && word != lowercase_word
            && !self.has_entry(word)
            && (word == word.to_uppercase() || is_tail_lowercase)
            && (self.has_entry(&lowercase_word)
                || self
                    .stem_s(&lowercase_word, stress_level, context)
                    .is_some()
                || self
                    .stem_ed(&lowercase_word, stress_level, context)
                    .is_some()
                || self
                    .stem_ing(&lowercase_word, stress_level, context)
                    .is_some());
        if should_lowercase {
            lowercase_word
        } else {
            word.to_owned()
        }
    }
}
