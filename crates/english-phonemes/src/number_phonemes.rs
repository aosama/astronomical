//! A number token's phonemes: the words `astronomical-english-number-words`
//! says it with, each read from the lexicon, and the ending ("1990s") added
//! to the last sound.

use astronomical_english_number_words::{NumberEnding, spoken_number};

use crate::lexicon::Lexicon;
use crate::word_endings::{with_ed_ending, with_ing_ending, with_s_ending};

/// "point" is read unstressed between the two halves of a decimal.
const UNSTRESSED_LEVEL: f32 = -2.0;

impl Lexicon {
    /// The phonemes of a number token; `is_dollar_amount` when a "$" stood
    /// right before it.
    pub(crate) fn number_phonemes(&self, token: &str, is_dollar_amount: bool) -> Option<String> {
        let spoken = spoken_number(token, is_dollar_amount)?;
        let mut word_readings = Vec::with_capacity(spoken.words.len());
        for spoken_word in &spoken.words {
            word_readings.push(self.number_word_reading(spoken_word)?);
        }
        let number_reading = word_readings.join(" ");
        Some(match spoken.ending {
            NumberEnding::Plural => with_s_ending(&number_reading),
            NumberEnding::PastTense => with_ed_ending(&number_reading),
            NumberEnding::Progressive => with_ing_ending(&number_reading),
            NumberEnding::None => number_reading,
        })
    }

    fn number_word_reading(&self, spoken_word: &str) -> Option<String> {
        match spoken_word {
            // Misaki builds the plural units from their singular, as the
            // lexicon may list only "dollar" and "cent".
            "dollars" | "cents" => {
                let singular = &spoken_word[..spoken_word.len() - 1];
                Some(with_s_ending(&self.lookup(singular, None, None)?))
            }
            "point" => self.lookup(spoken_word, Some(UNSTRESSED_LEVEL), None),
            _ => self.lookup(spoken_word, None, None),
        }
    }
}

#[cfg(test)]
#[path = "number_phonemes_tests.rs"]
mod tests;
