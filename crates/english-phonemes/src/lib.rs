//! English text to Kokoro phonemes.
//!
//! Kokoro-82M reads phonemes, not letters, and it learned them from the Misaki
//! grapheme-to-phoneme project. This crate ports the parts of Misaki that
//! running English needs (its American lexicon, stress rules, word endings,
//! numbers and context-sensitive small words) so the speech engine can run
//! without Python. Misaki's part-of-speech tagger and its neural fallback for
//! unknown words are not ported: a word with several readings takes its
//! common one, and an unknown word is spelled letter by letter.

mod lexicon;
mod number_phonemes;
mod phoneme_chunks;
mod phonemizer;
mod small_word_rules;
mod stress;
mod text_tokens;
mod word_endings;
mod word_reading;

pub use phoneme_chunks::english_phoneme_chunks;
pub use phonemizer::{english_phonemes, prepare_english_lexicon};
