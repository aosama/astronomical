//! Shared helpers for this crate's unit tests.

use crate::ordered_json::OrderedJsonValue;
use crate::question::DecisionQuestion;
use crate::system_one_questions::parse_system_one_questions;
use crate::token_encoder::{DecisionTokenEncoder, TokenEncodingError};

/// The first identifier the stand-in tokenizer gives a word; lower ones are special tokens.
pub const FIRST_WORD_TOKEN_ID: u32 = 100;
/// How many distinct word identifiers the stand-in tokenizer hands out.
pub const WORD_TOKEN_ID_RANGE: u32 = 50_000;

/// A stand-in tokenizer: one token per whitespace-separated word.
///
/// It is the same scheme the golden-vector generator uses on the reference
/// side, so sequence layouts can be compared token for token without a real
/// tokenizer file. A word's identifier is its 32-bit FNV-1a hash folded into
/// a fixed range.
///
/// Rust and Python agree on what whitespace is for spaces, tabs and line
/// breaks but differ on a few rare control characters, so the golden
/// sequences use only ordinary whitespace.
pub struct WordHashTokenEncoder;

impl DecisionTokenEncoder for WordHashTokenEncoder {
    fn encode_without_special_tokens(&self, text: &str) -> Result<Vec<u32>, TokenEncodingError> {
        let mut word_token_ids = Vec::new();
        for word in text.split_whitespace() {
            word_token_ids
                .push(FIRST_WORD_TOKEN_ID + fnv1a_32(word.as_bytes()) % WORD_TOKEN_ID_RANGE);
        }
        Ok(word_token_ids)
    }
}

fn fnv1a_32(word_bytes: &[u8]) -> u32 {
    let mut hash_value: u32 = 0x811C_9DC5;
    for word_byte in word_bytes {
        hash_value ^= u32::from(*word_byte);
        hash_value = hash_value.wrapping_mul(0x0100_0193);
    }
    hash_value
}

/// An encoder that always fails, for tests of error propagation.
pub struct FailingTokenEncoder;

impl DecisionTokenEncoder for FailingTokenEncoder {
    fn encode_without_special_tokens(&self, _text: &str) -> Result<Vec<u32>, TokenEncodingError> {
        Err(TokenEncodingError {
            reason: "the stand-in encoder always fails".to_owned(),
        })
    }
}

/// Parses JSON text into an ordered value, panicking on malformed test input.
pub fn ordered_json(json_text: &str) -> OrderedJsonValue {
    serde_json::from_str(json_text).expect("test JSON is well formed")
}

/// Parses a `questions` object that defines exactly one question.
pub fn single_question(questions_json_text: &str) -> DecisionQuestion {
    let mut parsed_questions = parse_system_one_questions(&ordered_json(questions_json_text))
        .expect("the test question is valid");
    assert_eq!(parsed_questions.len(), 1, "the test defines one question");
    parsed_questions.remove(0)
}

/// Asserts two floats agree within `tolerance`.
pub fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual} (tolerance {tolerance})"
    );
}
