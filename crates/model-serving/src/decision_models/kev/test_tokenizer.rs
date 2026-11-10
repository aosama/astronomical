//! A word-level tokenizer with Kev's five control tokens, for unit tests of
//! the layout: every Shakespeare word below is one token, and anything else
//! reads as the unknown token.

pub(super) const UNKNOWN_TOKEN_ID: u32 = 0;
pub(super) const STATE_TOKEN_ID: u32 = 1;
pub(super) const QUESTION_TOKEN_ID: u32 = 2;
pub(super) const OPTION_OPENING_TOKEN_ID: u32 = 3;
pub(super) const OPTION_CLOSING_TOKEN_ID: u32 = 4;
pub(super) const DECISION_TOKEN_ID: u32 = 5;
pub(super) const FIRST_WORD_TOKEN_ID: u32 = 10;
pub(super) const WORDS: [&str; 6] = ["to", "be", "or", "not", "no", "yes"];
pub(super) const KEV_CONTROL_TOKEN_NAMES: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];

fn added_special_token(token_id: u32, token_content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": token_id,
        "content": token_content,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true,
    })
}

/// The tokenizer's JSON, with `control_token_names` in Kev's order.
pub(super) fn tokenizer_bytes_with(control_token_names: [&str; 5]) -> Vec<u8> {
    let mut vocabulary = serde_json::Map::new();
    vocabulary.insert("<unk>".to_owned(), serde_json::json!(UNKNOWN_TOKEN_ID));
    let mut added_tokens = vec![added_special_token(UNKNOWN_TOKEN_ID, "<unk>")];
    for (control_index, control_token_name) in control_token_names.iter().enumerate() {
        let token_id = STATE_TOKEN_ID + control_index as u32;
        vocabulary.insert(
            (*control_token_name).to_owned(),
            serde_json::json!(token_id),
        );
        added_tokens.push(added_special_token(token_id, control_token_name));
    }
    for (word_index, word) in WORDS.iter().enumerate() {
        vocabulary.insert(
            (*word).to_owned(),
            serde_json::json!(FIRST_WORD_TOKEN_ID + word_index as u32),
        );
    }
    serde_json::to_vec(&serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocabulary, "unk_token": "<unk>" },
    }))
    .expect("the fixture tokenizer serializes")
}

/// The token of one of `WORDS`.
pub(super) fn word_token_id(word: &str) -> u32 {
    let word_index = WORDS
        .iter()
        .position(|known_word| *known_word == word)
        .expect("the word is in the fixture vocabulary");
    FIRST_WORD_TOKEN_ID + word_index as u32
}
