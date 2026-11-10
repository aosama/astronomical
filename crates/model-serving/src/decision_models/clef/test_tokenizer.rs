//! A word-level tokenizer with Qwen's chat markers, for unit tests of Clef's
//! layout: every word below is one token, chat markers are special tokens,
//! and anything else reads as the unknown token.

pub(super) const UNKNOWN_TOKEN_ID: u32 = 0;
pub(super) const CHAT_MARKERS: [&str; 2] = ["<|im_start|>", "<|im_end|>"];
pub(super) const FIRST_WORD_TOKEN_ID: u32 = 10;
pub(super) const WORDS: [&str; 4] = ["to", "be", "or", "not"];

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

/// The tokenizer's JSON, with `chat_markers` as its special tokens.
pub(super) fn tokenizer_bytes_with(chat_markers: [&str; 2]) -> Vec<u8> {
    let mut vocabulary = serde_json::Map::new();
    vocabulary.insert("<unk>".to_owned(), serde_json::json!(UNKNOWN_TOKEN_ID));
    let mut added_tokens = vec![added_special_token(UNKNOWN_TOKEN_ID, "<unk>")];
    for (marker_index, chat_marker) in chat_markers.iter().enumerate() {
        let token_id = 1 + marker_index as u32;
        vocabulary.insert((*chat_marker).to_owned(), serde_json::json!(token_id));
        added_tokens.push(added_special_token(token_id, chat_marker));
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
