//! Unit tests for resolving a Laya tokenizer's special tokens.

use astronomical_decision_contract::DecisionTokenEncoder;

use super::{LayaTokenizer, LayaTokenizerError};

const UNKNOWN_TOKEN_ID: u32 = 0;
const CLS_TOKEN_ID: u32 = 1;
const SEP_TOKEN_ID: u32 = 2;
const MASK_TOKEN_ID: u32 = 3;
const PAD_TOKEN_ID: u32 = 4;

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

/// A word-level tokenizer over a few Shakespeare words, with special tokens
/// named `special_token_names` in the order unknown, cls, sep, mask, pad.
fn fixture_tokenizer_bytes(special_token_names: [&str; 5]) -> Vec<u8> {
    let mut vocabulary = serde_json::Map::new();
    for (token_id, special_token_name) in special_token_names.iter().enumerate() {
        vocabulary.insert(
            (*special_token_name).to_owned(),
            serde_json::json!(token_id),
        );
    }
    for (word_index, word) in ["to", "be", "or", "not"].iter().enumerate() {
        vocabulary.insert((*word).to_owned(), serde_json::json!(10 + word_index));
    }
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            added_special_token(UNKNOWN_TOKEN_ID, special_token_names[0]),
            added_special_token(CLS_TOKEN_ID, special_token_names[1]),
            added_special_token(SEP_TOKEN_ID, special_token_names[2]),
            added_special_token(MASK_TOKEN_ID, special_token_names[3]),
            added_special_token(PAD_TOKEN_ID, special_token_names[4]),
        ],
        "normalizer": null,
        "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                { "SpecialToken": { "id": special_token_names[1], "type_id": 0 } },
                { "Sequence": { "id": "A", "type_id": 0 } },
                { "SpecialToken": { "id": special_token_names[2], "type_id": 0 } },
            ],
            "pair": [
                { "Sequence": { "id": "A", "type_id": 0 } },
                { "Sequence": { "id": "B", "type_id": 0 } },
            ],
            "special_tokens": {
                special_token_names[1]: { "id": special_token_names[1], "ids": [CLS_TOKEN_ID], "tokens": [special_token_names[1]] },
                special_token_names[2]: { "id": special_token_names[2], "ids": [SEP_TOKEN_ID], "tokens": [special_token_names[2]] },
            },
        },
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "unk_token": special_token_names[0],
            "vocab": vocabulary,
        },
    });
    serde_json::to_vec(&tokenizer_json).expect("the fixture tokenizer should serialize")
}

const BRACKETED_NAMES: [&str; 5] = ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]"];

fn load_error(tokenizer_json_bytes: &[u8], tokenizer_settings_json: &str) -> LayaTokenizerError {
    match LayaTokenizer::from_artifact_bytes(
        tokenizer_json_bytes,
        tokenizer_settings_json.as_bytes(),
    ) {
        Ok(_) => panic!("the tokenizer must be refused"),
        Err(tokenizer_error) => tokenizer_error,
    }
}

#[test]
fn should_resolve_special_tokens_named_in_the_settings() {
    let tokenizer = LayaTokenizer::from_artifact_bytes(
        &fixture_tokenizer_bytes(["<unk>", "<bos>", "<eos>", "<mask>", "<pad>"]),
        br#"{"cls_token": "<bos>", "sep_token": "<eos>", "mask_token": "<mask>", "pad_token": "<pad>"}"#,
    )
    .expect("the tokenizer should load");
    assert_eq!(tokenizer.special_tokens.cls_token_id, CLS_TOKEN_ID);
    assert_eq!(tokenizer.special_tokens.sep_token_id, SEP_TOKEN_ID);
    assert_eq!(tokenizer.special_tokens.mask_token_id, MASK_TOKEN_ID);
    assert_eq!(tokenizer.special_tokens.mask_token_text, "<mask>");
    assert_eq!(tokenizer.pad_token_id, PAD_TOKEN_ID);
}

#[test]
fn should_read_a_special_token_given_as_an_object_with_content() {
    let tokenizer = LayaTokenizer::from_artifact_bytes(
        &fixture_tokenizer_bytes(BRACKETED_NAMES),
        br#"{"mask_token": {"content": "[MASK]", "lstrip": true}}"#,
    )
    .expect("the tokenizer should load");
    assert_eq!(tokenizer.special_tokens.mask_token_id, MASK_TOKEN_ID);
}

#[test]
fn should_fall_back_to_the_bracketed_names_when_the_settings_omit_them() {
    let tokenizer =
        LayaTokenizer::from_artifact_bytes(&fixture_tokenizer_bytes(BRACKETED_NAMES), b"{}")
            .expect("the tokenizer should load");
    assert_eq!(tokenizer.special_tokens.cls_token_id, CLS_TOKEN_ID);
    assert_eq!(tokenizer.special_tokens.mask_token_text, "[MASK]");
}

#[test]
fn should_encode_text_without_adding_the_declared_special_tokens() {
    let tokenizer =
        LayaTokenizer::from_artifact_bytes(&fixture_tokenizer_bytes(BRACKETED_NAMES), b"{}")
            .expect("the tokenizer should load");
    assert_eq!(
        tokenizer.encode_without_special_tokens("to be or not to be"),
        Ok(vec![10, 11, 12, 13, 10, 11])
    );
    assert_eq!(tokenizer.encode_without_special_tokens(""), Ok(Vec::new()));
}

#[test]
fn should_name_a_special_token_the_tokenizer_does_not_have() {
    assert_eq!(
        load_error(
            &fixture_tokenizer_bytes(BRACKETED_NAMES),
            r#"{"mask_token": "<mask>"}"#
        ),
        LayaTokenizerError::UnknownSpecialToken {
            role: "mask",
            token_text: "<mask>".to_owned(),
        }
    );
}

#[test]
fn should_refuse_settings_that_are_not_json() {
    assert!(matches!(
        load_error(&fixture_tokenizer_bytes(BRACKETED_NAMES), "{not json"),
        LayaTokenizerError::MalformedSettings { .. }
    ));
}

#[test]
fn should_refuse_tokenizer_bytes_that_do_not_parse() {
    assert!(matches!(
        load_error(b"not a tokenizer", "{}"),
        LayaTokenizerError::Unloadable { .. }
    ));
}
