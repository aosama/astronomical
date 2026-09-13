//! Hermetic reproduction of the issue #596 tokenizer gap.
//!
//! The nomicai-modernbert-embed-base artifact declares a `TemplateProcessing`
//! post-processor that wraps every input as `[CLS] content [SEP]`, and its
//! `config.json` publishes the matching `cls_token_id`/`sep_token_id`. The
//! production encoder must therefore submit those declared special tokens to
//! the bidirectional encoder — the model was trained and evaluated with them,
//! and the published worked-example cosines assume them.
//!
//! This test builds a fixture tokenizer carrying the same post-processor
//! declaration, drives the real production `encode_embedding_input`, and
//! asserts the encoded sequence contains both declared specials and that
//! pooling exclusion has real tokens to act on. It runs hermetically: no GPU,
//! no installed checkpoint, no Python.

use astronomical_model_serving::{ModernBertConfiguration, encode_embedding_input};
use tokenizers::Tokenizer;

const CLS_TOKEN_ID: u32 = 50_281;
const SEP_TOKEN_ID: u32 = 50_282;
const PAD_TOKEN_ID: u32 = 50_283;

/// Upstream worked-example inputs, exactly as the model card instructs callers
/// to prefix them (`search_query: ` / `search_document: `).
const TSNE_QUERY: &str = "search_query: What is TSNE?";
const LAURENS_QUERY: &str = "search_query: Who is Laurens van der Maaten?";
const TSNE_DOCUMENT: &str = "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten";

#[test]
fn should_encode_embedding_inputs_with_the_artifact_declared_special_tokens() {
    let configuration = fixture_configuration();
    let tokenizer = fixture_tokenizer();

    for input_text in [TSNE_QUERY, LAURENS_QUERY, TSNE_DOCUMENT] {
        let encoded_input = encode_embedding_input(&tokenizer, &configuration, input_text)
            .unwrap_or_else(|failure| {
                panic!("the declared worked example should encode: {failure:?}")
            });

        let content_only_ids = fixture_content_ids(input_text);
        assert_eq!(
            encoded_input.token_ids.first(),
            Some(&CLS_TOKEN_ID),
            "the sequence submitted to the encoder must start with the artifact-declared [CLS]: input={input_text:?} ids={:?}",
            encoded_input.token_ids
        );
        assert_eq!(
            encoded_input.token_ids.last(),
            Some(&SEP_TOKEN_ID),
            "the sequence submitted to the encoder must end with the artifact-declared [SEP]: input={input_text:?} ids={:?}",
            encoded_input.token_ids
        );
        assert_eq!(
            encoded_input.token_ids.len(),
            content_only_ids.len() + 2,
            "the declared post-processor adds exactly one [CLS] and one [SEP]: input={input_text:?}"
        );
        assert_eq!(
            &encoded_input.token_ids[1..encoded_input.token_ids.len() - 1],
            content_only_ids.as_slice(),
            "content token identity must be preserved between the declared and served encodings"
        );
    }
}

/// Builds the bounded forward-pass geometry matching the published artifact.
fn fixture_configuration() -> ModernBertConfiguration {
    let config_value = serde_json::json!({
        "hidden_size": 768,
        "num_hidden_layers": 22,
        "num_attention_heads": 12,
        "max_position_embeddings": 8_192,
        "local_attention": 128,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160_000.0,
        "local_rope_theta": 10_000.0,
        "layer_norm_eps": 1e-5,
        "pad_token_id": PAD_TOKEN_ID,
        "cls_token_id": CLS_TOKEN_ID,
        "sep_token_id": SEP_TOKEN_ID,
        "quantization": { "group_size": 64, "bits": 8 },
    });
    ModernBertConfiguration::from_config_value(&config_value)
        .expect("the published artifact geometry should validate")
}

/// Builds a tokenizer whose post-processor matches the artifact declaration:
/// `TemplateProcessing(single=[CLS] A [SEP])` over a minimal word vocabulary.
fn fixture_tokenizer() -> Tokenizer {
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            added_special_token(50_280, "[UNK]"),
            added_special_token(CLS_TOKEN_ID, "[CLS]"),
            added_special_token(SEP_TOKEN_ID, "[SEP]"),
            added_special_token(PAD_TOKEN_ID, "[PAD]"),
        ],
        "normalizer": null,
        "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                { "SpecialToken": { "id": "[CLS]", "type_id": 0 } },
                { "Sequence": { "id": "A", "type_id": 0 } },
                { "SpecialToken": { "id": "[SEP]", "type_id": 0 } },
            ],
            "pair": [
                { "SpecialToken": { "id": "[CLS]", "type_id": 0 } },
                { "Sequence": { "id": "A", "type_id": 0 } },
                { "SpecialToken": { "id": "[SEP]", "type_id": 0 } },
                { "Sequence": { "id": "B", "type_id": 0 } },
                { "SpecialToken": { "id": "[SEP]", "type_id": 0 } },
            ],
            "special_tokens": {
                "[CLS]": { "id": "[CLS]", "ids": [CLS_TOKEN_ID], "tokens": ["[CLS]"] },
                "[SEP]": { "id": "[SEP]", "ids": [SEP_TOKEN_ID], "tokens": ["[SEP]"] },
            },
        },
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "unk_token": "[UNK]",
            "vocab": fixture_vocabulary(),
        },
    });
    Tokenizer::from_bytes(
        serde_json::to_vec(&tokenizer_json)
            .expect("the fixture tokenizer document should serialize")
            .as_slice(),
    )
    .expect("the fixture tokenizer should parse")
}

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

/// WordLevel vocabulary covering every word the fixture inputs contain.
fn fixture_vocabulary() -> serde_json::Map<String, serde_json::Value> {
    let mut vocabulary = serde_json::Map::new();
    vocabulary.insert("[UNK]".to_owned(), serde_json::json!(50_280));
    vocabulary.insert("[CLS]".to_owned(), serde_json::json!(CLS_TOKEN_ID));
    vocabulary.insert("[SEP]".to_owned(), serde_json::json!(SEP_TOKEN_ID));
    vocabulary.insert("[PAD]".to_owned(), serde_json::json!(PAD_TOKEN_ID));
    for (word_position, word) in [
        "search_query:",
        "What",
        "is",
        "TSNE?",
        "Who",
        "Laurens",
        "van",
        "der",
        "Maaten?",
        "search_document:",
        "TSNE",
        "a",
        "dimensionality",
        "reduction",
        "algorithm",
        "created",
        "by",
        "Der",
        "Maaten",
    ]
    .into_iter()
    .enumerate()
    {
        vocabulary.insert(word.to_owned(), serde_json::json!(1 + word_position as u64));
    }
    vocabulary
}

/// Encodes one fixture input through the fixture tokenizer with the
/// post-processor disabled — the content-only baseline the served sequence
/// must extend, not replace.
fn fixture_content_ids(input_text: &str) -> Vec<u32> {
    let tokenizer = fixture_tokenizer();
    let encoding = tokenizer
        .encode(input_text, false)
        .expect("the fixture input should encode without specials");
    encoding.get_ids().to_vec()
}
