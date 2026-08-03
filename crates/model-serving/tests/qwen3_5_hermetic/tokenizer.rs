use astronomical_model_serving::{
    Qwen3_5ImageProcessor, Qwen3_5Tokenizer, Qwen3_5TokenizerError, validate_context_token_count,
};

const ORNITH_VOCABULARY_SIZE: u32 = 248_320;
const ORNITH_MAXIMUM_POSITION_COUNT: u32 = 262_144;

#[test]
fn should_discover_special_token_ids_from_tokenizer_json() {
    let tokenizer = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes(248_056),
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        Qwen3_5ImageProcessor::qwen3_5_moe_35b_optiq(),
    )
    .expect("the synthetic tokenizer should discover every special token ID");

    assert_eq!(tokenizer.image_pad_token_id(), 248_056);
    assert_eq!(tokenizer.end_of_text_token_id(), 248_044);
    assert_eq!(tokenizer.im_start_token_id(), 248_045);
    assert_eq!(tokenizer.im_end_token_id(), 248_046);
    assert_eq!(tokenizer.think_start_token_id(), 248_068);
    assert_eq!(tokenizer.think_end_token_id(), 248_069);
}

#[test]
fn should_reject_a_tokenizer_missing_a_required_special_token() {
    let tokenizer_error = Qwen3_5Tokenizer::from_json_bytes(
        &ornith_tokenizer_json_bytes_missing_image_pad(),
        ORNITH_VOCABULARY_SIZE,
        ORNITH_MAXIMUM_POSITION_COUNT,
        Qwen3_5ImageProcessor::qwen3_5_moe_35b_optiq(),
    )
    .expect_err("the tokenizer should reject a missing special token");

    assert!(matches!(
        tokenizer_error,
        Qwen3_5TokenizerError::DiscoverTokenIds { .. }
    ));
}

#[test]
fn should_accept_an_opencode_context_above_the_retired_32k_input_limit() {
    assert!(validate_context_token_count(40_665, 4_096, 262_144).is_ok());
}

#[test]
fn should_reject_a_context_above_the_certified_ornith_position_limit() {
    assert!(validate_context_token_count(258_049, 4_096, 262_144).is_err());
}

fn ornith_tokenizer_json_bytes(image_pad_token_id: u32) -> Vec<u8> {
    let vocab = serde_json::json!({
        "<unk>": 0,
        "<unk>": 0,
        "<|endoftext|>": 248_044,
        "<|im_start|>": 248_045,
        "<|im_end|>": 248_046,
        "<|image_pad|>": image_pad_token_id,
        "<tool_call>": 248_058,
        "</tool_call>": 248_059,
        "<tool_response>": 248_066,
        "</tool_response>": 248_067,
        "<think>": 248_068,
        "</think>": 248_069
    });
    serde_json::to_vec(&serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>"
        }
    }))
    .expect("the synthetic tokenizer JSON should serialize")
}

fn ornith_tokenizer_json_bytes_missing_image_pad() -> Vec<u8> {
    let vocab = serde_json::json!({
        "<unk>": 0,
        "<|endoftext|>": 248_044,
        "<|im_start|>": 248_045,
        "<|im_end|>": 248_046,
        "<tool_call>": 248_058,
        "</tool_call>": 248_059,
        "<tool_response>": 248_066,
        "</tool_response>": 248_067,
        "<think>": 248_068,
        "</think>": 248_069,
    });
    serde_json::to_vec(&serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>"
        }
    }))
    .expect("the synthetic tokenizer JSON should serialize")
}
