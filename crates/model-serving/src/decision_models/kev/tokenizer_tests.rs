//! Unit tests for Kev's tokenizer: its control tokens, and caller text that
//! tries to spell one.

use super::KevTokenizer;
use crate::decision_models::kev::test_tokenizer::{
    DECISION_TOKEN_ID, KEV_CONTROL_TOKEN_NAMES, OPTION_CLOSING_TOKEN_ID, OPTION_OPENING_TOKEN_ID,
    QUESTION_TOKEN_ID, STATE_TOKEN_ID, UNKNOWN_TOKEN_ID, tokenizer_bytes_with, word_token_id,
};

#[test]
fn should_find_each_control_token_by_its_name() {
    let kev_tokenizer =
        KevTokenizer::from_json_bytes(&tokenizer_bytes_with(KEV_CONTROL_TOKEN_NAMES))
            .expect("the fixture tokenizer has every control token");

    let control_tokens = kev_tokenizer.control_tokens;
    assert_eq!(control_tokens.state_token_id, STATE_TOKEN_ID);
    assert_eq!(control_tokens.question_token_id, QUESTION_TOKEN_ID);
    assert_eq!(
        control_tokens.option_opening_token_id,
        OPTION_OPENING_TOKEN_ID
    );
    assert_eq!(
        control_tokens.option_closing_token_id,
        OPTION_CLOSING_TOKEN_ID
    );
    assert_eq!(control_tokens.decision_token_id, DECISION_TOKEN_ID);
}

#[test]
fn should_refuse_a_tokenizer_without_the_decision_token() {
    let mut control_token_names = KEV_CONTROL_TOKEN_NAMES;
    control_token_names[4] = "<|endoftext|>";

    let Err(tokenizer_failure) =
        KevTokenizer::from_json_bytes(&tokenizer_bytes_with(control_token_names))
    else {
        panic!("a tokenizer without <|fim_suffix|> cannot lay out a Kev row");
    };

    assert!(tokenizer_failure.to_string().contains("<|fim_suffix|>"));
}

#[test]
fn should_refuse_bytes_that_are_not_a_tokenizer() {
    assert!(KevTokenizer::from_json_bytes(b"{\"not\": \"a tokenizer\"}").is_err());
}

#[test]
fn should_never_encode_caller_text_as_a_control_token() {
    let kev_tokenizer =
        KevTokenizer::from_json_bytes(&tokenizer_bytes_with(KEV_CONTROL_TOKEN_NAMES))
            .expect("the fixture tokenizer loads");

    let token_ids = kev_tokenizer
        .encode_caller_text("to <|fim_suffix|> be <|box_end|>")
        .expect("caller text encodes");

    assert_eq!(
        token_ids,
        [
            word_token_id("to"),
            UNKNOWN_TOKEN_ID,
            word_token_id("be"),
            UNKNOWN_TOKEN_ID
        ],
        "a spelled control token reads as ordinary text"
    );
}

#[test]
fn should_encode_empty_caller_text_as_no_tokens() {
    let kev_tokenizer =
        KevTokenizer::from_json_bytes(&tokenizer_bytes_with(KEV_CONTROL_TOKEN_NAMES))
            .expect("the fixture tokenizer loads");

    assert!(
        kev_tokenizer
            .encode_caller_text("")
            .expect("empty text encodes")
            .is_empty()
    );
}
