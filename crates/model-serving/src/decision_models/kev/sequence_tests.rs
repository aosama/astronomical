//! Unit tests for Kev's request layout: the rows, their readout offsets, and
//! the state and row limits.

use astronomical_decision_contract::{
    DecisionQuestion, OrderedJsonValue, parse_system_one_questions,
};

use super::{KEV_MAXIMUM_ROW_TOKENS, KEV_MAXIMUM_STATE_TOKENS, build_kev_sequence};
use crate::decision_models::kev::test_tokenizer::{
    DECISION_TOKEN_ID, KEV_CONTROL_TOKEN_NAMES, OPTION_CLOSING_TOKEN_ID, OPTION_OPENING_TOKEN_ID,
    QUESTION_TOKEN_ID, STATE_TOKEN_ID, tokenizer_bytes_with, word_token_id,
};
use crate::decision_models::kev::tokenizer::KevTokenizer;

fn fixture_tokenizer() -> KevTokenizer {
    KevTokenizer::from_json_bytes(&tokenizer_bytes_with(KEV_CONTROL_TOKEN_NAMES))
        .expect("the fixture tokenizer loads")
}

fn questions_from(questions_json_text: &str) -> Vec<DecisionQuestion> {
    let questions: OrderedJsonValue =
        serde_json::from_str(questions_json_text).expect("test questions are JSON");
    parse_system_one_questions(&questions).expect("test questions are valid")
}

fn text_of_words(word: &str, word_count: usize) -> OrderedJsonValue {
    OrderedJsonValue::Text(vec![word; word_count].join(" "))
}

#[test]
fn should_lay_out_the_state_then_each_question_as_its_own_row() {
    let questions = questions_from(r#"{"q": {"type": "noul", "instructions": "to be or not"}}"#);

    let request_sequence =
        build_kev_sequence(&fixture_tokenizer(), &text_of_words("be", 2), &questions)
            .expect("the request lays out");

    let be_token_id = word_token_id("be");
    assert_eq!(
        request_sequence.state_token_ids,
        [STATE_TOKEN_ID, be_token_id, be_token_id]
    );
    let question_row = &request_sequence.question_rows[0];
    assert_eq!(
        question_row.token_ids,
        [
            QUESTION_TOKEN_ID,
            word_token_id("to"),
            be_token_id,
            word_token_id("or"),
            word_token_id("not"),
            OPTION_OPENING_TOKEN_ID,
            word_token_id("no"),
            OPTION_CLOSING_TOKEN_ID,
            OPTION_OPENING_TOKEN_ID,
            word_token_id("yes"),
            OPTION_CLOSING_TOKEN_ID,
            DECISION_TOKEN_ID,
        ]
    );
    assert_eq!(question_row.option_closing_offsets, [7, 10]);
    assert_eq!(question_row.decision_offset(), 11);
}

#[test]
fn should_place_row_layout_offsets_after_the_state() {
    let questions = questions_from(
        r#"{"first": {"type": "noul"}, "second": {"type": "noul", "instructions": "to be"}}"#,
    );

    let row_layouts = build_kev_sequence(&fixture_tokenizer(), &text_of_words("or", 3), &questions)
        .expect("the request lays out")
        .row_layouts();

    assert_eq!(row_layouts.len(), 2);
    assert_eq!(row_layouts[0].state_token_count, 4);
    assert_eq!(row_layouts[0].option_closing_offsets, [4 + 3, 4 + 6]);
    assert_eq!(row_layouts[1].option_closing_offsets, [4 + 5, 4 + 8]);
    assert_eq!(
        &row_layouts[1].token_ids[..4],
        &row_layouts[0].token_ids[..4]
    );
}

#[test]
fn should_count_options_that_read_as_the_same_tokens_once() {
    let questions = questions_from(
        r#"{"q": {"type": "choice", "criteria": {"thou": null, "thee": null, "be": null}}}"#,
    );

    let request_sequence =
        build_kev_sequence(&fixture_tokenizer(), &text_of_words("to", 1), &questions)
            .expect("the request lays out");

    assert_eq!(
        request_sequence.question_rows[0].distinguishable_option_count, 2,
        "two words outside the vocabulary both read as the unknown token"
    );
}

#[test]
fn should_accept_a_state_exactly_at_the_limit() {
    let questions = questions_from(r#"{"q": {"type": "noul"}}"#);
    let state = text_of_words("to", KEV_MAXIMUM_STATE_TOKENS - 1);

    let request_sequence = build_kev_sequence(&fixture_tokenizer(), &state, &questions)
        .expect("a state of exactly the limit, its state token included, is served");

    assert_eq!(
        request_sequence.state_token_ids.len(),
        KEV_MAXIMUM_STATE_TOKENS
    );
}

#[test]
fn should_refuse_a_state_one_token_over_the_limit_instead_of_cutting_it() {
    let questions = questions_from(r#"{"q": {"type": "noul"}}"#);
    let state = text_of_words("to", KEV_MAXIMUM_STATE_TOKENS);

    let Err(refusal) = build_kev_sequence(&fixture_tokenizer(), &state, &questions) else {
        panic!("a state over the limit must be refused");
    };

    assert!(refusal.to_string().contains("65537 tokens"));
}

#[test]
fn should_refuse_a_question_whose_row_is_over_the_row_limit() {
    let row_room = KEV_MAXIMUM_ROW_TOKENS - KEV_MAXIMUM_STATE_TOKENS;
    let long_instructions = vec!["be"; row_room].join(" ");
    let questions = questions_from(&format!(
        r#"{{"long": {{"type": "noul", "instructions": "{long_instructions}"}}}}"#
    ));
    let state = text_of_words("to", KEV_MAXIMUM_STATE_TOKENS - 1);

    let Err(refusal) = build_kev_sequence(&fixture_tokenizer(), &state, &questions) else {
        panic!("a row over the limit must be refused");
    };

    assert!(refusal.to_string().contains("row limit"));
}
