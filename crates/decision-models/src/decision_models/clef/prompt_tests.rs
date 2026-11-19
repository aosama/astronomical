//! Unit tests for Clef's sequence layout: spans, the state cut and the
//! schema limit.

use astronomical_decision_contract::{
    DecisionQuestion, OrderedJsonValue, parse_system_one_questions,
};

use super::{CLEF_MAXIMUM_SEQUENCE_TOKENS, build_clef_sequence};
use crate::decision_models::clef::test_tokenizer::{
    CHAT_MARKERS, FIRST_WORD_TOKEN_ID, tokenizer_bytes_with,
};
use crate::decision_models::clef::tokenizer::ClefTokenizer;

fn fixture_tokenizer() -> ClefTokenizer {
    ClefTokenizer::from_json_bytes(&tokenizer_bytes_with(CHAT_MARKERS))
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
fn should_place_each_questions_instructions_and_options_on_their_own_spans() {
    let questions = questions_from(
        r#"{"q": {"type": "choice", "instructions": "to be or not", "criteria": {"or": null, "be": null}}}"#,
    );

    let sequence = build_clef_sequence(&fixture_tokenizer(), &text_of_words("be", 3), &questions)
        .expect("the request lays out");

    assert_eq!(
        sequence.token_ids[0], 1,
        "the sequence opens with <|im_start|>"
    );
    let question_layout = &sequence.questions[0];
    let instructions_span = question_layout.instructions_span;
    assert_eq!(
        sequence.token_ids[instructions_span.start..instructions_span.end],
        [
            FIRST_WORD_TOKEN_ID,
            FIRST_WORD_TOKEN_ID + 1,
            FIRST_WORD_TOKEN_ID + 2,
            FIRST_WORD_TOKEN_ID + 3
        ]
    );
    assert_eq!(question_layout.option_spans.len(), 2);
    let option_ids: Vec<&str> = question_layout
        .question_text
        .options
        .iter()
        .map(|option_text| option_text.option_id.as_str())
        .collect();
    assert_eq!(
        option_ids,
        ["be", "or"],
        "choice options are laid out in key order"
    );
    assert!(question_layout.option_spans[0].end <= question_layout.option_spans[1].start);
    assert_eq!(sequence.state_token_count, 3);
    assert_eq!(sequence.state_tokens_read, 3);
}

#[test]
fn should_cut_a_long_state_from_its_end_to_fill_the_sequence_exactly() {
    let questions = questions_from(r#"{"q": {"type": "noul", "instructions": "to be"}}"#);
    let state_word_count = CLEF_MAXIMUM_SEQUENCE_TOKENS + 100;

    let sequence = build_clef_sequence(
        &fixture_tokenizer(),
        &text_of_words("or", state_word_count),
        &questions,
    )
    .expect("a long state is cut, not refused");

    assert_eq!(sequence.token_ids.len(), CLEF_MAXIMUM_SEQUENCE_TOKENS);
    assert_eq!(sequence.state_token_count, state_word_count);
    assert!(sequence.state_tokens_read < state_word_count);
    let instructions_span = sequence.questions[0].instructions_span;
    assert_eq!(
        sequence.token_ids[instructions_span.start..instructions_span.end],
        [FIRST_WORD_TOKEN_ID, FIRST_WORD_TOKEN_ID + 1],
        "spans still point at the instructions after the cut"
    );
}

#[test]
fn should_keep_a_state_that_exactly_fills_the_sequence_whole() {
    let questions = questions_from(r#"{"q": {"type": "noul", "instructions": "to be"}}"#);
    let empty_state_length =
        build_clef_sequence(&fixture_tokenizer(), &text_of_words("or", 0), &questions)
            .expect("an empty state lays out")
            .token_ids
            .len();
    let fitting_word_count = CLEF_MAXIMUM_SEQUENCE_TOKENS - empty_state_length;

    let sequence = build_clef_sequence(
        &fixture_tokenizer(),
        &text_of_words("or", fitting_word_count),
        &questions,
    )
    .expect("a state that fits lays out");

    assert_eq!(sequence.token_ids.len(), CLEF_MAXIMUM_SEQUENCE_TOKENS);
    assert_eq!(sequence.state_tokens_read, fitting_word_count);
    assert_eq!(sequence.state_token_count, fitting_word_count);
}

#[test]
fn should_refuse_questions_that_alone_exceed_the_sequence_limit() {
    let long_instructions = vec!["be"; CLEF_MAXIMUM_SEQUENCE_TOKENS].join(" ");
    let questions = questions_from(&format!(
        r#"{{"q": {{"type": "noul", "instructions": "{long_instructions}"}}}}"#
    ));

    let Err(refusal) =
        build_clef_sequence(&fixture_tokenizer(), &text_of_words("or", 1), &questions)
    else {
        panic!("a schema longer than the limit leaves no room for any state");
    };

    assert!(refusal.to_string().contains("before any state"));
}

#[test]
fn should_refuse_a_tokenizer_without_the_chat_markers() {
    assert!(ClefTokenizer::from_json_bytes(&tokenizer_bytes_with(["<|a|>", "<|b|>"])).is_err());
}
