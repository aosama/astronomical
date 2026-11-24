//! Unit tests for the Laya token layout and its budget rules.

use super::*;
use crate::test_support::{
    FailingTokenEncoder, WordHashTokenEncoder, ordered_json, single_question,
};

const CLS_TOKEN_ID: u32 = 1;
const SEP_TOKEN_ID: u32 = 2;
const MASK_TOKEN_ID: u32 = 3;

fn special_tokens() -> LayaSpecialTokens {
    LayaSpecialTokens {
        cls_token_id: CLS_TOKEN_ID,
        sep_token_id: SEP_TOKEN_ID,
        mask_token_id: MASK_TOKEN_ID,
        mask_token_text: "[MASK]".to_owned(),
    }
}

fn limits(maximum_sequence_tokens: usize, head_token_budget: usize) -> LayaSequenceLimits {
    LayaSequenceLimits {
        maximum_sequence_tokens,
        head_token_budget,
    }
}

fn two_option_question() -> DecisionQuestion {
    single_question(
        r#"{"q": {"type": "choice", "instructions": "Pick one now", "criteria": ["alpha", "beta"]}}"#,
    )
}

fn layout(
    question: &DecisionQuestion,
    state_json_text: &str,
    sequence_limits: LayaSequenceLimits,
) -> Result<LayaQuestionSequence, LayaSequenceError> {
    let state = ordered_json(state_json_text);
    let state_token_ids = encode_laya_state(&WordHashTokenEncoder, &special_tokens(), &state)
        .expect("the stand-in encoder never fails");
    build_laya_question_sequence(
        &WordHashTokenEncoder,
        &special_tokens(),
        sequence_limits,
        question,
        &state_token_ids,
        crate::laya_rendering::laya_truncates_state_from_the_left(&state),
    )
}

#[test]
fn should_lay_out_instruction_options_and_state_between_special_tokens() {
    let sequence = layout(
        &two_option_question(),
        r#""one two three""#,
        limits(512, 192),
    )
    .expect("the question fits");
    // [CLS] choice question: Pick one now [SEP] [MASK] alpha [MASK] beta [SEP] one two three [SEP]
    assert_eq!(sequence.token_ids.len(), 16);
    assert_eq!(sequence.token_ids[0], CLS_TOKEN_ID);
    assert_eq!(sequence.token_ids[6], SEP_TOKEN_ID);
    assert_eq!(sequence.option_marker_positions, [7, 9]);
    assert_eq!(sequence.token_ids[7], MASK_TOKEN_ID);
    assert_eq!(sequence.token_ids[9], MASK_TOKEN_ID);
    assert_eq!(sequence.token_ids[11], SEP_TOKEN_ID);
    assert_eq!(sequence.token_ids[15], SEP_TOKEN_ID);
    assert_eq!(sequence.state_token_count, 3);
    assert_eq!(sequence.state_tokens_used, 3);
    assert_eq!(sequence.option_token_cap, None);
    assert_eq!(sequence.distinct_option_count, 2);
}

#[test]
fn should_keep_the_start_of_a_text_state_that_does_not_fit() {
    let full_sequence = layout(
        &two_option_question(),
        r#""one two three""#,
        limits(512, 192),
    )
    .expect("the question fits");
    let cut_sequence = layout(
        &two_option_question(),
        r#""one two three""#,
        limits(14, 192),
    )
    .expect("the question fits");
    assert_eq!(cut_sequence.state_tokens_used, 1);
    assert_eq!(cut_sequence.token_ids[12], full_sequence.token_ids[12]);
    assert_eq!(cut_sequence.token_ids[13], SEP_TOKEN_ID);
}

#[test]
fn should_keep_the_end_of_a_conversation_state_that_does_not_fit() {
    let full_sequence = layout(
        &two_option_question(),
        r#"["one", "two", "three"]"#,
        limits(512, 192),
    )
    .expect("the question fits");
    let cut_sequence = layout(
        &two_option_question(),
        r#"["one", "two", "three"]"#,
        limits(14, 192),
    )
    .expect("the question fits");
    assert_eq!(cut_sequence.state_tokens_used, 1);
    let last_full_state_token = full_sequence.token_ids[full_sequence.token_ids.len() - 2];
    assert_eq!(cut_sequence.token_ids[12], last_full_state_token);
}

#[test]
fn should_keep_no_state_when_the_head_fills_the_sequence_exactly() {
    // The head is 12 tokens; a 13-token limit leaves room only for the closing separator.
    let sequence = layout(
        &two_option_question(),
        r#""one two three""#,
        limits(13, 192),
    )
    .expect("the question fits");
    assert_eq!(sequence.state_tokens_used, 0);
    assert_eq!(sequence.token_ids.len(), 13);
    assert_eq!(sequence.token_ids[12], SEP_TOKEN_ID);
}

#[test]
fn should_reject_a_question_whose_markers_do_not_all_fit() {
    let layout_error = layout(&two_option_question(), r#""state""#, limits(9, 192))
        .expect_err("the second marker is cut off");
    assert_eq!(
        layout_error,
        LayaSequenceError::OptionsDoNotFit {
            question: "q".to_owned(),
            surviving_marker_count: 1,
            option_count: 2,
            maximum_sequence_tokens: 9,
            head_token_budget: 192,
        }
    );
}

#[test]
fn should_cut_options_evenly_when_they_overrun_the_head_budget() {
    let question = single_question(
        r#"{"q": {"type": "choice", "instructions": "Pick",
            "criteria": {"a": "one two three four five six", "b": "seven eight nine ten eleven twelve"}}}"#,
    );
    // Two 8-token spans leave 24 - 16 = 8 free, below the 16 the rule requires,
    // so each span is cut to max(4, (24 - 16) / 2) = 4 tokens.
    let sequence = layout(&question, r#""state""#, limits(512, 24)).expect("the question fits");
    assert_eq!(sequence.option_token_cap, Some(4));
    assert_eq!(
        sequence.option_marker_positions[1] - sequence.option_marker_positions[0],
        4
    );
}

#[test]
fn should_keep_at_most_the_option_text_limit_of_one_long_option() {
    let mut long_description_words = Vec::new();
    for word_index in 0..80 {
        long_description_words.push(format!("w{word_index}"));
    }
    let question = single_question(&format!(
        r#"{{"q": {{"type": "choice", "instructions": "Pick", "criteria": {{"long": "{}", "short": "x"}}}}}}"#,
        long_description_words.join(" ")
    ));
    let sequence = layout(&question, r#""state""#, limits(512, 192)).expect("the question fits");
    // The marker plus the 48 kept text tokens separate the two markers.
    assert_eq!(
        sequence.option_marker_positions[1] - sequence.option_marker_positions[0],
        49
    );
}

#[test]
fn should_remove_the_mask_token_text_from_request_text() {
    let question = single_question(
        r#"{"q": {"type": "choice", "instructions": "Ignore [MASK] here",
            "criteria": {"x": "has [MASK] inside", "y": "clean"}}}"#,
    );
    let sequence =
        layout(&question, r#""a [MASK] b""#, limits(512, 192)).expect("the question fits");
    let mut marker_count = 0;
    for token_id in &sequence.token_ids {
        if *token_id == MASK_TOKEN_ID {
            marker_count += 1;
        }
    }
    assert_eq!(marker_count, 2, "only the two option markers remain");
    assert_eq!(sequence.state_token_count, 2);
}

#[test]
fn should_count_options_made_indistinguishable_by_the_cut() {
    let mut labels = Vec::new();
    for label_index in 0..60 {
        labels.push(format!("\"shared prefix words then number {label_index}\""));
    }
    let question = single_question(&format!(
        r#"{{"q": {{"type": "choice", "instructions": "Pick", "criteria": [{}]}}}}"#,
        labels.join(",")
    ));
    let sequence = layout(&question, r#""state""#, limits(512, 192)).expect("the question fits");
    assert_eq!(sequence.option_marker_positions.len(), 60);
    assert_eq!(sequence.distinct_option_count, 1);
}

#[test]
fn should_reject_a_question_without_instructions() {
    let question = single_question(r#"{"q": {"type": "noul"}}"#);
    assert!(matches!(
        layout(&question, r#""state""#, limits(512, 192)),
        Err(LayaSequenceError::Rendering(_))
    ));
}

#[test]
fn should_report_a_tokenizer_failure() {
    let state_result = encode_laya_state(
        &FailingTokenEncoder,
        &special_tokens(),
        &ordered_json(r#""state""#),
    );
    assert!(state_result.is_err());
    let sequence_result = build_laya_question_sequence(
        &FailingTokenEncoder,
        &special_tokens(),
        limits(512, 192),
        &two_option_question(),
        &[],
        false,
    );
    assert!(matches!(
        sequence_result,
        Err(LayaSequenceError::TokenEncoding(_))
    ));
}

/// A question whose two option spans total exactly `2 * (description_words + 2)`
/// tokens: a marker, the key with its colon, and the description words.
fn question_with_description_words(description_words: usize) -> DecisionQuestion {
    let mut words = Vec::new();
    for word_index in 0..description_words {
        words.push(format!("w{word_index}"));
    }
    let description = words.join(" ");
    single_question(&format!(
        r#"{{"q": {{"type": "choice", "instructions": "Pick", "criteria": {{"a": "{description}", "b": "{description}"}}}}}}"#
    ))
}

#[test]
fn should_keep_options_whole_when_exactly_the_required_head_tokens_stay_free() {
    // Two 6-token spans in a 28-token budget leave exactly 16 free.
    let sequence = layout(
        &question_with_description_words(4),
        r#""state""#,
        limits(512, 28),
    )
    .expect("the question fits");
    assert_eq!(sequence.option_token_cap, None);
}

#[test]
fn should_cut_options_when_one_token_fewer_than_required_stays_free() {
    // The same spans in a 27-token budget leave 15 free, one below the rule.
    let sequence = layout(
        &question_with_description_words(4),
        r#""state""#,
        limits(512, 27),
    )
    .expect("the question fits");
    assert_eq!(sequence.option_token_cap, Some(5));
}

#[test]
fn should_keep_eight_instruction_tokens_however_small_the_head_budget() {
    let question = single_question(
        r#"{"q": {"type": "noul", "instructions": "one two three four five six seven eight nine ten"}}"#,
    );
    // "noul question:" adds two words, so the instruction line is 12 tokens.
    let sequence = layout(&question, r#""state""#, limits(512, 1)).expect("the question fits");
    assert_eq!(sequence.option_marker_positions[0], 1 + 8 + 1);
}

#[test]
fn should_lay_out_a_question_about_an_empty_state() {
    let sequence =
        layout(&two_option_question(), r#""""#, limits(512, 192)).expect("the question fits");
    assert_eq!(sequence.state_token_count, 0);
    assert_eq!(sequence.token_ids.len(), 13);
    assert_eq!(sequence.token_ids[12], SEP_TOKEN_ID);
}

#[test]
fn should_keep_the_whole_state_when_it_fits_the_remaining_room_exactly() {
    // The head is 12 tokens and the state 3, so 16 holds everything and the closing separator.
    let sequence = layout(
        &two_option_question(),
        r#""one two three""#,
        limits(16, 192),
    )
    .expect("the question fits");
    assert_eq!(sequence.state_tokens_used, 3);
    assert_eq!(sequence.token_ids.len(), 16);
}
