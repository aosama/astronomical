//! Unit tests for the host-side layout the joint head reads.

use super::{SpanLayout, averaging_matrix};
use crate::decision_models::clef::prompt::{ClefQuestionLayout, ClefSequence, TokenSpan};
use astronomical_decision_contract::clef_rendering::clef_question_text;
use astronomical_decision_contract::{OrderedJsonValue, parse_system_one_questions};

#[test]
fn should_average_each_span_and_leave_other_columns_zero() {
    let averaging_values = averaging_matrix(
        &[
            TokenSpan { start: 1, end: 3 },
            TokenSpan { start: 0, end: 4 },
        ],
        4,
    );

    assert_eq!(
        averaging_values,
        [0.0, 0.5, 0.5, 0.0, 0.25, 0.25, 0.25, 0.25]
    );
}

#[test]
fn should_leave_an_empty_span_as_a_zero_row() {
    assert_eq!(
        averaging_matrix(&[TokenSpan { start: 2, end: 2 }], 3),
        [0.0, 0.0, 0.0]
    );
}

fn two_question_sequence() -> ClefSequence {
    let questions: OrderedJsonValue = serde_json::from_str(
        r#"{"a": {"type": "noul"}, "b": {"type": "score", "criteria": ["low", "mid", "high"]}}"#,
    )
    .expect("test questions are JSON");
    let questions = parse_system_one_questions(&questions).expect("test questions are valid");
    ClefSequence {
        token_ids: (0..20).collect(),
        questions: vec![
            ClefQuestionLayout {
                question_text: clef_question_text(&questions[0]),
                instructions_span: TokenSpan { start: 2, end: 4 },
                option_spans: vec![
                    TokenSpan { start: 5, end: 7 },
                    TokenSpan { start: 8, end: 9 },
                ],
            },
            ClefQuestionLayout {
                question_text: clef_question_text(&questions[1]),
                instructions_span: TokenSpan { start: 10, end: 11 },
                option_spans: vec![
                    TokenSpan { start: 12, end: 13 },
                    TokenSpan { start: 14, end: 16 },
                    TokenSpan { start: 17, end: 18 },
                ],
            },
        ],
        state_token_count: 2,
        state_tokens_read: 2,
    }
}

#[test]
fn should_flatten_options_with_their_question_type_and_own_question_only_softmax() {
    let span_layout = SpanLayout::of(&two_question_sequence());

    assert_eq!(span_layout.question_of_option, [0, 0, 1, 1, 1]);
    assert_eq!(
        span_layout.type_ids,
        [0, 2],
        "noul is type 0 and score type 2"
    );
    assert_eq!(span_layout.option_counts, [2, 3]);
    assert_eq!(span_layout.option_token_count, 2 + 1 + 1 + 2 + 1);
    assert_eq!(
        span_layout.option_token_spans[1],
        TokenSpan { start: 2, end: 3 }
    );
    let other_question_bias = span_layout.membership_bias[2];
    assert!(other_question_bias < -1.0e29);
    assert_eq!(&span_layout.membership_bias[..2], [0.0, 0.0]);
    assert_eq!(&span_layout.membership_bias[7..], [0.0, 0.0, 0.0]);
}

#[test]
fn should_list_option_tokens_option_by_option() {
    assert_eq!(
        SpanLayout::option_token_ids(&two_question_sequence()),
        [5, 6, 8, 12, 14, 15, 17]
    );
}
