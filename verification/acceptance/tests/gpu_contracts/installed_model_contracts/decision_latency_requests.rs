//! The questions and the timed request every decision-model latency
//! measurement (Clef, Kev, Laya) sends, so each engine is timed on the same
//! workload and only the state and limits differ between them.

use std::time::Instant;

use decision_contract::{
    DecisionQuestion, HeadInputRequest, OrderedJsonValue, parse_system_one_questions,
};
use model_serving::DecisionEngine;

pub(super) fn questions_of_count(question_count: usize) -> Vec<DecisionQuestion> {
    let subjects = [
        "death", "love", "a sword", "the sea", "a feud", "night", "a prince", "poison", "a letter",
        "a tomb",
    ];
    let mut question_members = Vec::with_capacity(question_count);
    for question_index in 0..question_count {
        question_members.push(format!(
            r#""q{question_index}": {{"type": "noul", "instructions": "Does the passage mention {}?"}}"#,
            subjects[question_index % subjects.len()]
        ));
    }
    let questions: OrderedJsonValue =
        serde_json::from_str(&format!("{{{}}}", question_members.join(",")))
            .expect("the measurement questions are well formed");
    parse_system_one_questions(&questions).expect("the measurement questions are valid")
}

/// One request's wall time in milliseconds and the tokens the engine read.
pub(super) fn timed_request_milliseconds(
    decision_engine: &mut impl DecisionEngine,
    state: &OrderedJsonValue,
    question_count: usize,
) -> (f64, u32) {
    let questions = questions_of_count(question_count);
    let request_started_at = Instant::now();
    let decision_output = decision_engine
        .decide(state, &questions, HeadInputRequest::Omit)
        .expect("the timed request should be answered");
    assert_eq!(decision_output.question_outcomes.len(), question_count);
    (
        request_started_at.elapsed().as_secs_f64() * 1_000.0,
        decision_output.input_token_count,
    )
}
