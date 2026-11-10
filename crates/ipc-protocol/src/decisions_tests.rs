//! Unit tests for decision commands at the worker boundary.

use astronomical_decision_contract::{OrderedJsonValue, QuestionDefinitionError};

use crate::{
    DecisionsCommand, DecisionsValidationError, MINIMUM_DECISION_TOKENS_PER_QUESTION, RequestId,
    WorkerDecisionCapabilities,
};

fn ordered_json(json_text: &str) -> OrderedJsonValue {
    serde_json::from_str(json_text).expect("test JSON is well formed")
}

fn decisions_command(model: &str, questions_json_text: &str) -> DecisionsCommand {
    DecisionsCommand {
        request_id: RequestId::new(7),
        model: model.to_owned(),
        state: ordered_json(r#"{"speaker": "Hamlet", "act": 3}"#),
        questions: ordered_json(questions_json_text),
    }
}

const ONE_NOUL_QUESTION: &str =
    r#"{"grave": {"type": "noul", "instructions": "Is the mood grave?"}}"#;

#[test]
fn should_return_the_parsed_questions_of_a_valid_command() {
    let questions = decisions_command("laya", ONE_NOUL_QUESTION)
        .validated_questions()
        .expect("a valid command");
    assert_eq!(questions.len(), 1);
    assert_eq!(questions[0].name, "grave");
}

#[test]
fn should_reject_a_blank_model_id() {
    assert_eq!(
        decisions_command("  ", ONE_NOUL_QUESTION).validated_questions(),
        Err(DecisionsValidationError::EmptyModelId)
    );
}

#[test]
fn should_reject_questions_the_contract_refuses() {
    assert_eq!(
        decisions_command("laya", "{}").validated_questions(),
        Err(DecisionsValidationError::Questions(
            QuestionDefinitionError::NoQuestions
        ))
    );
}

#[test]
fn should_keep_state_member_order_across_the_wire() {
    let command = DecisionsCommand {
        request_id: RequestId::new(7),
        model: "laya".to_owned(),
        state: ordered_json(r#"{"zebra": 1, "apple": 2}"#),
        questions: ordered_json(ONE_NOUL_QUESTION),
    };
    let wire_text = serde_json::to_string(&command).expect("a command serializes");
    assert!(
        wire_text.contains(r#""state":{"zebra":1,"apple":2}"#),
        "member order must survive: {wire_text}"
    );
    let restored: DecisionsCommand = serde_json::from_str(&wire_text).expect("it parses back");
    assert_eq!(restored, command);
}

#[test]
fn should_refuse_a_command_with_an_unknown_field() {
    let wire_text =
        r#"{"request_id": 7, "model": "laya", "state": "s", "questions": {}, "extra": 1}"#;
    assert!(serde_json::from_str::<DecisionsCommand>(wire_text).is_err());
}

#[test]
fn should_call_a_limit_usable_from_the_smallest_question_upward() {
    let smallest_usable = WorkerDecisionCapabilities {
        maximum_tokens_per_question: MINIMUM_DECISION_TOKENS_PER_QUESTION,
    };
    let one_below = WorkerDecisionCapabilities {
        maximum_tokens_per_question: MINIMUM_DECISION_TOKENS_PER_QUESTION - 1,
    };
    assert!(smallest_usable.is_usable());
    assert!(!one_below.is_usable());
}
