//! Unit contracts for choosing the model that answers a decisions request
//! and for refusing malformed questions before a worker command exists.

use std::path::PathBuf;

use astronomical_config::{
    ChatModelCapabilities, DecisionModelCapabilities, DiscoveredModel, ModelCapabilities,
    ModelFamily,
};
use astronomical_decision_contract::OrderedJsonValue;

use super::{DecisionsModelResolutionError, build_decisions_command, resolve_decisions_model};

fn decision_model(model_id: &str) -> DiscoveredModel {
    DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: None,
        model_family: ModelFamily::Laya,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/laya"),
        capabilities: ModelCapabilities::Decisions(DecisionModelCapabilities {
            maximum_tokens_per_question: 512,
        }),
        license: None,
        model_size_bytes: 1,
    }
}

fn chat_model(model_id: &str) -> DiscoveredModel {
    DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: None,
        model_family: ModelFamily::Qwen3_5,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/chat"),
        capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
            context_window: 2_048,
            max_input_tokens: 1_024,
            max_output_tokens: 128,
            supports_vision: false,
            supports_reasoning: false,
            supports_tool_calls: false,
        }),
        license: None,
        model_size_bytes: 1,
    }
}

fn questions_from(questions_json: &str) -> OrderedJsonValue {
    serde_json::from_str(questions_json).expect("the test questions should be JSON")
}

#[test]
fn should_choose_the_named_decision_model() {
    let discovered_models = [decision_model("decider-a"), decision_model("decider-b")];

    let resolved_model_id = resolve_decisions_model(&discovered_models, None, Some("decider-b"));

    assert_eq!(resolved_model_id, Ok("decider-b".to_owned()));
}

#[test]
fn should_refuse_a_named_model_that_is_not_installed() {
    let discovered_models = [decision_model("decider-a")];

    let resolved_model_id = resolve_decisions_model(&discovered_models, None, Some("missing"));

    assert_eq!(
        resolved_model_id,
        Err(DecisionsModelResolutionError::ModelNotFound)
    );
}

#[test]
fn should_refuse_a_named_model_that_cannot_decide() {
    let discovered_models = [chat_model("chatter"), decision_model("decider-a")];

    let resolved_model_id = resolve_decisions_model(&discovered_models, None, Some("chatter"));

    assert_eq!(
        resolved_model_id,
        Err(DecisionsModelResolutionError::ModelCannotDecide)
    );
}

#[test]
fn should_choose_the_only_installed_decision_model_when_none_is_named() {
    let discovered_models = [chat_model("chatter"), decision_model("decider-a")];

    let resolved_model_id = resolve_decisions_model(&discovered_models, Some("chatter"), None);

    assert_eq!(resolved_model_id, Ok("decider-a".to_owned()));
}

#[test]
fn should_keep_the_resident_decision_model_when_several_are_installed() {
    let discovered_models = [decision_model("decider-a"), decision_model("decider-b")];

    let resolved_model_id = resolve_decisions_model(&discovered_models, Some("decider-b"), None);

    assert_eq!(resolved_model_id, Ok("decider-b".to_owned()));
}

#[test]
fn should_refuse_to_guess_between_several_decision_models() {
    let discovered_models = [decision_model("decider-b"), decision_model("decider-a")];

    let resolved_model_id = resolve_decisions_model(&discovered_models, Some("chatter"), None);

    assert_eq!(
        resolved_model_id,
        Err(
            DecisionsModelResolutionError::SeveralDecisionModelsInstalled {
                installed_model_ids: vec!["decider-a".to_owned(), "decider-b".to_owned()],
            }
        ),
        "the installed ids are listed in a stable order"
    );
}

#[test]
fn should_prefer_the_default_decision_model_when_several_are_installed() {
    let default_model_id = astronomical_config::BUILTIN_DEFAULT_DECISION_MODEL_ID;
    let bundled_model_id = format!("{default_model_id}-multilingual");
    let discovered_models = [
        decision_model(&bundled_model_id),
        decision_model(default_model_id),
    ];

    let resolved_model_id = resolve_decisions_model(&discovered_models, Some("chatter"), None);

    assert_eq!(resolved_model_id, Ok(default_model_id.to_owned()));
}

#[test]
fn should_keep_a_resident_decision_model_over_the_default_one() {
    let default_model_id = astronomical_config::BUILTIN_DEFAULT_DECISION_MODEL_ID;
    let bundled_model_id = format!("{default_model_id}-multilingual");
    let discovered_models = [
        decision_model(default_model_id),
        decision_model(&bundled_model_id),
    ];

    let resolved_model_id =
        resolve_decisions_model(&discovered_models, Some(&bundled_model_id), None);

    assert_eq!(resolved_model_id, Ok(bundled_model_id));
}

#[test]
fn should_report_that_no_decision_model_is_installed() {
    let resolved_model_id = resolve_decisions_model(&[chat_model("chatter")], None, None);

    assert_eq!(
        resolved_model_id,
        Err(DecisionsModelResolutionError::NoDecisionModelInstalled)
    );
}

#[test]
fn should_report_no_decision_model_when_nothing_is_installed_at_all() {
    let resolved_model_id = resolve_decisions_model(&[], None, None);

    assert_eq!(
        resolved_model_id,
        Err(DecisionsModelResolutionError::NoDecisionModelInstalled)
    );
}

#[test]
fn should_build_a_command_that_carries_the_request_as_written() {
    let questions = questions_from(
        r#"{"zeta": {"type": "noul", "instructions": "Z?"}, "alpha": {"type": "noul", "instructions": "A?"}}"#,
    );
    let state = OrderedJsonValue::Text("Now is the winter of our discontent.".to_owned());

    let decisions_command =
        build_decisions_command(7, "decider-a", state.clone(), questions.clone())
            .expect("well-formed questions should build a command");

    assert_eq!(
        decisions_command.request_id,
        astronomical_ipc_protocol::RequestId::new(7)
    );
    assert_eq!(decisions_command.model, "decider-a");
    assert_eq!(decisions_command.state, state);
    assert_eq!(
        decisions_command.questions, questions,
        "the questions keep the order the caller wrote them in"
    );
}

#[test]
fn should_refuse_to_build_a_command_from_questions_that_are_not_an_object() {
    let command_outcome = build_decisions_command(
        1,
        "decider-a",
        OrderedJsonValue::Text("state".to_owned()),
        questions_from(r#"["urgent"]"#),
    );

    assert!(command_outcome.is_err());
}

#[test]
fn should_refuse_to_build_a_command_from_an_empty_questions_object() {
    let command_outcome = build_decisions_command(
        1,
        "decider-a",
        OrderedJsonValue::Text("state".to_owned()),
        questions_from("{}"),
    );

    assert!(command_outcome.is_err());
}
