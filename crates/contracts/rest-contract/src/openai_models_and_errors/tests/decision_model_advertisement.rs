//! Decision-model advertisement contracts: what `/v1/models` says about a
//! model that answers typed questions and generates no text.

use crate::{OpenAiDecisionModelParts, OpenAiModel};

fn decision_model_parts(maximum_tokens_per_question: u32) -> OpenAiDecisionModelParts {
    OpenAiDecisionModelParts {
        model_id: "laya-MLX-8bit".to_owned(),
        created: 1_784_231_803,
        owned_by: "astronomical".to_owned(),
        maximum_tokens_per_question,
    }
}

#[test]
fn should_advertise_a_decision_model_with_only_the_decisions_endpoint() {
    let model = OpenAiModel::from_decision_parts(decision_model_parts(512))
        .expect("a valid decision model should be advertised");

    let serialized = serde_json::to_string(&model).expect("the decision model should serialize");
    assert!(serialized.contains(r#""supported_endpoints":["/v1/systemone"]"#));
    assert!(serialized.contains(r#""output_modalities":["decision"]"#));
    assert!(serialized.contains(r#""max_input_tokens":512"#));
    assert!(serialized.contains(r#""supports_streaming":false"#));
    assert!(
        !serialized.contains("context_window") && !serialized.contains("max_output_tokens"),
        "a decision model has no context window and generates nothing: {serialized}"
    );
}

#[test]
fn should_advertise_the_smallest_positive_token_limit() {
    let model = OpenAiModel::from_decision_parts(decision_model_parts(1))
        .expect("a one-token limit is positive and must be advertised");

    let serialized = serde_json::to_string(&model).expect("the decision model should serialize");
    assert!(serialized.contains(r#""max_input_tokens":1"#));
}

#[test]
fn should_reject_a_zero_token_limit_decision_model() {
    let validation_error = OpenAiModel::from_decision_parts(decision_model_parts(0))
        .expect_err("a zero token limit must fail advertisement");

    assert!(validation_error.to_string().contains("token limit"));
}
