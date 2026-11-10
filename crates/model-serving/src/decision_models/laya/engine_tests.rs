//! Unit tests for the engine's pure decisions: whose fault a failure is, and
//! how logits become an outcome. Running a real checkpoint is the acceptance
//! journeys' job.

use astronomical_decision_contract::laya_rendering::LayaRenderingError;
use astronomical_decision_contract::laya_sequence::{LayaQuestionSequence, LayaSequenceError};
use astronomical_decision_contract::{
    DecisionAnswer, DecisionFailure, DecisionQuestion, OrderedJsonValue, TokenEncodingError,
    parse_system_one_questions,
};

use super::{question_outcome, sequence_failure};
use crate::decision_models::laya::configuration::LayaConfiguration;

fn test_configuration() -> LayaConfiguration {
    LayaConfiguration::from_config_value(&serde_json::json!({
        "encoder": {
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
            "max_position_embeddings": 512, "local_attention": 8,
            "global_attn_every_n_layers": 3,
            "rope_parameters": {
                "full_attention": { "rope_theta": 160_000.0 },
                "sliding_attention": { "rope_theta": 10_000.0 },
            },
            "layer_norm_eps": 1e-5, "pad_token_id": 0, "cls_token_id": 1, "sep_token_id": 2,
        },
        "temperature": [2.0, 1.0, 1.0],
        "quantization": { "group_size": 32, "bits": 8 },
    }))
    .expect("the test settings are supported")
}

fn two_option_question() -> DecisionQuestion {
    let questions: OrderedJsonValue = serde_json::from_str(
        r#"{"mood": {"type": "choice", "instructions": "Pick.", "criteria": ["grave", "merry"]}}"#,
    )
    .expect("the test question is well formed");
    parse_system_one_questions(&questions)
        .expect("the test question is valid")
        .remove(0)
}

fn sequence_with_a_cut_state() -> LayaQuestionSequence {
    LayaQuestionSequence {
        token_ids: vec![1, 2, 3],
        option_marker_positions: vec![1, 2],
        option_token_cap: Some(4),
        distinct_option_count: 2,
        state_token_count: 900,
        state_tokens_used: 300,
    }
}

#[test]
fn should_blame_the_request_for_a_question_that_cannot_be_laid_out() {
    let missing_instructions =
        LayaSequenceError::Rendering(LayaRenderingError::MissingInstructions {
            question: "mood".to_owned(),
        });
    assert!(matches!(
        sequence_failure(missing_instructions),
        DecisionFailure::InvalidRequest { .. }
    ));
    let options_do_not_fit = LayaSequenceError::OptionsDoNotFit {
        question: "mood".to_owned(),
        surviving_marker_count: 1,
        option_count: 2,
        maximum_sequence_tokens: 9,
        head_token_budget: 192,
    };
    assert!(matches!(
        sequence_failure(options_do_not_fit),
        DecisionFailure::InvalidRequest { .. }
    ));
}

#[test]
fn should_blame_the_model_for_a_tokenizer_that_cannot_encode() {
    let token_encoding = LayaSequenceError::TokenEncoding(TokenEncodingError {
        reason: "the vocabulary is unreadable".to_owned(),
    });
    assert!(matches!(
        sequence_failure(token_encoding),
        DecisionFailure::FatalExecution { .. }
    ));
}

#[test]
fn should_calibrate_logits_with_the_kinds_temperature_and_report_coverage() {
    let outcome = question_outcome(
        &test_configuration(),
        &two_option_question(),
        &sequence_with_a_cut_state(),
        vec![2.0, 0.0],
    )
    .expect("finite logits give an outcome");
    assert_eq!(outcome.question_name, "mood");
    assert_eq!(outcome.applied_temperature, 2.0);
    assert_eq!(outcome.option_logits, [2.0, 0.0]);
    let DecisionAnswer::Choice(choice_answer) = &outcome.answer else {
        panic!("a choice answer");
    };
    assert_eq!(choice_answer.chosen_key, "grave");
    // softmax([2, 0] / 2) puts e / (e + 1) on the first option.
    let expected_probability = std::f64::consts::E / (std::f64::consts::E + 1.0);
    assert!(
        (choice_answer.option_probabilities[0].probability - expected_probability).abs() < 1e-12
    );
    assert_eq!(outcome.input_coverage.state_token_count, 900);
    assert_eq!(outcome.input_coverage.state_tokens_read, 300);
    assert_eq!(outcome.input_coverage.option_token_cap, Some(4));
    assert_eq!(outcome.input_coverage.distinguishable_option_count, 2);
}

#[test]
fn should_refuse_a_non_finite_logit() {
    for non_finite_logit in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            question_outcome(
                &test_configuration(),
                &two_option_question(),
                &sequence_with_a_cut_state(),
                vec![0.5, non_finite_logit],
            ),
            Err(DecisionFailure::FatalExecution { .. })
        ));
    }
}

#[test]
fn should_refuse_a_logit_count_that_differs_from_the_option_count() {
    assert!(matches!(
        question_outcome(
            &test_configuration(),
            &two_option_question(),
            &sequence_with_a_cut_state(),
            vec![0.5],
        ),
        Err(DecisionFailure::FatalExecution { .. })
    ));
}
