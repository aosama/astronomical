//! Unit tests for the training job's wire form.

use super::*;

#[test]
fn should_carry_each_question_role_with_its_right_answer() {
    let job = DecisionTrainingJob {
        examples: vec![TrainingExample {
            state: OrderedJsonValue::Text("O happy dagger!".to_owned()),
            questions: serde_json::from_str(r#"{"mood": {"type": "noul"}}"#).unwrap(),
            question_roles: vec![QuestionRole::Judging { correct_option: 1 }],
        }],
        scope: TrainingScope::WholeModel,
    };
    let wire_text = serde_json::to_string(&job).unwrap();
    assert!(
        wire_text.contains(r#"{"role":"judging","correct_option":1}"#),
        "{wire_text}"
    );
    assert!(
        wire_text.contains(r#""scope":"whole_model""#),
        "{wire_text}"
    );
    assert_eq!(
        serde_json::from_str::<DecisionTrainingJob>(&wire_text).unwrap(),
        job
    );
}

#[test]
fn should_name_the_right_option_only_for_labelled_roles() {
    assert_eq!(
        QuestionRole::Training { correct_option: 2 }.correct_option(),
        Some(2)
    );
    assert_eq!(QuestionRole::Check.correct_option(), None);
}

#[test]
fn should_refuse_a_role_it_does_not_know() {
    assert!(serde_json::from_str::<QuestionRole>(r#"{"role":"guessing"}"#).is_err());
    // A labelled role without its right answer is refused.
    assert!(serde_json::from_str::<QuestionRole>(r#"{"role":"training"}"#).is_err());
}

#[test]
fn should_round_trip_a_run_that_stopped_at_a_non_finite_step() {
    let summary = RunSummary {
        seed: 3,
        best_choosing_loss: 0.25,
        best_epoch: 0,
        epochs_run: 1,
        ending: RunEnding::NonFiniteStep { epoch: 1 },
    };
    let wire_text = serde_json::to_string(&summary).unwrap();
    assert_eq!(
        serde_json::from_str::<RunSummary>(&wire_text).unwrap(),
        summary
    );
}

#[test]
fn should_train_only_the_final_layers_unless_asked_for_the_whole_model() {
    assert_eq!(
        DecisionTrainingJob::default().scope,
        TrainingScope::FinalLayers
    );
}

#[test]
fn should_refuse_a_scope_it_does_not_know() {
    assert!(serde_json::from_str::<TrainingScope>(r#""half_the_model""#).is_err());
}

#[test]
fn should_read_an_outcome_written_before_parity_gaps_were_reported() {
    let outcome_without_gap_field = r#"{"examples":[],"trained_tensors":[],"adapter_tensors":[],"runs":[],"best_seed":1,"largest_parity_gap":0.0,"indistinguishable_question_count":0}"#;

    let outcome: DecisionTrainingOutcome = serde_json::from_str(outcome_without_gap_field).unwrap();

    assert_eq!(outcome.parity_gap_beyond_rounding, None);
}

#[test]
fn should_carry_the_hand_over_file_name_and_leave_it_out_when_there_is_none() {
    let outcome_with_file = r#"{"examples":[],"trained_tensors":[],"adapter_tensors":[],"runs":[],"best_seed":1,"largest_parity_gap":0.0,"trained_tensors_file":"astronomical-trained-tensors-1-2.safetensors","indistinguishable_question_count":0}"#;

    let outcome: DecisionTrainingOutcome = serde_json::from_str(outcome_with_file).unwrap();
    assert_eq!(
        outcome.trained_tensors_file.as_deref(),
        Some("astronomical-trained-tensors-1-2.safetensors")
    );

    let without_file = DecisionTrainingOutcome {
        trained_tensors_file: None,
        ..outcome
    };
    assert!(
        !serde_json::to_string(&without_file)
            .unwrap()
            .contains("trained_tensors_file")
    );
}
