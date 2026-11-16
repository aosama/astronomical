//! Unit tests for the head-training job's wire form.

use super::*;

#[test]
fn should_carry_each_question_role_with_its_right_answer() {
    let job = HeadTrainingJob {
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
        serde_json::from_str::<HeadTrainingJob>(&wire_text).unwrap(),
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
    assert_eq!(HeadTrainingJob::default().scope, TrainingScope::FinalLayers);
}

#[test]
fn should_refuse_a_scope_it_does_not_know() {
    assert!(serde_json::from_str::<TrainingScope>(r#""half_the_model""#).is_err());
}
