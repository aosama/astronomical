//! Unit tests for building the training job and reading its outcome.

use astronomical_decision_contract::decision_training_job::{RunEnding, RunSummary, TrainingScope};
use astronomical_decision_contract::{
    DecisionQuestion, QuestionInputCoverage, answer_from_probabilities,
};
use astronomical_decision_training::labelled_example::parse_labelled_example;
use astronomical_decision_training::temperature_softmax::probabilities_at;

use super::*;
use crate::verbs::decision::check_set::built_in_check_examples;

const SERVED_LOGITS: [f32; 2] = [0.0, 1.0];
const TRAINED_LOGITS: [f32; 2] = [-0.5, 2.0];

fn examples_about(state_count: usize) -> Vec<LabelledExample> {
    let mut labelled_examples = Vec::new();
    for state_number in 0..state_count {
        let line_text = format!(
            r#"{{"state": "Romeo line {state_number}", "questions": {{"in_love": {{"type": "noul"}}}}, "labels": {{"in_love": true}}}}"#
        );
        labelled_examples.push(parse_labelled_example(&line_text, state_number + 1).unwrap());
    }
    labelled_examples
}

fn served_outcome(question: &DecisionQuestion, question_name: &str) -> QuestionOutcome {
    QuestionOutcome {
        question_name: question_name.to_owned(),
        answer: answer_from_probabilities(question, &probabilities_at(&SERVED_LOGITS, 1.0))
            .unwrap(),
        option_logits: SERVED_LOGITS.to_vec(),
        applied_temperature: 1.0,
        input_coverage: QuestionInputCoverage {
            state_token_count: 4,
            state_tokens_read: 4,
            option_token_cap: None,
            distinguishable_option_count: 2,
        },
        head_inputs: None,
        training_record: None,
    }
}

/// An outcome answering every question of `job`, each with the same
/// served and trained scores.
fn outcome_for(
    job: &DecisionTrainingJob,
    labelled_examples: &[LabelledExample],
) -> DecisionTrainingOutcome {
    let noul_question = &labelled_examples[0].labelled_questions[0].question;
    let mut examples = Vec::new();
    for training_example in &job.examples {
        let mut trained_questions = Vec::new();
        for _ in &training_example.question_roles {
            trained_questions.push(TrainedQuestion {
                served: served_outcome(noul_question, "in_love"),
                trained_logits: TRAINED_LOGITS.to_vec(),
            });
        }
        examples.push(trained_questions);
    }
    DecisionTrainingOutcome {
        examples,
        trained_tensors: Vec::new(),
        adapter_tensors: Vec::new(),
        runs: vec![RunSummary {
            seed: 1,
            best_choosing_loss: 0.3,
            best_epoch: 4,
            epochs_run: 10,
            ending: RunEnding::StoppedImproving,
        }],
        best_seed: 1,
        largest_parity_gap: 0.0,
        parity_gap_beyond_rounding: None,
        indistinguishable_question_count: 0,
    }
}

fn prepared_for(labelled_examples: &[LabelledExample]) -> PreparedJob {
    training_job(
        labelled_examples,
        &built_in_check_examples().unwrap(),
        TrainingScope::FinalLayers,
    )
    .unwrap()
}

#[test]
fn should_give_every_question_its_states_part_and_mark_the_check_set_as_checks() {
    let labelled_examples = examples_about(40);
    let prepared = prepared_for(&labelled_examples);
    let counts = prepared.part_counts;
    assert_eq!(counts.training + counts.choosing + counts.judging, 40);
    // Of every 20 states, three judge and three choose.
    assert_eq!((counts.judging, counts.choosing), (6, 6));
    for (training_example, part) in prepared.job.examples.iter().zip(&prepared.example_parts) {
        let expected_role = match part {
            SplitPart::Training => QuestionRole::Training { correct_option: 1 },
            SplitPart::Choosing => QuestionRole::Choosing { correct_option: 1 },
            SplitPart::Judging => QuestionRole::Judging { correct_option: 1 },
        };
        assert_eq!(training_example.question_roles, vec![expected_role]);
    }
    let check_examples = &prepared.job.examples[40..];
    assert_eq!(check_examples.len(), 24);
    for check_example in check_examples {
        assert_eq!(check_example.question_roles, vec![QuestionRole::Check; 5]);
    }
}

#[test]
fn should_route_trained_scores_by_part_and_skip_questions_left_out() {
    let labelled_examples = examples_about(40);
    let prepared = prepared_for(&labelled_examples);
    let mut outcome = outcome_for(&prepared.job, &labelled_examples);
    let mut judging_positions = Vec::new();
    for (example_position, part) in prepared.example_parts.iter().enumerate() {
        if *part == SplitPart::Judging {
            judging_positions.push(example_position);
        }
    }
    // The worker leaves out a question whose options read alike.
    outcome.examples[judging_positions[0]][0]
        .trained_logits
        .clear();
    let training_scores = training_scores(
        &labelled_examples,
        &prepared.example_parts,
        prepared.sent_example_count(),
        &outcome,
    )
    .unwrap();
    assert_eq!(training_scores.choosing.len(), 6);
    assert_eq!(training_scores.judging.len(), 5);
    assert_eq!(training_scores.check.len(), 24 * 5);
    assert_eq!(training_scores.judging[0].trained_logits, TRAINED_LOGITS);
    assert_eq!(
        training_scores.judging[0].served.option_logits,
        SERVED_LOGITS
    );
    let served_probabilities = &training_scores.check[0].served_probabilities;
    assert!((served_probabilities[0] + served_probabilities[1] - 1.0).abs() < 1e-9);
}

#[test]
fn should_refuse_an_outcome_that_does_not_answer_the_job() {
    let labelled_examples = examples_about(40);
    let prepared = prepared_for(&labelled_examples);
    let mut outcome = outcome_for(&prepared.job, &labelled_examples);
    let mut missing_one_check_example = outcome_for(&prepared.job, &labelled_examples);
    missing_one_check_example.examples.pop();
    assert!(matches!(
        training_scores(
            &labelled_examples,
            &prepared.example_parts,
            prepared.sent_example_count(),
            &missing_one_check_example,
        ),
        Err(DecisionToolError::TrainingFailed(_))
    ));
    outcome.examples[3][0].trained_logits.push(0.0);
    assert!(matches!(
        training_scores(
            &labelled_examples,
            &prepared.example_parts,
            prepared.sent_example_count(),
            &outcome,
        ),
        Err(DecisionToolError::TrainingFailed(_))
    ));
    outcome.examples.truncate(10);
    assert!(matches!(
        training_scores(
            &labelled_examples,
            &prepared.example_parts,
            prepared.sent_example_count(),
            &outcome,
        ),
        Err(DecisionToolError::TrainingFailed(_))
    ));
}

#[test]
fn should_compare_on_the_training_examples_when_too_few_texts_hold_any_back() {
    let labelled_examples = examples_about(1);
    let prepared = prepared_for(&labelled_examples);
    assert_eq!(prepared.example_parts, vec![SplitPart::Training]);
    let outcome = outcome_for(&prepared.job, &labelled_examples);

    let training_scores = training_scores(
        &labelled_examples,
        &prepared.example_parts,
        prepared.sent_example_count(),
        &outcome,
    )
    .unwrap();

    assert_eq!(training_scores.judged_on, JudgedOn::LearnedFromExamples);
    assert_eq!(training_scores.choosing.len(), 1);
    assert_eq!(training_scores.judging.len(), 1);
}

#[test]
fn should_compare_on_held_back_questions_when_enough_texts_allow_it() {
    let labelled_examples = examples_about(40);
    let prepared = prepared_for(&labelled_examples);
    let outcome = outcome_for(&prepared.job, &labelled_examples);

    let training_scores = training_scores(
        &labelled_examples,
        &prepared.example_parts,
        prepared.sent_example_count(),
        &outcome,
    )
    .unwrap();

    assert_eq!(training_scores.judged_on, JudgedOn::HeldBackQuestions);
}
