use decision_contract::training_record::{
    DECISION_TRAINING_RECORD_FILE_NAME, RecordedVerdict, training_record_document,
};
use decision_contract::{DecisionAnswer, NoulAnswer, QuestionInputCoverage, QuestionOutcome};

use super::*;

const ROMEO_LINE: &str = "But soft, what light through yonder window breaks?";

/// Answers every request with one yes-or-no outcome.
struct OneAnswerEngine;

impl memory_policy::live_memory_limit::LiveMemoryLimitEngine for OneAnswerEngine {}

impl DecisionEngine for OneAnswerEngine {
    fn load(&mut self) -> Result<DecisionEngineLoadResult, DecisionFailure> {
        Ok(DecisionEngineLoadResult {
            model_id: "romeo-trained".to_owned(),
            maximum_tokens_per_question: 512,
        })
    }

    fn decide(
        &mut self,
        _state: &OrderedJsonValue,
        _questions: &[DecisionQuestion],
        _head_input_request: HeadInputRequest,
    ) -> Result<DecisionEngineOutput, DecisionFailure> {
        Ok(DecisionEngineOutput {
            question_outcomes: vec![QuestionOutcome {
                question_name: "is_dawn".to_owned(),
                answer: DecisionAnswer::Noul(NoulAnswer {
                    probability_true: 0.8,
                    answer_confidence: 0.8,
                }),
                option_logits: vec![1.0, 0.0],
                applied_temperature: 1.0,
                input_coverage: QuestionInputCoverage {
                    state_token_count: 12,
                    state_tokens_read: 12,
                    option_token_cap: None,
                    distinguishable_option_count: 2,
                },
                head_inputs: None,
                training_record: None,
            }],
            input_token_count: 12,
            elapsed_millis: 1,
        })
    }
}

fn judged_not_better_record() -> TrainingRecord {
    TrainingRecord {
        verdict: RecordedVerdict::JudgedNotBetter,
        labelled_question_count: 240,
        distinct_state_count: 48,
        warnings: vec!["right 61.0% of the time, down from 64.0%".to_owned()],
    }
}

fn answers_of(engine: &mut dyn DecisionEngine) -> DecisionEngineOutput {
    engine
        .decide(
            &OrderedJsonValue::Text(ROMEO_LINE.to_owned()),
            &[],
            HeadInputRequest::Omit,
        )
        .unwrap()
}

#[test]
fn should_carry_the_folders_record_into_every_answer() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory
            .path()
            .join(DECISION_TRAINING_RECORD_FILE_NAME),
        training_record_document(&judged_not_better_record()),
    )
    .unwrap();

    let mut engine =
        RecordedTrainingEngine::wrapping(Box::new(OneAnswerEngine), model_directory.path())
            .unwrap();

    let engine_output = answers_of(engine.as_mut());
    assert_eq!(
        engine_output.question_outcomes[0].training_record,
        Some(judged_not_better_record())
    );
}

#[test]
fn should_answer_as_before_from_a_folder_without_a_record() {
    let model_directory = tempfile::tempdir().unwrap();

    let mut engine =
        RecordedTrainingEngine::wrapping(Box::new(OneAnswerEngine), model_directory.path())
            .unwrap();

    assert_eq!(
        answers_of(engine.as_mut()).question_outcomes[0].training_record,
        None
    );
}

#[test]
fn should_refuse_to_load_a_folder_whose_record_cannot_be_read() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory
            .path()
            .join(DECISION_TRAINING_RECORD_FILE_NAME),
        b"not a record",
    )
    .unwrap();

    let wrapped =
        RecordedTrainingEngine::wrapping(Box::new(OneAnswerEngine), model_directory.path());

    assert!(wrapped.is_err());
}
