//! The `astronomical decision calibrate` journey.
//!
//! Everything that can be checked without the model is checked first: the
//! examples file, the model's folder, and the output folder. Only then is
//! every example sent to the daemon, one request per example, keeping each
//! question's raw scores. The fit, the never-worse gate and the proven copy
//! all run on the CPU in this process; the daemon's only part is scoring.

use std::io::Write;

use astronomical_decision_contract::QuestionOutcome;
use astronomical_decision_contract::calibration_file::CalibrationOverride;
use astronomical_decision_training::calibrated_model_folder::FolderAdditions;
use astronomical_decision_training::calibration_plan::plan_calibration;
use astronomical_decision_training::holdout_split::{split_by_state, state_group_key};
use astronomical_decision_training::labelled_example::LabelledExample;
use astronomical_decision_training::scored_question::{
    ScoredQuestion, scored_question_from_outcome,
};

use super::calibration_report::{CalibrationCounts, calibration_report_text};
use super::decision_arguments::DecisionToolArguments;
use super::decision_tool::{
    DecisionToolDependencies, DecisionToolError, announce_model_folder, check_output_is_new,
    model_lifecycle, prepare_decision_model, read_labelled_examples, write_model_folder,
    write_report,
};
use super::example_scoring::{ExampleRequest, ScoringPass, answer_examples};
use super::model_folder_lookup::find_model_folder;

/// Runs the whole calibration against the resident daemon.
pub async fn run_calibrate(
    calibrate_arguments: &DecisionToolArguments,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<(), DecisionToolError> {
    let labelled_examples = read_labelled_examples(&calibrate_arguments.examples_file_path)?;
    let model_lifecycle = model_lifecycle(dependencies);
    let decision_model_id = prepare_decision_model(
        calibrate_arguments.model_id.as_deref(),
        &model_lifecycle,
        dependencies,
    )
    .await?;
    let model_folder = find_model_folder(&dependencies.instance_paths, &decision_model_id)?;
    let output_directory = &calibrate_arguments.output_directory;
    check_output_is_new(&model_folder, output_directory)?;
    let answered_examples = answer_examples(
        &example_requests(&labelled_examples),
        &ScoringPass {
            decision_model_id: &decision_model_id,
            stage_name: "Scoring examples",
        },
        &model_lifecycle.daemon_probe,
        dependencies,
    )
    .await?;
    let mut scored_questions = Vec::new();
    for (labelled_example, question_outcomes) in labelled_examples.iter().zip(&answered_examples) {
        scored_questions.extend(scored_example(labelled_example, question_outcomes)?);
    }
    let calibration_override = fit_and_report(
        &decision_model_id,
        labelled_examples.len(),
        scored_questions,
        dependencies.stdout,
    )?;
    let folder_additions = FolderAdditions {
        calibration_override: Some(&calibration_override),
        head_tuning: None,
    };
    write_model_folder(
        output_directory,
        &model_folder,
        folder_additions,
        dependencies,
    )?;
    announce_model_folder(
        output_directory,
        &model_folder,
        &model_lifecycle.daemon_probe,
        dependencies,
    )
    .await
}

/// Each labelled example as a request for the daemon.
pub(super) fn example_requests(labelled_examples: &[LabelledExample]) -> Vec<ExampleRequest<'_>> {
    let mut example_requests = Vec::with_capacity(labelled_examples.len());
    for labelled_example in labelled_examples {
        example_requests.push(ExampleRequest {
            state: &labelled_example.state,
            questions_document: &labelled_example.questions_document,
        });
    }
    example_requests
}

/// Splits, fits and gates, prints the report, and returns the calibration
/// to write; refuses when no kind improved, so nothing is written.
fn fit_and_report(
    decision_model_id: &str,
    example_count: usize,
    scored_questions: Vec<ScoredQuestion>,
    stdout: &mut dyn Write,
) -> Result<CalibrationOverride, DecisionToolError> {
    let question_count = scored_questions.len();
    let holdout_split = split_by_state(scored_questions);
    let calibration_plan = plan_calibration(&holdout_split)?;
    let calibration_counts = CalibrationCounts {
        example_count,
        question_count,
        holdout_question_count: holdout_split.holdout_questions.len(),
    };
    let report_text =
        calibration_report_text(decision_model_id, calibration_counts, &calibration_plan);
    write_report(stdout, &report_text)?;
    if calibration_plan.calibration_override.decides_no_kind() {
        return Err(DecisionToolError::NothingImproved {
            model_id: decision_model_id.to_owned(),
        });
    }
    Ok(calibration_plan.calibration_override)
}

/// One example's questions with the scores the model gave them, each
/// checked: the outcomes must answer the questions asked, in order, and the
/// served probabilities must follow from the raw scores.
pub(super) fn scored_example(
    labelled_example: &LabelledExample,
    question_outcomes: &[QuestionOutcome],
) -> Result<Vec<ScoredQuestion>, DecisionToolError> {
    let line_number = labelled_example.line_number;
    if question_outcomes.len() != labelled_example.labelled_questions.len() {
        return Err(DecisionToolError::AnswerCountMismatch {
            line_number,
            asked: labelled_example.labelled_questions.len(),
            answered: question_outcomes.len(),
        });
    }
    let state_key = state_group_key(&labelled_example.state);
    let mut scored_questions = Vec::with_capacity(question_outcomes.len());
    for (labelled_question, question_outcome) in labelled_example
        .labelled_questions
        .iter()
        .zip(question_outcomes)
    {
        match scored_question_from_outcome(labelled_question, question_outcome, state_key) {
            Ok(scored_question) => scored_questions.push(scored_question),
            Err(source) => {
                return Err(DecisionToolError::Scoring {
                    line_number,
                    source,
                });
            }
        }
    }
    Ok(scored_questions)
}

#[cfg(test)]
#[path = "calibrate_tests.rs"]
mod tests;
