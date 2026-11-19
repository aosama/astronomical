//! The `astronomical decision train` journey.
//!
//! 1. Check what needs no model: the examples file, the model's folder and
//!    family, and the output folder. Any number of examples trains.
//! 2. Send one job to the daemon: the worker reads every example and the
//!    built-in check set, proves its trainable copy (the final layers, or
//!    with `--whole-model` the whole model with add-on weights in every
//!    layer) reproduces the model's scores, and trains on the GPU in several
//!    runs. It returns every answer beside the trained model's scores.
//! 3. Fit fresh temperatures and compare the result with the shipped model
//!    recalibrated the same way, on questions training never saw, or, with
//!    too few different texts to hold any back, on the training examples.
//! 4. Whatever the comparison shows: write the new folder (originals copied
//!    byte for byte, trained layers and any add-ons in 32-bit files beside
//!    them, and the training record with the verdict and every warning),
//!    reload, and check the written folder answers as the trained model did.
//!    The person decides what to do with it; nothing here refuses or removes
//!    a result.

use std::path::Path;

use config::ModelFamily;
use decision_contract::calibration_file::{
    CalibrationOverride, DECISION_CALIBRATION_FILE_NAME, parse_calibration_document,
};
use decision_contract::decision_adapters_file::DecisionAdapters;
use decision_contract::decision_training_job::{DecisionTrainingOutcome, TrainingScope};
use decision_contract::head_tuning_file::{HeadTuning, TunableHeadFamily};
use decision_contract::trained_tensors::{UnpairedAddOns, decision_adapters_of, head_tuning_of};
use decision_contract::training_record::TrainingRecord;
use decision_training::calibrated_model_folder::FolderAdditions;
use decision_training::labelled_example::{LabelledExample, labelled_data_counts};
use decision_training::training_split::SplitPart;
use decision_training::training_verdict::{JudgedOn, ScoredPair, TrainingVerdict, judge_training};

use super::check_set::built_in_check_examples;
use super::decision_arguments::DecisionToolArguments;
use super::decision_tool::{
    DecisionToolDependencies, DecisionToolError, check_output_is_new, earlier_training_record,
    model_lifecycle, output_model_id, prepare_decision_model, read_labelled_examples,
    write_model_folder, write_report,
};
use super::model_folder_lookup::{ModelFolderLookup, find_model_folder};
use super::new_model_visibility::{
    NewModelVisibility, make_new_model_visible, written_folder_text,
};
use super::saved_model_check::{TrainedExpectation, check_saved_model};
use super::training_job::{PartCounts, training_job, training_scores};
use super::training_report::{TrainingSummary, training_report_text, training_warnings};
use super::training_request::request_training;
use crate::DaemonProbe;

/// What the worker's training produced, with what judging it needs.
struct FinishedTraining<'finished> {
    decision_model_id: &'finished str,
    scope: TrainingScope,
    labelled_examples: &'finished [LabelledExample],
    example_parts: &'finished [SplitPart],
    sent_example_count: usize,
    part_counts: PartCounts,
    outcome: DecisionTrainingOutcome,
    /// The source folder's own record, when it was made by a calibration or
    /// training; the new record builds on it.
    earlier_record: Option<TrainingRecord>,
    /// The source folder's own calibration, which training's refits replace
    /// kind by kind.
    earlier_calibration: CalibrationOverride,
}

/// A trained head, what it was judged by, and the record that travels with it.
struct JudgedTraining {
    /// The calibration the folder will carry: the earlier one with
    /// training's refits; `None` when no kind has one.
    calibration_override: Option<CalibrationOverride>,
    head_tuning: HeadTuning,
    /// The add-ons of a whole-model training; `None` for the final layers.
    decision_adapters: Option<DecisionAdapters>,
    verdict: TrainingVerdict,
    judging_pairs: Vec<ScoredPair>,
    training_record: TrainingRecord,
}

/// Runs the whole training against the resident daemon.
pub async fn run_train(
    train_arguments: &DecisionToolArguments,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<(), DecisionToolError> {
    let labelled_examples = read_labelled_examples(&train_arguments.examples_file_path)?;
    let model_lifecycle = model_lifecycle(dependencies);
    let decision_model_id = prepare_decision_model(
        train_arguments.model_id.as_deref(),
        &model_lifecycle,
        dependencies,
    )
    .await?;
    let model_folder = find_model_folder(&dependencies.instance_paths, &decision_model_id)?;
    let family = tunable_family(model_folder.model_family, &decision_model_id)?;
    let output_directory = &train_arguments.output_directory;
    check_output_is_new(&model_folder, output_directory)?;
    let earlier_record = earlier_training_record(&model_folder.model_directory)?;
    let earlier_calibration = earlier_calibration(&model_folder.model_directory)?;
    let check_examples = match built_in_check_examples() {
        Ok(check_examples) => check_examples,
        Err(check_set_error) => return Err(DecisionToolError::CheckSetUnreadable(check_set_error)),
    };
    let prepared_job = training_job(
        &labelled_examples,
        &check_examples,
        train_arguments.training_scope,
    )?;
    let sent_example_count = prepared_job.sent_example_count();
    let daemon_probe = &model_lifecycle.daemon_probe;
    let outcome = request_training(
        &decision_model_id,
        prepared_job.job,
        daemon_probe,
        dependencies,
    )
    .await?;
    let judged_training = judge_and_report(
        FinishedTraining {
            decision_model_id: &decision_model_id,
            scope: train_arguments.training_scope,
            labelled_examples: &labelled_examples,
            example_parts: &prepared_job.example_parts,
            sent_example_count,
            part_counts: prepared_job.part_counts,
            outcome,
            earlier_record,
            earlier_calibration,
        },
        dependencies,
    )?;
    write_and_prove(
        &WrittenTraining {
            output_directory,
            model_folder: &model_folder,
            family,
            judged_training: &judged_training,
            labelled_examples: &labelled_examples,
        },
        daemon_probe,
        dependencies,
    )
    .await
}

fn tunable_family(
    model_family: ModelFamily,
    model_id: &str,
) -> Result<TunableHeadFamily, DecisionToolError> {
    match model_family {
        ModelFamily::Kev => Ok(TunableHeadFamily::Kev),
        ModelFamily::Laya => Ok(TunableHeadFamily::Laya),
        ModelFamily::Clef => Ok(TunableHeadFamily::Clef),
        _ => Err(DecisionToolError::NotTrainable {
            model_id: model_id.to_owned(),
        }),
    }
}

/// Judges the trained head, prints the report, and returns the head with
/// its training record, whatever the judgement found.
fn judge_and_report(
    finished_training: FinishedTraining<'_>,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<JudgedTraining, DecisionToolError> {
    let outcome = &finished_training.outcome;
    let training_scores = training_scores(
        finished_training.labelled_examples,
        finished_training.example_parts,
        finished_training.sent_example_count,
        outcome,
    )?;
    let verdict = judge_training(&training_scores)?;
    let data_counts = labelled_data_counts(finished_training.labelled_examples);
    let training_summary = TrainingSummary {
        model_id: finished_training.decision_model_id,
        scope: finished_training.scope,
        example_count: finished_training.labelled_examples.len(),
        distinct_state_count: data_counts.distinct_state_count,
        part_counts: finished_training.part_counts,
        indistinguishable_question_count: outcome.indistinguishable_question_count,
        largest_parity_gap: outcome.largest_parity_gap,
        parity_gap_beyond_rounding: outcome.parity_gap_beyond_rounding,
        recomputed_text_count: outcome.recomputed_text_count,
        runs: &outcome.runs,
        best_seed: outcome.best_seed,
        verdict: &verdict,
    };
    // Everything that can still fail is done before the report says the
    // folder will be written.
    let decision_adapters =
        decision_adapters_of(&outcome.adapter_tensors).map_err(unpaired_add_ons)?;
    let calibration_override = merged_calibration(
        &finished_training.earlier_calibration,
        verdict.calibration_override(),
    );
    write_report(
        dependencies.stdout,
        &training_report_text(&training_summary),
    )?;
    let training_record = TrainingRecord {
        verdict: verdict.recorded_verdict(),
        labelled_question_count: data_counts.labelled_question_count,
        distinct_state_count: data_counts.distinct_state_count,
        warnings: training_warnings(&training_summary),
    }
    .after_earlier(finished_training.earlier_record.as_ref());
    Ok(JudgedTraining {
        calibration_override,
        head_tuning: head_tuning_of(&outcome.trained_tensors),
        decision_adapters,
        verdict,
        judging_pairs: training_scores.judging,
        training_record,
    })
}

fn unpaired_add_ons(unpaired: UnpairedAddOns) -> DecisionToolError {
    DecisionToolError::TrainingFailed(format!("{unpaired}; nothing was written"))
}

/// What writing and proving the trained folder reads.
struct WrittenTraining<'written> {
    output_directory: &'written Path,
    model_folder: &'written ModelFolderLookup,
    family: TunableHeadFamily,
    judged_training: &'written JudgedTraining,
    labelled_examples: &'written [LabelledExample],
}

/// Writes the folder, has the daemon serve it, and checks its answers. The
/// folder is kept whatever the check finds: one that could not be checked
/// (Astronomical busy, say) may be perfectly sound, and one whose answers
/// differ from the trained model is the person's to inspect; the failure
/// says which, and the run exits 1.
async fn write_and_prove(
    written_training: &WrittenTraining<'_>,
    daemon_probe: &DaemonProbe,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<(), DecisionToolError> {
    let judged_training = written_training.judged_training;
    let folder_additions = FolderAdditions {
        calibration_override: judged_training.calibration_override.as_ref(),
        head_tuning: Some((&judged_training.head_tuning, written_training.family)),
        decision_adapters: judged_training.decision_adapters.as_ref(),
        training_record: Some(&judged_training.training_record),
    };
    let output_directory = written_training.output_directory;
    write_model_folder(
        output_directory,
        written_training.model_folder,
        folder_additions,
        dependencies,
    )?;
    let new_model_id = output_model_id(output_directory);
    let proof =
        prove_written_folder(written_training, &new_model_id, daemon_probe, dependencies).await;
    let largest_gap = match proof {
        Ok(largest_gap) => largest_gap,
        Err(proof_error) => return Err(kept_folder_failure(proof_error)),
    };
    let mut closing_text = written_folder_text(
        output_directory,
        &new_model_id,
        NewModelVisibility::Listed,
        &written_training.model_folder.scanned_model_roots,
    );
    closing_text.push_str(&checked_answers_text(
        judged_training.verdict.judged_on,
        judged_training.judging_pairs.len(),
        largest_gap,
    ));
    write_report(dependencies.stdout, &closing_text)
}

/// How closely the written folder answered as the trained model did.
fn checked_answers_text(judged_on: JudgedOn, checked_count: usize, largest_gap: f32) -> String {
    if checked_count == 0 {
        return "No questions were left to check its answers against.\n".to_owned();
    }
    let checked_questions = match judged_on {
        JudgedOn::HeldBackQuestions => "judging questions",
        JudgedOn::LearnedFromExamples => "training questions",
    };
    format!(
        "Its answers to the {checked_questions} match the trained model's to within {largest_gap:.4}.\n"
    )
}

/// A failed check of the written folder, said so the person knows the
/// folder is still there: a disagreement as it is, anything else as a check
/// that could not be made.
fn kept_folder_failure(proof_error: DecisionToolError) -> DecisionToolError {
    match proof_error {
        DecisionToolError::SavedModelDisagrees { .. }
        | DecisionToolError::SavedModelUnchecked(_) => proof_error,
        other_error => DecisionToolError::SavedModelUnchecked(other_error.to_string()),
    }
}

async fn prove_written_folder(
    written_training: &WrittenTraining<'_>,
    new_model_id: &str,
    daemon_probe: &DaemonProbe,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<f32, DecisionToolError> {
    let visibility =
        make_new_model_visible(dependencies.loopback_address, daemon_probe, new_model_id).await;
    if visibility != NewModelVisibility::Listed {
        return Err(DecisionToolError::SavedModelUnchecked(
            "Astronomical did not serve the new folder, so its answers could not be checked; try again when it is idle".to_owned(),
        ));
    }
    let judged_training = written_training.judged_training;
    check_saved_model(
        new_model_id,
        written_training.labelled_examples,
        &TrainedExpectation {
            calibration_plan: &judged_training.verdict.calibration_plan,
            judging_pairs: &judged_training.judging_pairs,
        },
        daemon_probe,
        dependencies,
    )
    .await
}

/// The source folder's own calibration, read before training so a file
/// that cannot be read stops the run before anything is reported; empty
/// when the folder has none.
fn earlier_calibration(source_directory: &Path) -> Result<CalibrationOverride, DecisionToolError> {
    let calibration_path = source_directory.join(DECISION_CALIBRATION_FILE_NAME);
    match std::fs::read(&calibration_path) {
        Ok(calibration_bytes) => match parse_calibration_document(&calibration_bytes) {
            Ok(earlier_calibration) => Ok(earlier_calibration),
            Err(parse_error) => Err(DecisionToolError::EarlierCalibrationUnreadable(
                parse_error.to_string(),
            )),
        },
        Err(read_error) if read_error.kind() == std::io::ErrorKind::NotFound => {
            Ok(CalibrationOverride::default())
        }
        Err(read_error) => Err(DecisionToolError::EarlierCalibrationUnreadable(
            read_error.to_string(),
        )),
    }
}

/// The folder's temperatures after training: its earlier calibration, with
/// each kind training refitted replaced. A kind training did not refit keeps
/// the temperature it was served and judged with; `None` when no kind has
/// one, since an empty calibration file would not load.
fn merged_calibration(
    earlier_calibration: &CalibrationOverride,
    trained_calibration: &CalibrationOverride,
) -> Option<CalibrationOverride> {
    let mut merged = earlier_calibration.clone();
    for kind in [
        decision_contract::DecisionQuestionKind::Choice,
        decision_contract::DecisionQuestionKind::Score,
        decision_contract::DecisionQuestionKind::Noul,
    ] {
        if let Some(kind_calibration) = trained_calibration.kind_calibration(kind) {
            merged.set_kind_calibration(kind, kind_calibration.clone());
        }
    }
    if merged.decides_no_kind() {
        return None;
    }
    Some(merged)
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod tests;
