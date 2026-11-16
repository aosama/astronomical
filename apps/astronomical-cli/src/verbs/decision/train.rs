//! The `astronomical decision train` journey.
//!
//! 1. Check what needs no model: the examples file, that there is enough of
//!    it, the model's folder and family, and the output folder.
//! 2. Send one job to the daemon: the worker reads every example and the
//!    built-in check set, proves its trainable copy (the final layers, or
//!    with `--whole-model` the whole model with add-on weights in every
//!    layer) reproduces the model's scores, and trains on the GPU in several
//!    runs. It returns every answer beside the trained model's scores.
//! 3. Fit fresh temperatures and judge the result against the shipped model
//!    recalibrated the same way, on questions training never saw.
//! 4. Only if it is kept: write the new folder (originals copied byte for
//!    byte, trained layers and any add-ons in 32-bit files beside them),
//!    reload, and prove
//!    the written folder answers as the trained model did; otherwise remove
//!    it.

use std::collections::HashSet;
use std::path::Path;

use astronomical_config::ModelFamily;
use astronomical_decision_contract::calibration_file::{
    CalibrationOverride, DECISION_CALIBRATION_FILE_NAME, parse_calibration_document,
};
use astronomical_decision_contract::decision_adapters_file::DecisionAdapters;
use astronomical_decision_contract::decision_training_job::{
    DecisionTrainingOutcome, TrainingScope,
};
use astronomical_decision_contract::head_tuning_file::{HeadTuning, TunableHeadFamily};
use astronomical_decision_contract::trained_tensors::{
    UnpairedAddOns, decision_adapters_of, head_tuning_of,
};
use astronomical_decision_training::calibrated_model_folder::FolderAdditions;
use astronomical_decision_training::holdout_split::state_group_key;
use astronomical_decision_training::labelled_example::LabelledExample;
use astronomical_decision_training::training_split::{SplitPart, check_training_data};
use astronomical_decision_training::training_verdict::{
    ScoredPair, TrainingVerdict, judge_training,
};

use super::check_set::built_in_check_examples;
use super::decision_arguments::DecisionToolArguments;
use super::decision_tool::{
    DecisionToolDependencies, DecisionToolError, check_output_is_new, model_lifecycle,
    output_model_id, prepare_decision_model, read_labelled_examples, write_model_folder,
    write_report,
};
use super::model_folder_lookup::{ModelFolderLookup, find_model_folder};
use super::new_model_visibility::{
    NewModelVisibility, make_new_model_visible, written_folder_text,
};
use super::saved_model_check::{TrainedExpectation, check_saved_model};
use super::training_job::{PartCounts, training_job, training_scores};
use super::training_report::{TrainingSummary, training_report_text};
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
}

/// A head the gate kept, with what it was judged by.
struct KeptTraining {
    head_tuning: HeadTuning,
    /// The add-ons of a whole-model training; `None` for the final layers.
    decision_adapters: Option<DecisionAdapters>,
    verdict: TrainingVerdict,
    judging_pairs: Vec<ScoredPair>,
}

/// Runs the whole training against the resident daemon.
pub async fn run_train(
    train_arguments: &DecisionToolArguments,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<(), DecisionToolError> {
    let labelled_examples = read_labelled_examples(&train_arguments.examples_file_path)?;
    check_enough_data(&labelled_examples, train_arguments.training_scope)?;
    let model_lifecycle = model_lifecycle(dependencies);
    let decision_model_id = prepare_decision_model(
        train_arguments.model_id.as_deref(),
        &model_lifecycle,
        dependencies,
    )
    .await?;
    let model_folder = find_model_folder(&dependencies.instance_paths, &decision_model_id)?;
    let family = tunable_family(model_folder.model_family, &decision_model_id)?;
    check_scope_is_trainable(family, train_arguments.training_scope, &decision_model_id)?;
    let output_directory = &train_arguments.output_directory;
    check_output_is_new(&model_folder, output_directory)?;
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
    let kept_training = judge_and_report(
        FinishedTraining {
            decision_model_id: &decision_model_id,
            scope: train_arguments.training_scope,
            labelled_examples: &labelled_examples,
            example_parts: &prepared_job.example_parts,
            sent_example_count,
            part_counts: prepared_job.part_counts,
            outcome,
        },
        dependencies,
    )?;
    write_and_prove(
        &WrittenTraining {
            output_directory,
            model_folder: &model_folder,
            family,
            kept_training: &kept_training,
            labelled_examples: &labelled_examples,
        },
        daemon_probe,
        dependencies,
    )
    .await
}

fn check_enough_data(
    labelled_examples: &[LabelledExample],
    scope: TrainingScope,
) -> Result<(), DecisionToolError> {
    let mut distinct_states = HashSet::new();
    let mut labelled_question_count = 0;
    for labelled_example in labelled_examples {
        distinct_states.insert(state_group_key(&labelled_example.state));
        labelled_question_count += labelled_example.labelled_questions.len();
    }
    match check_training_data(labelled_question_count, distinct_states.len(), scope) {
        Ok(()) => Ok(()),
        Err(too_little_data) => Err(DecisionToolError::TooLittleData(too_little_data)),
    }
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

/// Refuses a whole-model training of a family that has none yet, before
/// any example is read: Laya trains as a whole; Kev and Clef train only
/// their final layers so far.
fn check_scope_is_trainable(
    family: TunableHeadFamily,
    scope: TrainingScope,
    model_id: &str,
) -> Result<(), DecisionToolError> {
    match (scope, family) {
        (TrainingScope::FinalLayers, _) | (TrainingScope::WholeModel, TunableHeadFamily::Laya) => {
            Ok(())
        }
        (TrainingScope::WholeModel, TunableHeadFamily::Kev | TunableHeadFamily::Clef) => {
            Err(DecisionToolError::WholeModelNotTrainable {
                model_id: model_id.to_owned(),
            })
        }
    }
}

/// Judges the trained head, prints the report, and returns the head only
/// when the gate keeps it.
fn judge_and_report(
    finished_training: FinishedTraining<'_>,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<KeptTraining, DecisionToolError> {
    let outcome = &finished_training.outcome;
    let training_scores = training_scores(
        finished_training.labelled_examples,
        finished_training.example_parts,
        finished_training.sent_example_count,
        outcome,
    )?;
    let verdict = judge_training(&training_scores)?;
    let report_text = training_report_text(&TrainingSummary {
        model_id: finished_training.decision_model_id,
        scope: finished_training.scope,
        example_count: finished_training.labelled_examples.len(),
        part_counts: finished_training.part_counts,
        indistinguishable_question_count: outcome.indistinguishable_question_count,
        largest_parity_gap: outcome.largest_parity_gap,
        runs: &outcome.runs,
        best_seed: outcome.best_seed,
        verdict: &verdict,
    });
    write_report(dependencies.stdout, &report_text)?;
    if !verdict.is_kept() {
        return Err(DecisionToolError::TrainingRefused {
            model_id: finished_training.decision_model_id.to_owned(),
        });
    }
    Ok(KeptTraining {
        head_tuning: head_tuning_of(&outcome.trained_tensors),
        decision_adapters: decision_adapters_of(&outcome.adapter_tensors)
            .map_err(unpaired_add_ons)?,
        verdict,
        judging_pairs: training_scores.judging,
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
    kept_training: &'written KeptTraining,
    labelled_examples: &'written [LabelledExample],
}

/// Writes the folder, has the daemon serve it, and proves its answers. A
/// folder whose answers disagree with the trained model is removed; one that
/// could not be checked (Astronomical busy, say) is kept, because removing
/// it would throw away a training that may be perfectly sound.
async fn write_and_prove(
    written_training: &WrittenTraining<'_>,
    daemon_probe: &DaemonProbe,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<(), DecisionToolError> {
    let kept_training = written_training.kept_training;
    let calibration_override = merged_calibration(
        &written_training.model_folder.model_directory,
        kept_training.verdict.calibration_override(),
    )?;
    let folder_additions = FolderAdditions {
        calibration_override: calibration_override.as_ref(),
        head_tuning: Some((&kept_training.head_tuning, written_training.family)),
        decision_adapters: kept_training.decision_adapters.as_ref(),
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
        Err(proof_error) if disproves_written_folder(&proof_error) => {
            // Only the folder this run wrote is removed; the source is never touched.
            let _ = std::fs::remove_dir_all(output_directory);
            let _ =
                make_new_model_visible(dependencies.loopback_address, daemon_probe, &new_model_id)
                    .await;
            return Err(proof_error);
        }
        Err(DecisionToolError::SavedModelUnchecked(reason)) => {
            return Err(DecisionToolError::SavedModelUnchecked(reason));
        }
        Err(proof_error) => {
            return Err(DecisionToolError::SavedModelUnchecked(
                proof_error.to_string(),
            ));
        }
    };
    let mut closing_text = written_folder_text(
        output_directory,
        &new_model_id,
        NewModelVisibility::Listed,
        &written_training.model_folder.scanned_model_roots,
    );
    closing_text.push_str(&format!(
        "Its answers to the judging questions match the trained model's to within {largest_gap:.4}.\n"
    ));
    write_report(dependencies.stdout, &closing_text)
}

/// Whether a failed proof showed the written folder answers differently from
/// the trained model, rather than only that it could not be checked.
fn disproves_written_folder(proof_error: &DecisionToolError) -> bool {
    matches!(proof_error, DecisionToolError::SavedModelDisagrees { .. })
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
    let kept_training = written_training.kept_training;
    check_saved_model(
        new_model_id,
        written_training.labelled_examples,
        &TrainedExpectation {
            calibration_plan: &kept_training.verdict.calibration_plan,
            judging_pairs: &kept_training.judging_pairs,
        },
        daemon_probe,
        dependencies,
    )
    .await
}

/// The folder's temperatures after training: its earlier calibration, with
/// each kind training refitted replaced. A kind training did not refit keeps
/// the temperature it was served and judged with; `None` when no kind has
/// one, since an empty calibration file would not load.
fn merged_calibration(
    source_directory: &Path,
    trained_calibration: &CalibrationOverride,
) -> Result<Option<CalibrationOverride>, DecisionToolError> {
    let calibration_path = source_directory.join(DECISION_CALIBRATION_FILE_NAME);
    let mut merged = match std::fs::read(&calibration_path) {
        Ok(calibration_bytes) => match parse_calibration_document(&calibration_bytes) {
            Ok(earlier_calibration) => earlier_calibration,
            Err(parse_error) => {
                return Err(DecisionToolError::EarlierCalibrationUnreadable(
                    parse_error.to_string(),
                ));
            }
        },
        Err(read_error) if read_error.kind() == std::io::ErrorKind::NotFound => {
            CalibrationOverride::default()
        }
        Err(read_error) => {
            return Err(DecisionToolError::EarlierCalibrationUnreadable(
                read_error.to_string(),
            ));
        }
    };
    for kind in [
        astronomical_decision_contract::DecisionQuestionKind::Choice,
        astronomical_decision_contract::DecisionQuestionKind::Score,
        astronomical_decision_contract::DecisionQuestionKind::Noul,
    ] {
        if let Some(kind_calibration) = trained_calibration.kind_calibration(kind) {
            merged.set_kind_calibration(kind, kind_calibration.clone());
        }
    }
    if merged.decides_no_kind() {
        return Ok(None);
    }
    Ok(Some(merged))
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod tests;
