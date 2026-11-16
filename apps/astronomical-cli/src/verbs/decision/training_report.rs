//! The plain-language report `decision train` prints.
//!
//! The numbers keep calibration's names: "right" is accuracy, "surprise at
//! the right answer" the log loss, and "confidence off by" the expected
//! calibration error. Three rows are shown so the person can see where an
//! improvement comes from: the model as it serves today, the shipped head
//! with fresh temperatures (what calibration alone would give), and the
//! trained head.

use std::fmt::Write as _;

use astronomical_decision_contract::DecisionQuestionKind;
use astronomical_decision_contract::head_training_job::{RunEnding, RunSummary, TrainingScope};
use astronomical_decision_training::calibration_metrics::CalibrationMetrics;
use astronomical_decision_training::training_verdict::{TrainingRefusal, TrainingVerdict};

use super::training_job::PartCounts;

/// Everything the report describes.
pub(super) struct TrainingSummary<'summary> {
    pub(super) model_id: &'summary str,
    pub(super) scope: TrainingScope,
    pub(super) example_count: usize,
    pub(super) part_counts: PartCounts,
    pub(super) indistinguishable_question_count: usize,
    pub(super) largest_parity_gap: f32,
    pub(super) runs: &'summary [RunSummary],
    pub(super) best_seed: u64,
    pub(super) verdict: &'summary TrainingVerdict,
}

/// The whole report, ending in a newline.
pub(super) fn training_report_text(training_summary: &TrainingSummary<'_>) -> String {
    let mut report_text = opening_text(training_summary);
    report_text.push_str(&runs_text(
        training_summary.runs,
        training_summary.best_seed,
    ));
    report_text.push_str(&comparison_text(training_summary.verdict));
    report_text.push_str(&verdict_text(training_summary.verdict));
    report_text
}

fn opening_text(training_summary: &TrainingSummary<'_>) -> String {
    let part_counts = training_summary.part_counts;
    let training_count = part_counts.training;
    let choosing_count = part_counts.choosing;
    let judging_count = part_counts.judging;
    let (what_trains, trainable_copy) = match training_summary.scope {
        TrainingScope::FinalLayers => ("the final layers of", "the GPU copy of the final layers"),
        TrainingScope::WholeModel => ("the whole of", "the trainable copy of the whole model"),
    };
    let mut opening = format!(
        "Training {what_trains} {} on {} examples: {} questions, {training_count} to train on, {choosing_count} to choose with and {judging_count} held back to judge.\n",
        training_summary.model_id,
        training_summary.example_count,
        training_count + choosing_count + judging_count,
    );
    if training_summary.scope == TrainingScope::WholeModel {
        opening.push_str(
            "Every layer gets small add-on weights that learn alongside the final layers; the original weights are never changed.\n",
        );
    }
    if training_summary.indistinguishable_question_count > 0 {
        let _ = writeln!(
            opening,
            "{} questions were left out because some of their options read the same to the model.",
            training_summary.indistinguishable_question_count
        );
    }
    let _ = writeln!(
        opening,
        "Before training, {trainable_copy} reproduced the model's own scores to within {:.4}.\n",
        training_summary.largest_parity_gap
    );
    opening
}

fn runs_text(runs: &[RunSummary], best_seed: u64) -> String {
    let mut runs_text = String::new();
    for (run_position, seed_run) in runs.iter().enumerate() {
        let run_number = run_position + 1;
        let story = match seed_run.ending {
            RunEnding::NonFiniteStep { epoch } => format!(
                "a number stopped being finite in pass {epoch}, so it kept its weights from before"
            ),
            _ if seed_run.best_epoch == 0 => format!(
                "no pass beat the starting weights in {} passes",
                seed_run.epochs_run
            ),
            _ => format!(
                "best after pass {} of {}",
                seed_run.best_epoch, seed_run.epochs_run
            ),
        };
        let marker = if seed_run.seed == best_seed {
            ", the best on the questions held back for choosing"
        } else {
            ""
        };
        let _ = writeln!(runs_text, "Run {run_number}: {story}{marker}.");
    }
    runs_text.push('\n');
    runs_text
}

fn comparison_text(verdict: &TrainingVerdict) -> String {
    let mut comparison = format!(
        "On the {} judging questions, which training never saw:\n                          right   surprise   confidence off by\n",
        verdict.served_overall.question_count
    );
    comparison.push_str(&metrics_row("as served today", &verdict.served_overall));
    comparison.push_str(&metrics_row(
        "shipped, recalibrated",
        &verdict.shipped_overall,
    ));
    comparison.push_str(&metrics_row("trained", &verdict.trained_overall));
    for kind_comparison in &verdict.kind_comparisons {
        let _ = writeln!(
            comparison,
            "  {} ({}): right {:.1}% -> {:.1}%, surprise {:.3} -> {:.3}",
            kind_label(kind_comparison.kind),
            kind_comparison.shipped.question_count,
            kind_comparison.shipped.accuracy() * 100.0,
            kind_comparison.trained.accuracy() * 100.0,
            kind_comparison.shipped.mean_negative_log_likelihood,
            kind_comparison.trained.mean_negative_log_likelihood,
        );
    }
    let agreement = verdict.check_agreement;
    let _ = writeln!(
        comparison,
        "Questions nobody labelled: {} of {} answered as before.\n",
        agreement.agreeing_count, agreement.question_count
    );
    comparison
}

fn metrics_row(row_label: &str, metrics: &CalibrationMetrics) -> String {
    format!(
        "  {row_label:<22} {:>6.1}%   {:>8.3}   {:>16.1}%\n",
        metrics.accuracy() * 100.0,
        metrics.mean_negative_log_likelihood,
        metrics.expected_calibration_error * 100.0,
    )
}

fn verdict_text(verdict: &TrainingVerdict) -> String {
    if verdict.is_kept() {
        return "Kept: the trained model beats the shipped one on questions it never saw.\n"
            .to_owned();
    }
    let mut verdict_text = "Not kept:\n".to_owned();
    for refusal in &verdict.refusals {
        let _ = writeln!(verdict_text, "  - {}", refusal_text(refusal));
    }
    verdict_text
}

fn refusal_text(refusal: &TrainingRefusal) -> String {
    match refusal {
        TrainingRefusal::NothingToJudge => {
            "no questions were held back to judge it; add examples about more different states"
                .to_owned()
        }
        TrainingRefusal::LessAccurate { shipped, trained } => format!(
            "it was right less often ({:.1}% -> {:.1}%)",
            shipped * 100.0,
            trained * 100.0
        ),
        TrainingRefusal::NotLessSurprised { shipped, trained } => format!(
            "it was not less surprised by the right answers than the shipped model recalibrated ({shipped:.3} -> {trained:.3})"
        ),
        TrainingRefusal::KindMoreSurprised {
            kind,
            shipped,
            trained,
        } => format!(
            "{} got more surprised by the right answers ({shipped:.3} -> {trained:.3})",
            kind_label(*kind)
        ),
        TrainingRefusal::CheckAnswersChanged(agreement) => format!(
            "it changed the answers to {} of {} questions nobody labelled; at most one in twenty may change",
            agreement.question_count - agreement.agreeing_count,
            agreement.question_count
        ),
        TrainingRefusal::Collapsed { kind } => format!(
            "it gave nearly every {} the same answer although the right answers vary",
            kind_label(*kind).to_lowercase().trim_end_matches('s')
        ),
    }
}

fn kind_label(kind: DecisionQuestionKind) -> &'static str {
    match kind {
        DecisionQuestionKind::Choice => "Choice questions",
        DecisionQuestionKind::Score => "Score questions",
        DecisionQuestionKind::Noul => "Yes-or-no questions",
    }
}

#[cfg(test)]
#[path = "training_report_tests.rs"]
mod tests;
