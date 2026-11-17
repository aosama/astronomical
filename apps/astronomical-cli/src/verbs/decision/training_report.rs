//! The plain-language report `decision train` prints.
//!
//! The numbers keep calibration's names: "right" is accuracy, "surprise at
//! the right answer" the log loss, and "confidence off by" the expected
//! calibration error. Three rows are shown so the person can see where an
//! improvement comes from: the model as it serves today, the shipped head
//! with fresh temperatures (what calibration alone would give), and the
//! trained head.
//!
//! The report never refuses anything: the folder is written whatever the
//! comparison shows, and the report ends with the verdict and every concern
//! so the person can decide whether to use it. The same concern sentences
//! go into the folder's training record.

use std::fmt::Write as _;

use astronomical_decision_contract::DecisionQuestionKind;
use astronomical_decision_contract::decision_training_job::{RunEnding, RunSummary, TrainingScope};
use astronomical_decision_contract::training_record::RecordedVerdict;
use astronomical_decision_training::calibration_metrics::CalibrationMetrics;
use astronomical_decision_training::training_split::HOLD_BACK_STATE_COUNT;
use astronomical_decision_training::training_verdict::{
    JudgedOn, TrainingConcern, TrainingVerdict,
};

use super::counted_noun::counted_noun;
use super::training_job::PartCounts;

/// Everything the report describes.
pub(super) struct TrainingSummary<'summary> {
    pub(super) model_id: &'summary str,
    pub(super) scope: TrainingScope,
    pub(super) example_count: usize,
    pub(super) distinct_state_count: usize,
    pub(super) part_counts: PartCounts,
    pub(super) indistinguishable_question_count: usize,
    pub(super) largest_parity_gap: f32,
    /// The worst gap past rounding when the trainable copy did not
    /// reproduce the model.
    pub(super) parity_gap_beyond_rounding: Option<f32>,
    /// Texts too long to hold in memory at once, trained layer by layer.
    pub(super) recomputed_text_count: usize,
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
        training_summary.verdict.judged_on,
    ));
    report_text.push_str(&recomputation_text(training_summary.recomputed_text_count));
    report_text.push_str(&comparison_text(training_summary.verdict));
    report_text.push_str(&verdict_text(training_summary));
    report_text
}

/// Every warning the report gives, one sentence each, for the folder's
/// training record.
pub(super) fn training_warnings(training_summary: &TrainingSummary<'_>) -> Vec<String> {
    let mut warnings = Vec::new();
    if let Some(gap) = training_summary.parity_gap_beyond_rounding {
        warnings.push(parity_warning(gap));
    }
    if training_summary.verdict.judged_on == JudgedOn::LearnedFromExamples {
        warnings.push(not_judged_warning(training_summary.distinct_state_count));
    }
    for concern in &training_summary.verdict.concerns {
        warnings.push(concern_text(concern));
    }
    warnings
}

fn parity_warning(gap: f32) -> String {
    format!(
        "before training, the trainable copy scored up to {gap:.5} away from the model, more than rounding explains, so training started from slightly the wrong place"
    )
}

fn not_judged_warning(distinct_state_count: usize) -> String {
    format!(
        "with {} (fewer than {HOLD_BACK_STATE_COUNT}), none could be held back, so it was compared only on the examples it learned from; that shows it learned them, not that it does well on new text",
        counted_noun(distinct_state_count, "different text", "different texts")
    )
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
    let question_count = training_count + choosing_count + judging_count;
    let mut opening = if training_summary.verdict.judged_on == JudgedOn::LearnedFromExamples {
        format!(
            "Training {what_trains} {} on {}: {} about {}, all of them to train on; fewer than {HOLD_BACK_STATE_COUNT} different texts leaves none to hold back.\n",
            training_summary.model_id,
            counted_noun(training_summary.example_count, "example", "examples"),
            counted_noun(question_count, "question", "questions"),
            counted_noun(
                training_summary.distinct_state_count,
                "different text",
                "different texts"
            ),
        )
    } else {
        format!(
            "Training {what_trains} {} on {}: {}, {training_count} to train on, {choosing_count} to choose with and {judging_count} held back to judge.\n",
            training_summary.model_id,
            counted_noun(training_summary.example_count, "example", "examples"),
            counted_noun(question_count, "question", "questions"),
        )
    };
    if training_summary.scope == TrainingScope::WholeModel {
        opening.push_str(
            "Every layer gets small add-on weights that learn alongside the final layers; the original weights are never changed.\n",
        );
    }
    let left_out_count = training_summary.indistinguishable_question_count;
    if left_out_count > 0 {
        let was_or_were = if left_out_count == 1 { "was" } else { "were" };
        let _ = writeln!(
            opening,
            "{} {was_or_were} left out because some of their options read the same to the model.",
            counted_noun(left_out_count, "question", "questions"),
        );
    }
    match training_summary.parity_gap_beyond_rounding {
        None => {
            let _ = writeln!(
                opening,
                "Before training, {trainable_copy} reproduced the model's own scores to within {:.4}.\n",
                training_summary.largest_parity_gap
            );
        }
        Some(gap) => {
            let _ = writeln!(opening, "Warning: {}.\n", parity_warning(gap));
        }
    }
    opening
}

/// Says when a text was too long to hold in memory at once, which explains
/// a slower run; nothing for the usual case.
fn recomputation_text(recomputed_text_count: usize) -> String {
    if recomputed_text_count == 0 {
        return String::new();
    }
    let pronoun = if recomputed_text_count == 1 {
        "it"
    } else {
        "them"
    };
    format!(
        "{} too long to hold in memory at once, so the model trained on {pronoun} layer by layer: slower, same result.\n\n",
        counted_noun(recomputed_text_count, "text was", "texts were")
    )
}

fn runs_text(runs: &[RunSummary], best_seed: u64, judged_on: JudgedOn) -> String {
    let best_marker = match judged_on {
        JudgedOn::HeldBackQuestions => ", the best on the questions held back for choosing",
        JudgedOn::LearnedFromExamples => ", the best on its own training examples",
    };
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
            best_marker
        } else {
            ""
        };
        let _ = writeln!(runs_text, "Run {run_number}: {story}{marker}.");
    }
    runs_text.push('\n');
    runs_text
}

fn comparison_text(verdict: &TrainingVerdict) -> String {
    let question_count = verdict.served_overall.question_count;
    let judged_questions = match verdict.judged_on {
        JudgedOn::HeldBackQuestions => format!(
            "{}, which training never saw",
            counted_noun(question_count, "judging question", "judging questions")
        ),
        JudgedOn::LearnedFromExamples => format!(
            "{} themselves, which it learned from",
            counted_noun(question_count, "training question", "training questions")
        ),
    };
    let mut comparison = format!(
        "On the {judged_questions}:\n                          right   surprise   confidence off by\n"
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

fn verdict_text(training_summary: &TrainingSummary<'_>) -> String {
    let verdict = training_summary.verdict;
    let mut verdict_text = match verdict.recorded_verdict() {
        RecordedVerdict::JudgedBetter => {
            "Judged better: the trained model beats the shipped one on questions it never saw.\n"
                .to_owned()
        }
        RecordedVerdict::JudgedNotBetter => {
            "Judged not better: on questions it never saw, the trained model\n".to_owned()
        }
        RecordedVerdict::NotJudged => format!(
            "Not judged: {}.\n",
            not_judged_warning(training_summary.distinct_state_count)
        ),
    };
    if !verdict.concerns.is_empty() {
        if verdict.judged_on == JudgedOn::LearnedFromExamples {
            verdict_text.push_str("On its own examples, the trained model\n");
        }
        for concern in &verdict.concerns {
            let _ = writeln!(verdict_text, "  - {}", concern_text(concern));
        }
    }
    verdict_text.push_str(
        "The new folder is written either way, and its answers carry this verdict; whether to use it is your call.\n",
    );
    verdict_text
}

fn concern_text(concern: &TrainingConcern) -> String {
    match concern {
        TrainingConcern::LessAccurate { shipped, trained } => format!(
            "was right less often ({:.1}% -> {:.1}%)",
            shipped * 100.0,
            trained * 100.0
        ),
        TrainingConcern::NotLessSurprised { shipped, trained } => format!(
            "was not less surprised by the right answers than the shipped model recalibrated ({shipped:.3} -> {trained:.3})"
        ),
        TrainingConcern::KindMoreSurprised {
            kind,
            question_count,
            shipped,
            trained,
        } => format!(
            "was more surprised by the right answers to {} ({shipped:.3} -> {trained:.3}, over {})",
            kind_label(*kind).to_lowercase(),
            counted_noun(*question_count, "question", "questions")
        ),
        TrainingConcern::CheckAnswersChanged(agreement) => format!(
            "changed the answers to {} of {} questions nobody labelled",
            agreement.question_count - agreement.agreeing_count,
            agreement.question_count
        ),
        TrainingConcern::Collapsed { kind } => format!(
            "gave nearly every {} the same answer although the right answers vary",
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
