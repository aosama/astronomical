//! The plain-language report `decision calibrate` prints.
//!
//! The numbers are named for what they mean to the person, not by their
//! statistics names: "surprise at the right answer" is the log loss the fit
//! lowers, "confidence off by" is the expected calibration error, and
//! "right" is accuracy, which a temperature never changes.
//!
//! Every kind the examples ask is calibrated and written; the report says
//! for each whether the fit helped on held-out questions, did not, or could
//! only be compared on the questions it was fitted to. The person decides.

use std::fmt::Write as _;

use astronomical_decision_contract::DecisionQuestionKind;
use astronomical_decision_contract::calibration_file::KindCalibration;
use astronomical_decision_contract::training_record::RecordedVerdict;
use astronomical_decision_training::calibration_plan::{
    CalibrationPlan, KindFit, KindOutcome, KindVerdict,
};
use astronomical_decision_training::judged_on::JudgedOn;

use super::counted_noun::counted_noun;

/// The counts the report opens with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CalibrationCounts {
    pub example_count: usize,
    pub question_count: usize,
    pub distinct_state_count: usize,
    pub holdout_question_count: usize,
}

/// The whole report, ending in a newline.
#[must_use]
pub fn calibration_report_text(
    model_id: &str,
    calibration_counts: CalibrationCounts,
    calibration_plan: &CalibrationPlan,
) -> String {
    let mut report_text = format!(
        "Calibration of {model_id} on {}: {} about {}, {} of them held out to judge the fit.\n\n",
        counted_noun(calibration_counts.example_count, "example", "examples"),
        counted_noun(calibration_counts.question_count, "question", "questions"),
        counted_noun(
            calibration_counts.distinct_state_count,
            "different text",
            "different texts"
        ),
        calibration_counts.holdout_question_count,
    );
    for kind_verdict in &calibration_plan.kind_verdicts {
        report_text.push_str(&kind_verdict_text(kind_verdict));
    }
    report_text.push_str(&verdict_line(calibration_plan.recorded_verdict()));
    report_text
}

/// Each calibrated kind's finding in one sentence, for the folder's
/// training record; kinds that improved on held-out questions need none.
#[must_use]
pub fn calibration_warnings(calibration_plan: &CalibrationPlan) -> Vec<String> {
    let mut warnings = Vec::new();
    for kind_verdict in &calibration_plan.kind_verdicts {
        if let KindOutcome::Calibrated(kind_fit) = &kind_verdict.outcome
            && let Some(finding) = finding_text(kind_fit)
        {
            warnings.push(format!("{}: {finding}", kind_label(kind_verdict.kind)));
        }
    }
    warnings
}

fn verdict_line(recorded_verdict: RecordedVerdict) -> String {
    let verdict_text = match recorded_verdict {
        RecordedVerdict::JudgedBetter => "Judged better on held-out questions.",
        RecordedVerdict::JudgedNotBetter => {
            "Judged not better: at least one kind did no better on held-out questions."
        }
        RecordedVerdict::NotJudged => {
            "Not judged: at least one kind could only be compared on the questions it was fitted to."
        }
    };
    format!(
        "\n{verdict_text}\nThe new folder is written either way, and its answers carry this verdict; whether to use it is your call.\n"
    )
}

fn kind_verdict_text(kind_verdict: &KindVerdict) -> String {
    let kind_label = kind_label(kind_verdict.kind);
    match &kind_verdict.outcome {
        KindOutcome::Calibrated(kind_fit) => {
            let mut verdict_text = format!(
                "{kind_label}: calibrated, {}\n",
                temperatures_text(&kind_fit.kind_calibration)
            );
            verdict_text.push_str(&comparison_text(kind_fit));
            if let Some(finding) = finding_text(kind_fit) {
                let _ = writeln!(verdict_text, "  Note: {finding}.");
            }
            verdict_text
        }
        KindOutcome::NoQuestions => {
            format!("{kind_label}: as shipped, the examples ask none\n")
        }
    }
}

/// What a person should know about a fit, when anything: it was only
/// compared on what it learned from, or it did not help.
fn finding_text(kind_fit: &KindFit) -> Option<String> {
    match (kind_fit.judged_on, kind_fit.improved()) {
        (JudgedOn::LearnedFromExamples, _) => Some(
            "no question about a different text could be held back, so it was compared only on the questions it was fitted to"
                .to_owned(),
        ),
        (JudgedOn::HeldBackQuestions, false) => Some(
            "it did no better than the shipped temperatures on held-out questions".to_owned(),
        ),
        (JudgedOn::HeldBackQuestions, true) => None,
    }
}

fn kind_label(kind: DecisionQuestionKind) -> &'static str {
    match kind {
        DecisionQuestionKind::Choice => "Choice questions",
        DecisionQuestionKind::Score => "Score questions",
        DecisionQuestionKind::Noul => "Yes-or-no questions",
    }
}

fn temperatures_text(kind_calibration: &KindCalibration) -> String {
    let mut temperatures_text = format!("temperature {:.2}", kind_calibration.temperature);
    for (range_name, range_temperature) in &kind_calibration.range_temperatures {
        let _ = write!(
            temperatures_text,
            ", {range_temperature:.2} with {range_name} options"
        );
    }
    temperatures_text
}

fn comparison_text(kind_fit: &KindFit) -> String {
    let (shipped, calibrated) = (&kind_fit.shipped_metrics, &kind_fit.calibrated_metrics);
    let judged_questions = match kind_fit.judged_on {
        JudgedOn::HeldBackQuestions => counted_noun(
            calibrated.question_count,
            "held-out question",
            "held-out questions",
        ),
        JudgedOn::LearnedFromExamples => format!(
            "the {} it was fitted to",
            counted_noun(calibrated.question_count, "question", "questions")
        ),
    };
    format!(
        "  On {judged_questions}: surprise at the right answer {:.3} -> {:.3} (lower is better), \
         confidence off by {:.1}% -> {:.1}%, right {:.1}% of the time (unchanged).\n",
        shipped.mean_negative_log_likelihood,
        calibrated.mean_negative_log_likelihood,
        shipped.expected_calibration_error * 100.0,
        calibrated.expected_calibration_error * 100.0,
        calibrated.accuracy() * 100.0,
    )
}

#[cfg(test)]
#[path = "calibration_report_tests.rs"]
mod tests;
