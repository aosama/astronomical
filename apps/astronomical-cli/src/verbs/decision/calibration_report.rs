//! The plain-language report `decision calibrate` prints.
//!
//! The numbers are named for what they mean to the person, not by their
//! statistics names: "surprise at the right answer" is the log loss the fit
//! lowers, "confidence off by" is the expected calibration error, and
//! "right" is accuracy, which a temperature never changes.

use std::fmt::Write as _;

use astronomical_decision_contract::DecisionQuestionKind;
use astronomical_decision_contract::calibration_file::KindCalibration;
use astronomical_decision_training::calibration_metrics::CalibrationMetrics;
use astronomical_decision_training::calibration_plan::{
    CalibrationPlan, KeptShippedReason, KindOutcome, KindVerdict,
};

/// The counts the report opens with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CalibrationCounts {
    pub example_count: usize,
    pub question_count: usize,
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
        "Calibration of {model_id} on {} examples: {} questions, {} of them held out to judge the fit.\n\n",
        calibration_counts.example_count,
        calibration_counts.question_count,
        calibration_counts.holdout_question_count,
    );
    for kind_verdict in &calibration_plan.kind_verdicts {
        report_text.push_str(&kind_verdict_text(kind_verdict));
    }
    report_text
}

fn kind_verdict_text(kind_verdict: &KindVerdict) -> String {
    let kind_label = kind_label(kind_verdict.kind);
    match &kind_verdict.outcome {
        KindOutcome::Calibrated {
            kind_calibration,
            calibrated_holdout_metrics,
        } => {
            let mut verdict_text = format!(
                "{kind_label}: calibrated, {}\n",
                temperatures_text(kind_calibration)
            );
            if let Some(shipped_metrics) = &kind_verdict.shipped_holdout_metrics {
                verdict_text.push_str(&improvement_text(
                    shipped_metrics,
                    calibrated_holdout_metrics,
                ));
            }
            verdict_text
        }
        KindOutcome::KeptShipped(kept_reason) => {
            format!(
                "{kind_label}: kept as shipped, {}\n",
                kept_reason_text(kept_reason)
            )
        }
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

fn improvement_text(shipped: &CalibrationMetrics, calibrated: &CalibrationMetrics) -> String {
    format!(
        "  On held-out questions: surprise at the right answer {:.3} -> {:.3} (lower is better), \
         confidence off by {:.1}% -> {:.1}%, right {:.1}% of the time (unchanged).\n",
        shipped.mean_negative_log_likelihood,
        calibrated.mean_negative_log_likelihood,
        shipped.expected_calibration_error * 100.0,
        calibrated.expected_calibration_error * 100.0,
        calibrated.accuracy() * 100.0,
    )
}

fn kept_reason_text(kept_reason: &KeptShippedReason) -> String {
    match kept_reason {
        KeptShippedReason::NoQuestions => "the examples ask none".to_owned(),
        KeptShippedReason::TooFewFittingQuestions { have, need } => {
            format!("{have} questions to fit on, {need} needed")
        }
        KeptShippedReason::TooFewHoldoutQuestions { have, need } => format!(
            "{have} held-out questions, {need} needed; add examples about more different states"
        ),
        KeptShippedReason::NoImprovement { .. } => {
            "the fit did not beat the shipped temperatures on held-out questions".to_owned()
        }
    }
}

#[cfg(test)]
#[path = "calibration_report_tests.rs"]
mod tests;
