//! The running tally a decision-model parity journey keeps while it compares
//! an engine's answers with a float32 reference.
//!
//! Every decision family (Clef, Kev, Laya) reports the same figures on the
//! same log lines and judges them against the same kind of tolerance, so the
//! tally and its reporting live here once and each journey supplies only its
//! family's log prefix and fixture layout.

use decision_contract::QuestionOutcome;

use super::decision_reference_fixture::{answer_probabilities, most_probable_index};

/// What a comparison measured.
#[derive(Default)]
pub(super) struct ParityMeasurement {
    pub(super) question_count: usize,
    same_answer_count: usize,
    pub(super) largest_probability_difference: f64,
    probability_difference_total: f64,
    probability_count: usize,
    largest_logit_difference: f64,
}

impl ParityMeasurement {
    /// Folds one question's probabilities and logits into the tally and logs
    /// them. `extra_log_fields` is appended after `options=` for families that
    /// report more per question (Laya prints the applied temperature).
    pub(super) fn record_question(
        &mut self,
        family_label: &str,
        case_name: &str,
        question_outcome: &QuestionOutcome,
        reference_probabilities: &[f64],
        reference_logits: &[f64],
        extra_log_fields: &str,
    ) {
        let probabilities = answer_probabilities(&question_outcome.answer);
        assert_eq!(probabilities.len(), reference_probabilities.len());
        let mut question_probability_difference = 0.0_f64;
        for (probability, reference_probability) in
            probabilities.iter().zip(reference_probabilities)
        {
            let probability_difference = (probability - reference_probability).abs();
            question_probability_difference =
                question_probability_difference.max(probability_difference);
            self.probability_difference_total += probability_difference;
            self.probability_count += 1;
        }
        self.largest_probability_difference = self
            .largest_probability_difference
            .max(question_probability_difference);
        for (option_logit, reference_logit) in
            question_outcome.option_logits.iter().zip(reference_logits)
        {
            self.largest_logit_difference = self
                .largest_logit_difference
                .max((f64::from(*option_logit) - reference_logit).abs());
        }
        eprintln!(
            "[{family_label}-parity] case={case_name} question={} options={}{extra_log_fields} largest_probability_difference={question_probability_difference:.5}",
            question_outcome.question_name,
            probabilities.len(),
        );
        self.question_count += 1;
        if most_probable_index(&probabilities) == most_probable_index(reference_probabilities) {
            self.same_answer_count += 1;
        }
    }

    /// Logs the totals; `log_prefix` is the family's bracketed label plus any
    /// fields that identify the checkpoint.
    pub(super) fn report(&self, log_prefix: &str) {
        eprintln!(
            "{log_prefix} questions={} same_answer={} largest_probability_difference={:.5} mean_probability_difference={:.5} largest_logit_difference={:.5}",
            self.question_count,
            self.same_answer_count,
            self.largest_probability_difference,
            self.probability_difference_total / self.probability_count as f64,
            self.largest_logit_difference,
        );
    }

    pub(super) fn assert_within_tolerance(&self, maximum_probability_difference: f64) {
        assert!(
            self.largest_probability_difference <= maximum_probability_difference,
            "the largest probability difference {:.5} exceeds {maximum_probability_difference}",
            self.largest_probability_difference
        );
    }
}
