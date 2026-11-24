//! How Clef turns the probabilities of its scored options into answers.
//!
//! Clef scores options in its own order (`clef_rendering`), so the
//! probabilities are first put back in the contract's canonical order. Its
//! `systemone_answer` then reports, as a question's confidence, the chosen
//! option's probability for a choice and the most likely level's for a
//! score, not a measure of the whole distribution; the legend of a score is
//! each level as the caller wrote it.

use crate::answer::{AnswerError, DecisionAnswer, answer_from_probabilities};
use crate::clef_rendering::{ClefQuestionText, clef_render};
use crate::question::{DecisionQuestion, DecisionQuestionBody};

/// The answer Clef gives for `scored_probabilities`, one per option of
/// `question_text` in the order Clef scored them.
pub fn clef_answer_from_scored_probabilities(
    question: &DecisionQuestion,
    question_text: &ClefQuestionText,
    scored_probabilities: &[f64],
) -> Result<DecisionAnswer, AnswerError> {
    if question_text.options.len() != scored_probabilities.len() {
        // The shared answer reports the mismatch with both counts.
        return answer_from_probabilities(question, scored_probabilities);
    }
    let mut canonical_probabilities = vec![f64::NAN; scored_probabilities.len()];
    for (scored_option, scored_probability) in
        question_text.options.iter().zip(scored_probabilities)
    {
        if let Some(canonical_slot) = canonical_probabilities.get_mut(scored_option.canonical_index)
        {
            *canonical_slot = *scored_probability;
        }
    }
    let mut answer = answer_from_probabilities(question, &canonical_probabilities)?;
    match (&mut answer, &question.body) {
        (DecisionAnswer::Choice(choice_answer), _) => {
            choice_answer.distribution_confidence = choice_answer.answer_confidence;
        }
        (DecisionAnswer::Score(score_answer), DecisionQuestionBody::Score { levels }) => {
            score_answer.distribution_confidence = score_answer.answer_confidence;
            let mut level_legend = Vec::with_capacity(levels.len());
            for level in levels {
                level_legend.push(clef_render(level));
            }
            score_answer.level_legend = level_legend;
        }
        _ => {}
    }
    Ok(answer)
}

#[cfg(test)]
#[path = "clef_answers_tests.rs"]
mod tests;
