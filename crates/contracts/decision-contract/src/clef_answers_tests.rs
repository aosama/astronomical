//! Unit contracts for Clef's answers, with the expected values worked out
//! from the reference's `systemone_answer`.

use super::clef_answer_from_scored_probabilities;
use crate::answer::DecisionAnswer;
use crate::clef_rendering::clef_question_text;
use crate::test_support::single_question;

#[test]
fn should_put_choice_probabilities_back_in_the_callers_order() {
    let question = single_question(
        r#"{"house": {"type": "choice", "criteria": {"montague": null, "capulet": null}}}"#,
    );
    let question_text = clef_question_text(&question);

    // Scored in key order: capulet, then montague.
    let DecisionAnswer::Choice(choice_answer) =
        clef_answer_from_scored_probabilities(&question, &question_text, &[0.8, 0.2])
            .expect("a choice answers")
    else {
        panic!("a choice question answers with a choice");
    };

    assert_eq!(choice_answer.chosen_key, "capulet");
    assert_eq!(choice_answer.option_probabilities[0].key, "montague");
    assert!((choice_answer.option_probabilities[0].probability - 0.2).abs() < 1e-12);
    assert!(
        (choice_answer.distribution_confidence - 0.8).abs() < 1e-12,
        "Clef's confidence is the chosen option's probability"
    );
}

#[test]
fn should_read_noul_probabilities_scored_true_first() {
    let question = single_question(r#"{"dead": {"type": "noul"}}"#);
    let question_text = clef_question_text(&question);

    let DecisionAnswer::Noul(noul_answer) =
        clef_answer_from_scored_probabilities(&question, &question_text, &[0.9, 0.1])
            .expect("a noul answers")
    else {
        panic!("a noul question answers with a noul");
    };

    assert!((noul_answer.probability_true - 0.9).abs() < 1e-12);
}

#[test]
fn should_report_the_most_likely_level_as_a_scores_confidence_with_the_callers_legend() {
    let question = single_question(
        r#"{"grief": {"type": "score", "criteria": ["calm", {"level": "distraught"}]}}"#,
    );
    let question_text = clef_question_text(&question);

    let DecisionAnswer::Score(score_answer) =
        clef_answer_from_scored_probabilities(&question, &question_text, &[0.25, 0.75])
            .expect("a score answers")
    else {
        panic!("a score question answers with a score");
    };

    assert!((score_answer.expected_level - 0.75).abs() < 1e-12);
    assert!((score_answer.distribution_confidence - 0.75).abs() < 1e-12);
    assert_eq!(
        score_answer.level_legend,
        ["calm", r#"{"level":"distraught"}"#]
    );
}

#[test]
fn should_refuse_a_probability_count_that_does_not_match_the_options() {
    let question = single_question(r#"{"dead": {"type": "noul"}}"#);
    let question_text = clef_question_text(&question);

    assert!(clef_answer_from_scored_probabilities(&question, &question_text, &[1.0]).is_err());
    assert!(
        clef_answer_from_scored_probabilities(&question, &question_text, &[0.5, 0.25, 0.25])
            .is_err()
    );
}
