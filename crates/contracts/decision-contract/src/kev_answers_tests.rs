//! Unit contracts for Kev's answers and confidences, with the expected
//! values worked out from the reference's `choice_confidence`,
//! `score_confidence` and `to_answers`.

use super::{kev_answer_from_probabilities, kev_choice_confidence, kev_score_confidence};
use crate::answer::DecisionAnswer;
use crate::system_one_questions::parse_system_one_questions;
use crate::test_support::ordered_json;

fn single_question(questions_json: &str) -> crate::question::DecisionQuestion {
    parse_system_one_questions(&ordered_json(questions_json))
        .expect("the test question should be valid")
        .remove(0)
}

#[test]
fn should_compute_choice_confidence_from_uniform_to_certain() {
    assert_eq!(kev_choice_confidence(&[0.25, 0.25, 0.25, 0.25]), 0.0);
    assert_eq!(kev_choice_confidence(&[0.0, 1.0, 0.0]), 1.0);
    assert!((kev_choice_confidence(&[0.7, 0.3]) - 0.4).abs() < 1e-12);
    assert_eq!(kev_choice_confidence(&[1.0]), 1.0);
    assert_eq!(
        kev_choice_confidence(&[0.0, 0.0]),
        0.0,
        "all zeros reads as uniform"
    );
}

#[test]
fn should_compute_score_confidence_from_the_spread_around_the_mode() {
    assert_eq!(kev_score_confidence(&[0.0, 1.0, 0.0]), 1.0);
    // Uniform over three levels: mode 0, E|level - 0| = 1, D = 2/3.
    assert_eq!(kev_score_confidence(&[1.0 / 3.0; 3]), 0.0);
    // Mode 1, E|level - 1| = 0.2, D = 2/3, so 1 - 0.3 = 0.7.
    assert!((kev_score_confidence(&[0.1, 0.8, 0.1]) - 0.7).abs() < 1e-12);
    assert_eq!(kev_score_confidence(&[1.0]), 1.0);
}

#[test]
fn should_answer_with_kev_confidences_and_a_kev_score_legend() {
    let choice = single_question(
        r#"{"house": {"type": "choice", "criteria": {"montague": null, "capulet": null}}}"#,
    );
    let score = single_question(
        r#"{"grief": {"type": "score", "criteria": ["calm", {"level": "distraught"}]}}"#,
    );

    let DecisionAnswer::Choice(choice_answer) =
        kev_answer_from_probabilities(&choice, &[0.7, 0.3]).expect("a choice should answer")
    else {
        panic!("a choice question answers with a choice");
    };
    let DecisionAnswer::Score(score_answer) =
        kev_answer_from_probabilities(&score, &[0.25, 0.75]).expect("a score should answer")
    else {
        panic!("a score question answers with a score");
    };

    assert_eq!(choice_answer.chosen_key, "montague");
    assert!((choice_answer.distribution_confidence - 0.4).abs() < 1e-12);
    assert_eq!(score_answer.level_legend, ["calm", "level: distraught"]);
    assert!((score_answer.distribution_confidence - 0.5).abs() < 1e-12);
}

#[test]
fn should_refuse_probabilities_that_do_not_match_the_question() {
    let choice = single_question(
        r#"{"house": {"type": "choice", "criteria": {"montague": null, "capulet": null}}}"#,
    );

    assert!(kev_answer_from_probabilities(&choice, &[1.0]).is_err());
}
