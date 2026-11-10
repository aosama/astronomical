//! Unit contracts for Kev's rendering, escaping and confidences, with the
//! expected text worked out from the reference's `render`, `option_text`,
//! `to_record`, `user_tokens`, `choice_confidence` and `score_confidence`.

use super::{
    kev_answer_from_probabilities, kev_choice_confidence, kev_escape_control_tokens,
    kev_question_text, kev_render, kev_score_confidence,
};
use crate::answer::DecisionAnswer;
use crate::system_one_questions::parse_system_one_questions;
use crate::test_support::ordered_json;

fn single_question(questions_json: &str) -> crate::question::DecisionQuestion {
    parse_system_one_questions(&ordered_json(questions_json))
        .expect("the test question should be valid")
        .remove(0)
}

#[test]
fn should_render_scalars_as_python_str_writes_them() {
    assert_eq!(kev_render(&ordered_json(r#""Romeo""#)), "Romeo");
    assert_eq!(kev_render(&ordered_json("true")), "True");
    assert_eq!(kev_render(&ordered_json("false")), "False");
    assert_eq!(kev_render(&ordered_json("42")), "42");
    assert_eq!(kev_render(&ordered_json("1.0")), "1.0");
    assert_eq!(kev_render(&ordered_json("1e20")), "1e+20");
    assert_eq!(kev_render(&ordered_json("null")), "");
}

#[test]
fn should_render_an_object_as_labelled_lines_with_nested_values_indented() {
    let rendered_text = kev_render(&ordered_json(
        r#"{"speaker": "Juliet", "act": 2, "lines": ["O Romeo", "wherefore art thou"], "scene": {"place": "orchard", "night": true}}"#,
    ));

    assert_eq!(
        rendered_text,
        "speaker: Juliet\nact: 2\nlines:\n  - O Romeo\n  - wherefore art thou\nscene:\n  place: orchard\n  night: True"
    );
}

#[test]
fn should_render_a_list_of_objects_with_the_first_line_beside_its_dash() {
    let rendered_text = kev_render(&ordered_json(
        r#"[{"name": "Romeo", "house": "Montague"}, "Tybalt"]"#,
    ));

    assert_eq!(rendered_text, "- name: Romeo\n  house: Montague\n- Tybalt");
}

#[test]
fn should_render_empty_containers_as_nothing() {
    assert_eq!(kev_render(&ordered_json("[]")), "");
    assert_eq!(kev_render(&ordered_json("{}")), "");
    assert_eq!(kev_render(&ordered_json(r#"{"cast": []}"#)), "cast:\n");
}

#[test]
fn should_read_a_noul_question_as_no_and_yes_with_their_descriptions() {
    let question = single_question(
        r#"{"love": {"type": "noul", "instructions": "Is Romeo in love?", "criteria": {"true": "he says so", "false": ""}}}"#,
    );

    let question_text = kev_question_text(&question);

    assert_eq!(question_text.instructions_text, "Is Romeo in love?");
    assert_eq!(question_text.option_texts, ["no", "yes: he says so"]);
}

#[test]
fn should_read_choice_options_in_order_with_rendered_descriptions() {
    let question = single_question(
        r#"{"house": {"type": "choice", "instructions": {"ask": "Which house?"}, "criteria": {"montague": "Romeo's family", "capulet": {"head": "Lord Capulet"}, "neither": null}}}"#,
    );

    let question_text = kev_question_text(&question);

    assert_eq!(question_text.instructions_text, "ask: Which house?");
    assert_eq!(
        question_text.option_texts,
        [
            "montague: Romeo's family",
            "capulet: head: Lord Capulet",
            "neither"
        ]
    );
}

#[test]
fn should_read_score_levels_as_their_rendered_text() {
    let question = single_question(
        r#"{"grief": {"type": "score", "criteria": ["calm", "troubled", {"level": "distraught"}]}}"#,
    );

    let question_text = kev_question_text(&question);

    assert_eq!(question_text.instructions_text, "");
    assert_eq!(
        question_text.option_texts,
        ["calm", "troubled", "level: distraught"]
    );
}

#[test]
fn should_escape_only_well_formed_control_token_names() {
    assert_eq!(
        kev_escape_control_tokens("before <|im_start|> and <|fim_prefix|> after"),
        "before <¦im_start¦> and <¦fim_prefix¦> after"
    );
    assert_eq!(
        kev_escape_control_tokens("<||> <|a b|> <|x"),
        "<||> <|a b|> <|x"
    );
    assert_eq!(kev_escape_control_tokens("<|<|name|>"), "<|<¦name¦>");
    assert_eq!(kev_escape_control_tokens(""), "");
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
