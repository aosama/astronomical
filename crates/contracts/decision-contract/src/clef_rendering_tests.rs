//! Unit contracts for Clef's rendering and option order, with the expected
//! text worked out from the reference's `render`, `question_options` and
//! `encode_record` using Python's own `json.dumps`.

use super::{clef_question_text, clef_render};
use crate::test_support::{ordered_json, single_question};

#[test]
fn should_write_a_string_as_it_is_and_anything_else_as_compact_sorted_json() {
    assert_eq!(clef_render(&ordered_json(r#""O Romeo""#)), "O Romeo");
    assert_eq!(
        clef_render(&ordered_json(
            r#"{"speaker": "Juliet", "act": 2, "night": true, "w": 1e20, "é": null}"#
        )),
        r#"{"act":2,"night":true,"speaker":"Juliet","w":1e+20,"é":null}"#
    );
    assert_eq!(clef_render(&ordered_json("[]")), "[]");
}

#[test]
fn should_score_choice_options_in_key_order_and_remember_the_callers_order() {
    let question = single_question(
        r#"{"house": {"type": "choice", "instructions": "Whose house?", "criteria": {"montague": "a house", "capulet": null}}}"#,
    );

    let question_text = clef_question_text(&question);

    assert_eq!(question_text.type_id, 1);
    assert_eq!(question_text.instructions_text, "Whose house?");
    let option_ids: Vec<&str> = question_text
        .options
        .iter()
        .map(|option| option.option_id.as_str())
        .collect();
    assert_eq!(option_ids, ["capulet", "montague"]);
    assert_eq!(
        question_text.options[0].semantics_text,
        r#"{"option_id":"capulet"}"#
    );
    assert_eq!(
        question_text.options[1].semantics_text,
        r#"{"description":"a house","option_id":"montague"}"#
    );
    assert_eq!(question_text.options[0].canonical_index, 1);
    assert_eq!(question_text.options[1].canonical_index, 0);
}

#[test]
fn should_read_score_levels_as_numbered_options_with_their_json_description() {
    let question = single_question(
        r#"{"grief": {"type": "score", "criteria": ["calm", {"level": "distraught", "b": 1.5}]}}"#,
    );

    let question_text = clef_question_text(&question);

    assert_eq!(question_text.type_id, 2);
    assert_eq!(
        question_text.options[0].semantics_text,
        r#"{"description":"calm","option_id":"0"}"#
    );
    assert_eq!(
        question_text.options[1].semantics_text,
        r#"{"description":{"b":1.5,"level":"distraught"},"option_id":"1"}"#
    );
    assert_eq!(question_text.options[1].canonical_index, 1);
}

#[test]
fn should_read_a_noul_as_true_then_false_with_default_descriptions() {
    let question_text = clef_question_text(&single_question(r#"{"dead": {"type": "noul"}}"#));

    assert_eq!(question_text.type_id, 0);
    assert_eq!(
        question_text.options[0].semantics_text,
        r#"{"description":"The proposition is true or the answer is yes.","option_id":"true"}"#
    );
    assert_eq!(
        question_text.options[1].semantics_text,
        r#"{"description":"The proposition is false or the answer is no.","option_id":"false"}"#
    );
    assert_eq!(question_text.options[0].canonical_index, 1);
    assert_eq!(question_text.options[1].canonical_index, 0);
}

#[test]
fn should_replace_only_the_noul_descriptions_the_caller_gives() {
    let question_text = clef_question_text(&single_question(
        r#"{"dead": {"type": "noul", "criteria": {"true": "someone dies"}}}"#,
    ));

    assert_eq!(
        question_text.options[0].semantics_text,
        r#"{"description":"someone dies","option_id":"true"}"#
    );
    assert!(
        question_text.options[1]
            .semantics_text
            .contains("The proposition is false")
    );
}

#[test]
fn should_use_the_question_name_when_the_instructions_are_missing() {
    // Empty instructions never reach a renderer: the shared contract refuses them.
    let missing = clef_question_text(&single_question(r#"{"mentions_death": {"type": "noul"}}"#));
    let structured = clef_question_text(&single_question(
        r#"{"q": {"type": "noul", "instructions": {"ask": "Is it night?", "act": 2}}}"#,
    ));

    assert_eq!(missing.instructions_text, "mentions_death");
    assert_eq!(
        structured.instructions_text,
        r#"{"act":2,"ask":"Is it night?"}"#
    );
}
