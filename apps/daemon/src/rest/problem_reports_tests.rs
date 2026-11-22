//! Unit tests for reading the person's note from a Send report request.

use super::problem_reports::typed_note;

#[test]
fn should_read_the_note_from_a_json_body() {
    assert_eq!(
        typed_note(br#"{"note":"the window froze"}"#),
        "the window froze"
    );
}

#[test]
fn should_read_no_note_from_an_empty_unreadable_or_noteless_body() {
    for request_body in [
        &b""[..],
        b"not json",
        br#"{"comment":"wrong field"}"#,
        br#"{"note":42}"#,
        br#"["note"]"#,
    ] {
        assert_eq!(typed_note(request_body), "", "{request_body:?}");
    }
}
