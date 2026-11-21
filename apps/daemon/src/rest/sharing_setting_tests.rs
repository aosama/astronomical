//! Unit tests for the requests `PUT /v1/config/sharing` accepts.

use super::sharing_setting::SharingSettingRequest;

#[test]
fn should_name_each_choice_a_request_carries_in_its_stored_form() {
    let request: SharingSettingRequest =
        serde_json::from_str(r#"{"problem_reports":"always","daily_count":false}"#)
            .expect("request parses");
    assert_eq!(
        request.requested_values(),
        vec![
            ("problem_reports", serde_json::json!("always")),
            ("daily_count", serde_json::json!(false)),
        ]
    );
}

#[test]
fn should_name_nothing_for_an_empty_request() {
    let empty_request: SharingSettingRequest =
        serde_json::from_str("{}").expect("an empty request parses");
    assert!(empty_request.requested_values().is_empty());
}

#[test]
fn should_refuse_an_unknown_choice_or_field() {
    for request_text in [
        r#"{"problem_reports":"sometimes"}"#,
        r#"{"daily_count":"yes"}"#,
        r#"{"crash_reports":true}"#,
    ] {
        assert!(
            serde_json::from_str::<SharingSettingRequest>(request_text).is_err(),
            "{request_text}"
        );
    }
}
