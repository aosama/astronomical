use astronomical_supervisor::{
    build_openai_chat_request_diagnostic_snapshot,
    build_openai_chat_request_info_diagnostic_snapshot,
};

#[test]
fn should_keep_only_non_payload_request_metadata_in_diagnostics() {
    let request_body = br#"{
        "model":"astronomical/fake-mixture-of-experts",
        "api_key":"body-api-secret",
        "messages":[{"role":"user","content":"inspect the failure\napi_key: user-api-secret\ncontinue safely"}],
        "stream":true
    }"#;

    let diagnostic_snapshot = build_openai_chat_request_diagnostic_snapshot(request_body);
    let diagnostic_debug_text = format!("{diagnostic_snapshot:?}");

    assert_eq!(diagnostic_snapshot.request_body_bytes, request_body.len());
    assert_eq!(diagnostic_snapshot.request_body_sha256.len(), 64);
    assert!(!diagnostic_debug_text.contains("inspect the failure"));
    assert!(!diagnostic_debug_text.contains("body-api-secret"));
    assert!(!diagnostic_debug_text.contains("user-api-secret"));
}

#[test]
fn should_summarize_the_latest_user_message_for_info_diagnostics() {
    let request_body = br#"{
        "model":"astronomical/fake-mixture-of-experts",
        "messages":[
            {"role":"system","content":"system"},
            {"role":"user","content":"earlier"},
            {"role":"assistant","content":"reply"},
            {"role":"user","content":"yeah i am still thinking"}
        ],
        "stream":true
    }"#;

    let diagnostic_snapshot = build_openai_chat_request_info_diagnostic_snapshot(request_body);

    assert_eq!(diagnostic_snapshot.message_count, Some(4));
    assert_eq!(
        diagnostic_snapshot.last_user_message_character_count,
        Some(24)
    );
    assert!(diagnostic_snapshot.last_user_message_sha256.is_some());
    assert!(!format!("{diagnostic_snapshot:?}").contains("yeah i am still thinking"));
}

#[test]
fn should_summarize_message_roles_for_translation_rejection_diagnostics() {
    let request_body = br#"{
        "model":"astronomical/fake-mixture-of-experts",
        "messages":[
            {"role":"user","content":"earlier context"},
            {"role":"system","content":"a chronological update"},
            {"role":"assistant","content":"reply"},
            {"role":"tool","tool_call_id":"call_1","content":"tool output"},
            {"role":"untrusted role text that must not appear in logs","content":"ignored"}
        ],
        "stream":true
    }"#;

    let diagnostic_snapshot = build_openai_chat_request_info_diagnostic_snapshot(request_body);

    assert_eq!(
        diagnostic_snapshot.message_role_sequence_preview.as_deref(),
        Some("user,system,assistant,tool,unknown")
    );
}

#[test]
fn should_not_retain_secret_bearing_user_message_text_in_info_diagnostics() {
    let request_body = br#"{
        "model":"astronomical/fake-mixture-of-experts",
        "messages":[
            {"role":"user","content":"please use token: definitely-secret\nnormal followup"}
        ],
        "stream":true
    }"#;

    let diagnostic_snapshot = build_openai_chat_request_info_diagnostic_snapshot(request_body);

    assert_eq!(
        diagnostic_snapshot.last_user_message_character_count,
        Some(51)
    );
    assert!(diagnostic_snapshot.last_user_message_sha256.is_some());
    assert!(!format!("{diagnostic_snapshot:?}").contains("definitely-secret"));
}
