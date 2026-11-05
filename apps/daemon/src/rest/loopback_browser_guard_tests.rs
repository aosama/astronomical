use axum::http::{HeaderMap, HeaderValue, StatusCode, header};

use super::{names_loopback, refusal_for_foreign_request, request_comes_from_this_instance};

#[test]
fn should_allow_this_instances_own_page() {
    assert!(request_comes_from_this_instance(
        Some("127.0.0.1:6733"),
        Some("http://127.0.0.1:6733")
    ));
    assert!(request_comes_from_this_instance(
        Some("localhost:6733"),
        Some("http://localhost:6733")
    ));
}

#[test]
fn should_allow_a_tool_that_sends_no_origin() {
    assert!(request_comes_from_this_instance(
        Some("127.0.0.1:6733"),
        None
    ));
    assert!(request_comes_from_this_instance(None, None));
}

#[test]
fn should_refuse_a_rebound_domain_even_though_it_looks_same_origin() {
    assert!(!request_comes_from_this_instance(
        Some("attacker.example:6733"),
        Some("http://attacker.example:6733")
    ));
    assert!(!request_comes_from_this_instance(
        Some("attacker.example:6733"),
        None
    ));
}

#[test]
fn should_refuse_another_local_web_server() {
    assert!(!request_comes_from_this_instance(
        Some("127.0.0.1:6733"),
        Some("http://localhost:3000")
    ));
    assert!(!request_comes_from_this_instance(
        Some("127.0.0.1:6733"),
        Some("http://127.0.0.1:3000")
    ));
}

#[test]
fn should_refuse_opaque_and_other_scheme_origins() {
    assert!(!request_comes_from_this_instance(
        Some("127.0.0.1:6733"),
        Some("null")
    ));
    assert!(!request_comes_from_this_instance(
        Some("127.0.0.1:6733"),
        Some("https://127.0.0.1:6733")
    ));
    assert!(!request_comes_from_this_instance(
        None,
        Some("http://127.0.0.1:6733")
    ));
}

#[test]
fn should_recognize_only_exact_loopback_names() {
    assert!(names_loopback("127.0.0.1"));
    assert!(names_loopback("localhost:1"));
    assert!(names_loopback("[::1]:6733"));
    assert!(!names_loopback("localhost.attacker.example"));
    assert!(!names_loopback("127.0.0.1.attacker.example:6733"));
    assert!(!names_loopback("localhost:"));
    assert!(!names_loopback("localhost:80a"));
    assert!(!names_loopback(""));
}

#[test]
fn should_answer_a_foreign_request_with_forbidden() {
    let mut headers = HeaderMap::new();
    headers.insert(header::HOST, HeaderValue::from_static("attacker.example"));

    let refusal = refusal_for_foreign_request(&headers).expect("the request is refused");

    assert_eq!(refusal.status(), StatusCode::FORBIDDEN);
}

#[test]
fn should_refuse_a_header_that_is_not_text() {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::ORIGIN,
        HeaderValue::from_bytes(&[0xff]).expect("opaque bytes are a valid header value"),
    );

    assert!(refusal_for_foreign_request(&headers).is_some());
}

#[test]
fn should_let_a_request_from_this_instance_proceed() {
    let mut headers = HeaderMap::new();
    headers.insert(header::HOST, HeaderValue::from_static("127.0.0.1:6733"));
    headers.insert(
        header::ORIGIN,
        HeaderValue::from_static("http://127.0.0.1:6733"),
    );

    assert!(refusal_for_foreign_request(&headers).is_none());
}
