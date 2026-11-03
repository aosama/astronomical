//! REST coverage for the Thin Talk canvas CORS policy: which origins a
//! browser may call the daemon from. Preflight requests go through the full
//! router, so the policy is checked as the tower-http layer actually serves
//! it rather than as the predicate reads.

use axum::{
    body::Body,
    http::{Method, Request, Response, header},
};
use tower::ServiceExt;

use crate::support::ScriptedExecutor;

const CANVAS_SHELL_ORIGIN: &str = "thintalk-asset://shell";

/// Sends a browser-style preflight for a POST from `origin`.
async fn preflight_from(origin: &str) -> Response<Body> {
    let application = astronomical_daemon::build_application(ScriptedExecutor::ready(Vec::new()));
    application
        .oneshot(
            Request::builder()
                .method(Method::OPTIONS)
                .uri("/v1/models")
                .header(header::ORIGIN, origin)
                .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
                .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "content-type")
                .body(Body::empty())
                .expect("the preflight request should be valid"),
        )
        .await
        .expect("the application should answer the preflight")
}

fn allowed_origin_of(response: &Response<Body>) -> Option<String> {
    response
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .and_then(header_text)
}

fn header_text(header_value: &header::HeaderValue) -> Option<String> {
    header_value.to_str().ok().map(str::to_owned)
}

#[tokio::test]
async fn should_allow_the_thin_talk_canvas_origin_with_its_methods_and_cache_age() {
    let response = preflight_from(CANVAS_SHELL_ORIGIN).await;

    assert_eq!(
        allowed_origin_of(&response).as_deref(),
        Some(CANVAS_SHELL_ORIGIN)
    );
    let allowed_methods = response
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_METHODS)
        .and_then(header_text)
        .expect("an allowed preflight should list its methods");
    for expected_method in ["GET", "POST", "DELETE", "OPTIONS"] {
        assert!(
            allowed_methods.contains(expected_method),
            "{expected_method} missing from {allowed_methods}"
        );
    }
    assert_eq!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_MAX_AGE)
            .and_then(header_text)
            .as_deref(),
        Some("3600")
    );
}

#[tokio::test]
async fn should_allow_a_loopback_origin_on_any_port() {
    for loopback_origin in ["http://127.0.0.1:5173", "http://localhost:6733"] {
        let response = preflight_from(loopback_origin).await;

        assert_eq!(
            allowed_origin_of(&response).as_deref(),
            Some(loopback_origin)
        );
    }
}

#[tokio::test]
async fn should_not_allow_a_foreign_origin() {
    let response = preflight_from("https://example.com").await;

    assert_eq!(allowed_origin_of(&response), None);
}

/// A loopback prefix match without the port separator would admit any host
/// that merely starts with "localhost" or "127.0.0.1".
#[tokio::test]
async fn should_not_allow_a_lookalike_of_a_loopback_host() {
    for lookalike_origin in [
        "http://localhost.evil.example",
        "http://127.0.0.1.evil.example",
        "https://localhost:6733",
    ] {
        let response = preflight_from(lookalike_origin).await;

        assert_eq!(
            allowed_origin_of(&response),
            None,
            "{lookalike_origin} must not be allowed"
        );
    }
}
