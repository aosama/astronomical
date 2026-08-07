use astronomical_supervisor::build_application;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use tower::ServiceExt;

use crate::common::ScriptedExecutor;

const EXPECTED_HTML_TITLE_MARKER: &str = "<title>Astronomical Observatory</title>";
const EXPECTED_CHAT_TRANSCRIPT_MARKER: &str = "<section id=\"chat-transcript\" class=\"chat-transcript\" aria-live=\"polite\" aria-label=\"Chat transcript\"></section>";
const EXPECTED_NAVIGATION_LABEL_MARKER: &str = "aria-label=\"Observatory sections\"";
const EXPECTED_OVERVIEW_REGION_MARKER: &str =
    "data-observatory-view=\"overview\" aria-labelledby=\"overview-title\"";
const EXPECTED_OPTIMIZER_DESTINATION_MARKER: &str = "data-observatory-destination=\"optimizer\"";
const EXPECTED_OPTIMIZER_REGION_MARKER: &str =
    "data-observatory-view=\"optimizer\" aria-labelledby=\"optimizer-title\"";

#[tokio::test]
async fn should_serve_the_embedded_observatory_index_html_at_root() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/")
                .body(Body::empty())
                .expect("the root request should be valid"),
        )
        .await
        .expect("the application should return the observatory shell");

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .expect("the observatory shell should declare a content-type");
    assert!(
        content_type
            .to_str()
            .expect("the content-type should be valid ASCII")
            .starts_with("text/html"),
        "the observatory shell should be HTML"
    );
    let response_body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("the observatory shell body should be readable");
    let shell_text = String::from_utf8(response_body.to_vec())
        .expect("the observatory shell body should contain UTF-8");
    assert!(
        shell_text.contains(EXPECTED_HTML_TITLE_MARKER),
        "the observatory shell should declare its title"
    );
}

#[tokio::test]
async fn should_serve_the_observatory_shell_at_each_named_deep_link() {
    for observatory_path in [
        "/overview",
        "/chat",
        "/memory",
        "/cache",
        "/optimizer",
        "/model",
        "/settings",
    ] {
        let application = build_application(ScriptedExecutor::ready(Vec::new()));
        let response = application
            .oneshot(
                Request::builder()
                    .uri(observatory_path)
                    .body(Body::empty())
                    .expect("the Observatory deep-link request should be valid"),
            )
            .await
            .expect("the application should return the Observatory shell");

        assert_eq!(
            response.status(),
            StatusCode::OK,
            "path: {observatory_path}"
        );
        assert!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .is_some_and(|content_type| content_type
                    .to_str()
                    .is_ok_and(|content_type| content_type.starts_with("text/html"))),
            "deep-link response should be HTML for {observatory_path}"
        );
    }
}

#[tokio::test]
async fn should_return_not_found_for_an_unknown_rest_path() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/unknown-observatory-review-probe")
                .body(Body::empty())
                .expect("the unknown REST request should be valid"),
        )
        .await
        .expect("the application should return an HTTP response");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn should_expose_the_chat_transcript_as_a_named_live_region() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/")
                .body(Body::empty())
                .expect("the root request should be valid"),
        )
        .await
        .expect("the application should return the observatory shell");
    let response_body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("the observatory shell body should be readable");
    let shell_text = String::from_utf8(response_body.to_vec())
        .expect("the observatory shell body should contain UTF-8");

    assert!(
        shell_text.contains(EXPECTED_CHAT_TRANSCRIPT_MARKER),
        "the chat transcript must have a valid named live-region element"
    );
}

#[tokio::test]
async fn should_expose_named_observatory_navigation_and_overview_region() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/")
                .body(Body::empty())
                .expect("the root request should be valid"),
        )
        .await
        .expect("the application should return the observatory shell");
    let response_body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("the observatory shell body should be readable");
    let shell_text = String::from_utf8(response_body.to_vec())
        .expect("the observatory shell body should contain UTF-8");

    assert!(
        shell_text.contains(EXPECTED_NAVIGATION_LABEL_MARKER),
        "the Observatory must expose a named navigation landmark"
    );
    assert!(
        shell_text.contains(EXPECTED_OVERVIEW_REGION_MARKER),
        "the Observatory must expose a labelled Overview region"
    );
    assert!(
        shell_text.contains(EXPECTED_OPTIMIZER_DESTINATION_MARKER),
        "the Observatory must expose a direct Optimizer destination"
    );
    assert!(
        shell_text.contains(EXPECTED_OPTIMIZER_REGION_MARKER),
        "the Observatory must expose a labelled Optimizer region"
    );
}

#[tokio::test]
async fn should_serve_the_embedded_observatory_javascript_with_correct_content_type() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/console.js")
                .body(Body::empty())
                .expect("the console.js request should be valid"),
        )
        .await
        .expect("the application should return the console script");

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .expect("the console script should declare a content-type");
    assert!(
        content_type
            .to_str()
            .expect("the content-type should be valid ASCII")
            .starts_with("application/javascript"),
        "the console script should be JavaScript"
    );
    let response_body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("the console script body should be readable");
    assert!(
        !response_body.is_empty(),
        "the console script should not be empty"
    );
}

#[tokio::test]
async fn should_serve_the_embedded_memory_control_script() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/memory-control.js")
                .body(Body::empty())
                .expect("the memory-control request should be valid"),
        )
        .await
        .expect("the application should return the memory-control script");

    assert_eq!(response.status(), StatusCode::OK);
    let response_body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("the memory-control script body should be readable");
    assert!(
        String::from_utf8(response_body.to_vec())
            .expect("the memory-control script should be UTF-8")
            .contains("maximum-mlx-memory"),
        "the embedded memory-control script should contain its endpoint contract"
    );
}

#[tokio::test]
async fn should_serve_the_embedded_observatory_playground_javascript() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/playground.js")
                .body(Body::empty())
                .expect("the playground.js request should be valid"),
        )
        .await
        .expect("the application should return the playground script");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .is_some_and(|content_type| content_type
                .to_str()
                .is_ok_and(|content_type| content_type.starts_with("application/javascript")))
    );
    let response_body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("the playground script body should be readable");
    assert!(!response_body.is_empty());
}

#[tokio::test]
async fn should_serve_the_embedded_optimizer_javascript() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/optimizer.js")
                .body(Body::empty())
                .expect("the optimizer.js request should be valid"),
        )
        .await
        .expect("the application should return the optimizer script");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .is_some_and(|content_type| content_type
                .to_str()
                .is_ok_and(|content_type| content_type.starts_with("application/javascript")))
    );
    let response_body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("the optimizer script body should be readable");
    assert!(!response_body.is_empty());
}

#[tokio::test]
async fn should_serve_the_embedded_observatory_stylesheet_with_correct_content_type() {
    let application = build_application(ScriptedExecutor::ready(Vec::new()));
    let response = application
        .oneshot(
            Request::builder()
                .uri("/console.css")
                .body(Body::empty())
                .expect("the console.css request should be valid"),
        )
        .await
        .expect("the application should return the console stylesheet");

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .expect("the console stylesheet should declare a content-type");
    assert!(
        content_type
            .to_str()
            .expect("the content-type should be valid ASCII")
            .starts_with("text/css"),
        "the console stylesheet should be CSS"
    );
    let response_body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("the console stylesheet body should be readable");
    assert!(
        !response_body.is_empty(),
        "the console stylesheet should not be empty"
    );
}
