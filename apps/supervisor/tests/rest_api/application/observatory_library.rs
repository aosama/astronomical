//! REST acceptance coverage for the embedded Observatory Library destination and script.

use std::time::Duration;

use astronomical_supervisor::build_application;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use tokio::time::timeout;
use tower::ServiceExt;

use crate::common::ScriptedExecutor;

#[tokio::test]
async fn should_serve_a_labelled_library_destination_at_its_deep_link() {
    timeout(Duration::from_secs(5), async {
        let application = build_application(ScriptedExecutor::ready(Vec::new()));
        let response = application
            .oneshot(
                Request::builder()
                    .uri("/library")
                    .body(Body::empty())
                    .expect("the Library deep-link request should be valid"),
            )
            .await
            .expect("the application should return the Observatory shell");

        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .is_some_and(|content_type| content_type
                    .to_str()
                    .is_ok_and(|content_type| content_type.starts_with("text/html")))
        );
        let response_body = to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("the Observatory shell should be readable");
        let shell_text = String::from_utf8(response_body.to_vec())
            .expect("the Observatory shell should contain UTF-8");

        let library_navigation_tag =
            opening_tag_with_attribute(&shell_text, "data-observatory-destination=\"library\"");
        assert!(library_navigation_tag.contains("data-observatory-destination=\"library\""));
        assert!(library_navigation_tag.contains("aria-controls=\"library-view\""));
        let library_view_tag = opening_tag_with_id(&shell_text, "library-view");
        assert!(library_view_tag.contains("data-observatory-view=\"library\""));
        assert!(library_view_tag.contains("aria-labelledby=\"library-title\""));
        let library_catalog_tag = opening_tag_with_id(&shell_text, "library-catalog");
        assert!(!library_catalog_tag.contains("role=\"status\""));
        assert!(!library_catalog_tag.contains("aria-live="));
        let library_status_tag = opening_tag_with_id(&shell_text, "library-catalog-status");
        assert!(library_status_tag.contains("role=\"status\""));
        assert!(library_status_tag.contains("aria-live=\"polite\""));
    })
    .await
    .expect("the Library deep-link journey must finish within five seconds");
}

fn opening_tag_with_id<'a>(html_document: &'a str, element_id: &str) -> &'a str {
    opening_tag_with_attribute(html_document, &format!("id=\"{element_id}\""))
}

fn opening_tag_with_attribute<'a>(html_document: &'a str, attribute: &str) -> &'a str {
    let attribute_offset = html_document
        .find(attribute)
        .expect("the expected element attribute should be present");
    let tag_start = html_document[..attribute_offset]
        .rfind('<')
        .expect("the expected element should have an opening tag");
    let tag_end = html_document[attribute_offset..]
        .find('>')
        .map(|relative_tag_end| attribute_offset + relative_tag_end + 1)
        .expect("the expected element opening tag should end");
    &html_document[tag_start..tag_end]
}

#[tokio::test]
async fn should_serve_the_embedded_library_javascript() {
    timeout(Duration::from_secs(5), async {
        let application = build_application(ScriptedExecutor::ready(Vec::new()));
        let response = application
            .oneshot(
                Request::builder()
                    .uri("/library.js")
                    .body(Body::empty())
                    .expect("the library.js request should be valid"),
            )
            .await
            .expect("the application should return the Library script");

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
            .expect("the Library script should be readable");
        assert!(!response_body.is_empty());
    })
    .await
    .expect("the Library script journey must finish within five seconds");
}
