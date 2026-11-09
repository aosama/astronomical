//! REST acceptance coverage for the embedded Observatory Download Models destination and script.

use std::time::Duration;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use tokio::time::timeout;
use tower::ServiceExt;

use crate::support::ScriptedExecutor;

#[tokio::test]
async fn should_serve_a_labelled_download_models_destination_at_its_deep_link() {
    timeout(Duration::from_secs(5), async {
        let application =
            astronomical_daemon::build_application(ScriptedExecutor::ready(Vec::new()));
        let response = application
            .oneshot(
                Request::builder()
                    .uri("/download-models")
                    .body(Body::empty())
                    .expect("the Download Models deep-link request should be valid"),
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

        let download_models_navigation_tag = opening_tag_with_attribute(
            &shell_text,
            "data-observatory-destination=\"download-models\"",
        );
        assert!(
            download_models_navigation_tag
                .contains("data-observatory-destination=\"download-models\"")
        );
        assert!(download_models_navigation_tag.contains("aria-controls=\"download-models-view\""));
        let download_models_view_tag = opening_tag_with_id(&shell_text, "download-models-view");
        assert!(download_models_view_tag.contains("data-observatory-view=\"download-models\""));
        assert!(download_models_view_tag.contains("aria-labelledby=\"download-models-title\""));
        let download_models_catalog_tag =
            opening_tag_with_id(&shell_text, "download-models-catalog");
        assert!(!download_models_catalog_tag.contains("role=\"status\""));
        assert!(!download_models_catalog_tag.contains("aria-live="));
        let download_models_status_tag =
            opening_tag_with_id(&shell_text, "download-models-catalog-status");
        assert!(download_models_status_tag.contains("role=\"status\""));
        assert!(download_models_status_tag.contains("aria-live=\"polite\""));
    })
    .await
    .expect("the Download Models deep-link journey must finish within five seconds");
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
async fn should_serve_the_embedded_download_models_javascript() {
    timeout(Duration::from_secs(5), async {
        let application =
            astronomical_daemon::build_application(ScriptedExecutor::ready(Vec::new()));
        let response = application
            .oneshot(
                Request::builder()
                    .uri("/download-models.js")
                    .body(Body::empty())
                    .expect("the download-models.js request should be valid"),
            )
            .await
            .expect("the application should return the Download Models script");

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
            .expect("the Download Models script should be readable");
        assert!(!response_body.is_empty());
    })
    .await
    .expect("the Download Models script journey must finish within five seconds");
}
