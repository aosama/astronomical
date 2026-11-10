//! Library Show in Finder REST contract: every refusal path, none of which
//! reaches Finder, so the suite never opens a window.

use std::time::Duration;

use astronomical_daemon::DownloadCatalog;
use axum::{
    body::Body,
    http::{Method, Request, StatusCode, header},
};
use tokio::time::timeout;
use tower::ServiceExt;

use crate::support::ScriptedExecutor;

const NOT_DOWNLOADED_CATALOG_JSON: &str = r#"{
    "schema_version": 4,
    "entries": [{
        "huggingface_id": "astronomical-test/example-qwen",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "display_name": "Example Qwen",
        "family": "qwen3_5",
        "approximate_size_bytes": 4000000000,
        "approximate_memory_bytes": 5000000000,
        "role": "chat",
        "public": true
    }]
}"#;

async fn reveal_status(method: Method, request_body: &str) -> StatusCode {
    let download_catalog = DownloadCatalog::parse_json(NOT_DOWNLOADED_CATALOG_JSON)
        .expect("the fictional reveal catalog should parse");
    let application = astronomical_daemon::build_application_with_download_catalog(
        ScriptedExecutor::ready(Vec::new()),
        download_catalog,
    );
    application
        .oneshot(
            Request::builder()
                .method(method)
                .uri("/v1/library/reveal")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request_body.to_owned()))
                .expect("the reveal request should be valid"),
        )
        .await
        .expect("the application should return an HTTP response")
        .status()
}

#[tokio::test]
async fn should_refuse_to_reveal_a_model_that_is_not_in_the_catalog() {
    timeout(Duration::from_secs(5), async {
        let unknown_identity = r#"{"huggingface_id":"astronomical-test/not-in-catalog"}"#;
        assert_eq!(
            reveal_status(Method::POST, unknown_identity).await,
            StatusCode::NOT_FOUND
        );
    })
    .await
    .expect("the unknown-model reveal must finish within five seconds");
}

#[tokio::test]
async fn should_refuse_to_reveal_a_catalog_model_that_is_not_on_this_mac() {
    timeout(Duration::from_secs(5), async {
        let catalog_identity = r#"{"huggingface_id":"astronomical-test/example-qwen"}"#;
        assert_eq!(
            reveal_status(Method::POST, catalog_identity).await,
            StatusCode::CONFLICT
        );
    })
    .await
    .expect("the not-downloaded reveal must finish within five seconds");
}

#[tokio::test]
async fn should_reject_a_request_that_names_a_path_or_has_no_identity() {
    timeout(Duration::from_secs(5), async {
        for malformed_body in [
            r#"{"path":"/Applications"}"#,
            r#"{"huggingface_id":"astronomical-test/example-qwen","path":"/Applications"}"#,
            "{}",
            "not json",
        ] {
            assert!(
                reveal_status(Method::POST, malformed_body)
                    .await
                    .is_client_error(),
                "body {malformed_body} must be refused before any lookup"
            );
        }
        assert_eq!(
            reveal_status(Method::GET, "").await,
            StatusCode::METHOD_NOT_ALLOWED
        );
    })
    .await
    .expect("the malformed reveal requests must finish within five seconds");
}
