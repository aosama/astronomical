//! Library catalog REST contract coverage.

use std::time::Duration;

use astronomical_supervisor::{DownloadCatalog, build_application_with_download_catalog};
use axum::{
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode, header},
};
use tokio::time::timeout;
use tower::ServiceExt;

use crate::common::ScriptedExecutor;

const VALID_CATALOG_JSON: &str = r#"{
    "schema_version": 1,
    "entries": [
        {
            "huggingface_id": "astronomical-test/example-qwen",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "display_name": "Example Qwen",
            "family": "qwen3_5",
            "approximate_size_bytes": 4000000000,
            "public": true
        },
        {
            "huggingface_id": "astronomical-test/example-laguna",
            "revision": "89abcdef0123456789abcdef0123456789abcdef",
            "display_name": "Example Laguna",
            "family": "laguna",
            "approximate_size_bytes": 5000000000,
            "public": true
        }
    ]
}"#;

#[tokio::test]
async fn should_return_the_validated_catalog_in_authored_order_when_the_worker_is_unavailable() {
    timeout(Duration::from_secs(5), async {
        let download_catalog = DownloadCatalog::parse_json(VALID_CATALOG_JSON)
            .expect("the fictional REST catalog should parse");
        let application = build_application_with_download_catalog(
            ScriptedExecutor::unavailable(),
            download_catalog,
        );

        let catalog_response = application
            .oneshot(
                Request::builder()
                    .uri("/v1/library/catalog")
                    .body(Body::empty())
                    .expect("the catalog request should be valid"),
            )
            .await
            .expect("the application should return a catalog response");

        assert_eq!(catalog_response.status(), StatusCode::OK);
        assert!(
            catalog_response
                .headers()
                .get(header::CONTENT_TYPE)
                .is_some_and(|content_type| content_type == "application/json")
        );
        let catalog_body = to_bytes(catalog_response.into_body(), 16 * 1024)
            .await
            .expect("the catalog response body should be readable");
        let catalog_document: serde_json::Value = serde_json::from_slice(&catalog_body)
            .expect("the catalog response should contain JSON");

        assert_eq!(
            catalog_document,
            serde_json::json!({
                "schema_version": 1,
                "entries": [
                    {
                        "huggingface_id": "astronomical-test/example-qwen",
                        "revision": "0123456789abcdef0123456789abcdef01234567",
                        "display_name": "Example Qwen",
                        "family": "qwen3_5",
                        "approximate_size_bytes": 4_000_000_000_u64,
                        "public": true,
                        "ready_on_this_mac": false,
                        "download_state": null,
                        "capabilities": {
                            "supports_reasoning": false,
                            "supports_vision": false,
                            "supports_tool_calls": false,
                            "supports_image_generation": false
                        }
                    },
                    {
                        "huggingface_id": "astronomical-test/example-laguna",
                        "revision": "89abcdef0123456789abcdef0123456789abcdef",
                        "display_name": "Example Laguna",
                        "family": "laguna",
                        "approximate_size_bytes": 5_000_000_000_u64,
                        "public": true,
                        "ready_on_this_mac": false,
                        "download_state": null,
                        "capabilities": {
                            "supports_reasoning": false,
                            "supports_vision": false,
                            "supports_tool_calls": false,
                            "supports_image_generation": false
                        }
                    }
                ]
            })
        );
    })
    .await
    .expect("the catalog response journey must finish within five seconds");
}

#[tokio::test]
async fn should_reject_catalog_mutation_and_leave_unknown_library_paths_unmatched() {
    timeout(Duration::from_secs(5), async {
        let download_catalog = DownloadCatalog::parse_json(VALID_CATALOG_JSON)
            .expect("the fictional REST catalog should parse");
        for (method, path, expected_status) in [
            (
                Method::POST,
                "/v1/library/catalog",
                StatusCode::METHOD_NOT_ALLOWED,
            ),
            (Method::GET, "/v1/library/unknown", StatusCode::NOT_FOUND),
        ] {
            let application = build_application_with_download_catalog(
                ScriptedExecutor::ready(Vec::new()),
                download_catalog.clone(),
            );
            let response = application
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .body(Body::empty())
                        .expect("the Library request should be valid"),
                )
                .await
                .expect("the application should return an HTTP response");

            assert_eq!(response.status(), expected_status, "path: {path}");
        }
    })
    .await
    .expect("the immutable catalog routing journey must finish within five seconds");
}
