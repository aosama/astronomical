//! Library catalog REST contract coverage.

use std::time::Duration;

use astronomical_daemon::DownloadCatalog;
use axum::{
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode, header},
};
use tokio::time::timeout;
use tower::ServiceExt;

use crate::common::ScriptedExecutor;

const VALID_CATALOG_JSON: &str = r#"{
    "schema_version": 2,
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
            "huggingface_id": "astronomical-test/example-k2",
            "revision": "89abcdef0123456789abcdef0123456789abcdef",
            "display_name": "Example K2",
            "family": "k2_horizon_mova",
            "approximate_size_bytes": 5000000000,
            "public": true
        },
        {
            "huggingface_id": "astronomical-test/example-embedder",
            "revision": "fedcba9876543210fedcba9876543210fedcba98",
            "display_name": "Example Embedder",
            "family": "modernbert",
            "approximate_size_bytes": 200000000,
            "public": true,
            "capabilities": {"supports_embeddings": true}
        }
    ]
}"#;

#[tokio::test]
async fn should_return_the_validated_catalog_in_authored_order_when_the_worker_is_unavailable() {
    timeout(Duration::from_secs(5), async {
        let download_catalog = DownloadCatalog::parse_json(VALID_CATALOG_JSON)
            .expect("the fictional REST catalog should parse");
        let application = astronomical_daemon::build_application_with_download_catalog(
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
                "schema_version": 2,
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
                            "supports_image_generation": false,
                            "supports_embeddings": false
                        }
                    },
                    {
                        "huggingface_id": "astronomical-test/example-k2",
                        "revision": "89abcdef0123456789abcdef0123456789abcdef",
                        "display_name": "Example K2",
                        "family": "k2_horizon_mova",
                        "approximate_size_bytes": 5_000_000_000_u64,
                        "public": true,
                        "ready_on_this_mac": false,
                        "download_state": null,
                        "capabilities": {
                            "supports_reasoning": false,
                            "supports_vision": false,
                            "supports_tool_calls": false,
                            "supports_image_generation": false,
                            "supports_embeddings": false
                        }
                    },
                    {
                        "huggingface_id": "astronomical-test/example-embedder",
                        "revision": "fedcba9876543210fedcba9876543210fedcba98",
                        "display_name": "Example Embedder",
                        "family": "modernbert",
                        "approximate_size_bytes": 200_000_000_u64,
                        "public": true,
                        "ready_on_this_mac": false,
                        "download_state": null,
                        "capabilities": {
                            "supports_reasoning": false,
                            "supports_vision": false,
                            "supports_tool_calls": false,
                            "supports_image_generation": false,
                            "supports_embeddings": true
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
            let application = astronomical_daemon::build_application_with_download_catalog(
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

// The /library page renders one card per catalog entry and degrades the WHOLE
// page to "unavailable" when even one entry misses a field its renderer
// requires, so a shipped entry that parses in Rust but cannot render would
// take every model's download surface down with it. This journey walks the
// real shipped registry file through every seam a user-visible entry crosses —
// bundled parse, endpoint projection, and the browser renderability contract —
// so a hand-edited catalog entry cannot silently vanish from the Library.
#[tokio::test]
async fn should_publish_every_shipped_catalog_entry_to_the_library_catalog_endpoint() {
    timeout(Duration::from_secs(5), async {
        let shipped_catalog_text = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../registry/download_catalog.json"),
        )
        .expect("the shipped download catalog should be readable");
        let shipped_catalog: serde_json::Value = serde_json::from_str(&shipped_catalog_text)
            .expect("the shipped download catalog should be valid JSON");
        let shipped_entry_ids: Vec<&str> = shipped_catalog["entries"]
            .as_array()
            .expect("the shipped catalog should contain an entries array")
            .iter()
            .map(|shipped_entry| {
                shipped_entry["huggingface_id"]
                    .as_str()
                    .expect("every shipped entry should declare a string huggingface_id")
            })
            .collect();
        assert!(
            !shipped_entry_ids.is_empty(),
            "the shipped catalog should declare at least one entry"
        );

        let download_catalog =
            DownloadCatalog::load_bundled().expect("the shipped catalog should parse and validate");
        let application = astronomical_daemon::build_application_with_download_catalog(
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
            .expect("the catalog response should be available");

        assert_eq!(catalog_response.status(), StatusCode::OK);
        let catalog_body = to_bytes(catalog_response.into_body(), 256 * 1024)
            .await
            .expect("the catalog response body should be readable");
        let catalog_document: serde_json::Value =
            serde_json::from_slice(&catalog_body).expect("the catalog response should contain JSON");

        let published_entries = catalog_document["entries"]
            .as_array()
            .expect("the catalog response should contain an entries array");
        assert_eq!(
            published_entries.len(),
            shipped_entry_ids.len(),
            "every shipped entry must be published; none may be dropped or duplicated"
        );
        for (published_entry_index, (published_entry, shipped_entry_id)) in
            published_entries.iter().zip(&shipped_entry_ids).enumerate()
        {
            assert_eq!(
                published_entry["huggingface_id"].as_str(),
                Some(*shipped_entry_id),
                "published entry {published_entry_index} must be the shipped entry in authored order"
            );
        }

        // The browser contract (`isRenderableLibraryCatalogEntry`): these are
        // exactly the fields every card needs, so each published entry must
        // carry them with renderable values.
        for (published_entry_index, published_entry) in published_entries.iter().enumerate() {
            let huggingface_id = published_entry["huggingface_id"]
                .as_str()
                .unwrap_or_else(|| panic!("published entry {published_entry_index} must carry a string huggingface_id"));
            assert!(
                !huggingface_id.trim().is_empty(),
                "published entry {published_entry_index} must carry a non-empty huggingface_id"
            );
            let revision = published_entry["revision"]
                .as_str()
                .unwrap_or_else(|| panic!("published entry {published_entry_index} must carry a string revision"));
            assert_eq!(
                revision.len(),
                40,
                "published entry {published_entry_index} revision must be 40 characters"
            );
            assert!(
                revision
                    .chars()
                    .all(|revision_character| revision_character.is_ascii_hexdigit()),
                "published entry {published_entry_index} revision must be hexadecimal"
            );
            let display_name = published_entry["display_name"]
                .as_str()
                .unwrap_or_else(|| panic!("published entry {published_entry_index} must carry a string display_name"));
            assert!(
                !display_name.trim().is_empty(),
                "published entry {published_entry_index} must carry a non-empty display_name"
            );
            let family = published_entry["family"]
                .as_str()
                .unwrap_or_else(|| panic!("published entry {published_entry_index} must carry a string family"));
            assert!(
                !family.trim().is_empty(),
                "published entry {published_entry_index} must carry a non-empty family"
            );
            let approximate_size_bytes = published_entry["approximate_size_bytes"]
                .as_u64()
                .unwrap_or_else(|| {
                    panic!("published entry {published_entry_index} must carry an integer approximate_size_bytes")
                });
            assert!(
                approximate_size_bytes > 0
                    && approximate_size_bytes <= 9_007_199_254_740_991,
                "published entry {published_entry_index} size must be a positive JavaScript-safe integer"
            );
            assert_eq!(
                published_entry["public"], serde_json::json!(true),
                "published entry {published_entry_index} must be public"
            );
        }
    })
    .await
    .expect("the bundled catalog publication journey must finish within five seconds");
}
