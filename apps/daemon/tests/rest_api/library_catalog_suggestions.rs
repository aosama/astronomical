//! The catalog response carries the suggestions for this Mac, worded by the
//! engine, once the worker has reported the memory it loads models within.

use std::time::Duration;

use astronomical_daemon::DownloadCatalog;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tokio::time::timeout;
use tower::ServiceExt;

use crate::support::ScriptedExecutor;

const GIGABYTE: u64 = 1_073_741_824;

fn catalog_entry(
    model_name: &str,
    revision_digit: char,
    extra_fields: &serde_json::Value,
) -> serde_json::Value {
    let mut entry_value = serde_json::json!({
        "huggingface_id": format!("astronomical-test/{model_name}"),
        "revision": revision_digit.to_string().repeat(40),
        "family": "qwen3_5",
        "approximate_size_bytes": 4 * GIGABYTE,
        "role": "chat",
        "public": true
    });
    for (field_name, field_value) in extra_fields.as_object().expect("an object") {
        entry_value[field_name] = field_value.clone();
    }
    entry_value
}

fn ladder_catalog() -> DownloadCatalog {
    let catalog_document = serde_json::json!({
        "schema_version": 5,
        "entries": [
            catalog_entry("Small", '1', &serde_json::json!({"get_started": true, "approximate_memory_bytes": 6 * GIGABYTE})),
            catalog_entry("Large", '2', &serde_json::json!({"role": "coding", "get_started": true, "approximate_memory_bytes": 23 * GIGABYTE})),
            catalog_entry("Large-6bit", '3', &serde_json::json!({"approximate_memory_bytes": 28 * GIGABYTE, "build_of": "Large", "build_difference": "more_detail"}))
        ]
    });
    DownloadCatalog::parse_json(&catalog_document.to_string())
        .expect("the ladder catalog should parse")
}

async fn catalog_suggestions_with_ceiling(mlx_memory_ceiling_bytes: u64) -> serde_json::Value {
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor.health_snapshot.mlx_memory_ceiling_bytes = mlx_memory_ceiling_bytes;
    let application = astronomical_daemon::build_application_with_download_catalog(
        scripted_executor,
        ladder_catalog(),
    );
    let catalog_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/library/catalog")
                .body(Body::empty())
                .expect("a valid request"),
        )
        .await
        .expect("the catalog response should be available");
    assert_eq!(catalog_response.status(), StatusCode::OK);
    let catalog_body = to_bytes(catalog_response.into_body(), 64 * 1024)
        .await
        .expect("a readable body");
    let catalog_document: serde_json::Value = serde_json::from_slice(&catalog_body).expect("JSON");
    catalog_document["suggestions"].clone()
}

#[tokio::test]
async fn should_carry_the_suggestions_for_this_macs_memory_with_their_words() {
    timeout(Duration::from_secs(5), async {
        let suggestions = catalog_suggestions_with_ceiling(36 * GIGABYTE + GIGABYTE / 2).await;

        assert_eq!(
            suggestions,
            serde_json::json!([
                {
                    "kind": "best_fit",
                    "huggingface_id": "astronomical-test/Large",
                    "model_id": "Large",
                    "tag": "Best fit for this Mac",
                    "reason": "The strongest suggested model that runs entirely in this Mac's memory."
                },
                {
                    "kind": "more_detail",
                    "huggingface_id": "astronomical-test/Large-6bit",
                    "model_id": "Large-6bit",
                    "tag": "Closer to the original",
                    "reason": "The same model as Large with more of its original detail kept. It needs 5.0 GB more memory."
                }
            ])
        );
    })
    .await
    .expect("the suggestions journey must finish within five seconds");
}

#[tokio::test]
async fn should_carry_no_suggestions_before_the_worker_reports_its_memory() {
    timeout(Duration::from_secs(5), async {
        assert_eq!(
            catalog_suggestions_with_ceiling(0).await,
            serde_json::json!([])
        );
    })
    .await
    .expect("the suggestions journey must finish within five seconds");
}
