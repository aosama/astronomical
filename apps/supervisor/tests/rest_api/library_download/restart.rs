//! Restart and pre-existing-destination acceptance journeys for Library publication.

use std::fs;

use axum::{body::Body, http::Request};
use tokio::time::{Duration, timeout};
use tower::ServiceExt;

use super::*;

#[tokio::test]
async fn should_remain_ready_after_the_application_restarts() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let application =
            build_test_application(test_directory.path(), Arc::new(ScriptedHub::new()));

        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        wait_for_download_state(&application, "idle").await;
        wait_for_catalog_ready(&application).await;
        drop(application);

        let restarted_application =
            build_test_application(test_directory.path(), Arc::new(ScriptedHub::new()));
        let catalog_response = restarted_application
            .oneshot(
                Request::get("/v1/library/catalog")
                    .body(Body::empty())
                    .expect("catalog request should be valid"),
            )
            .await
            .expect("catalog response should be available");
        let catalog_document = response_json(catalog_response).await;

        assert_eq!(catalog_document["entries"][0]["ready_on_this_mac"], true);
        assert_eq!(
            catalog_document["entries"][0]["download_state"],
            serde_json::Value::Null
        );
        assert_eq!(
            catalog_document["entries"][0]["requestable_model_id"],
            "example-qwen"
        );
    })
    .await
    .expect("Library restart journey should remain bounded");
}

#[tokio::test]
async fn should_adopt_an_exact_pre_existing_publication_after_manifest_verification() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        write_published_fixture(test_directory.path(), MODEL_CONFIG);
        let application =
            build_test_application(test_directory.path(), Arc::new(ScriptedHub::new()));

        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        wait_for_download_state(&application, "idle").await;
        let catalog_document = wait_for_catalog_ready(&application).await;

        assert_eq!(catalog_document["entries"][0]["ready_on_this_mac"], true);
        assert!(
            test_directory
                .path()
                .join(format!(
                    "models/{REPOSITORY_ID}/.astronomical-library-provenance.json"
                ))
                .is_file(),
            "adopted publications should retain immutable provenance across restart"
        );
    })
    .await
    .expect("exact destination reconciliation should remain bounded");
}

#[tokio::test]
async fn should_leave_a_mismatched_pre_existing_publication_untouched() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        write_published_fixture(test_directory.path(), CORRUPT_MODEL_CONFIG);
        let application =
            build_test_application(test_directory.path(), Arc::new(ScriptedHub::new()));

        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        let failed_job = wait_for_download_state(&application, "failed").await;
        let existing_config = fs::read(
            test_directory
                .path()
                .join(format!("models/{REPOSITORY_ID}/config.json")),
        )
        .expect("existing config should remain readable");

        assert_eq!(failed_job["error_code"], "model_already_present");
        assert_eq!(existing_config, CORRUPT_MODEL_CONFIG);
        assert!(
            !test_directory
                .path()
                .join(format!(
                    "models/{REPOSITORY_ID}/.astronomical-library-provenance.json"
                ))
                .exists(),
            "a mismatched destination must not receive Library provenance"
        );
    })
    .await
    .expect("mismatched destination rejection should remain bounded");
}

async fn wait_for_catalog_ready(application: &Router) -> serde_json::Value {
    for _poll_attempt in 0..100 {
        let catalog_response = application
            .clone()
            .oneshot(
                Request::get("/v1/library/catalog")
                    .body(Body::empty())
                    .expect("catalog request should be valid"),
            )
            .await
            .expect("catalog response should be available");
        let catalog_document = response_json(catalog_response).await;
        if catalog_document["entries"][0]["ready_on_this_mac"] == true {
            return catalog_document;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("published model did not become ready");
}

fn write_published_fixture(test_directory: &Path, model_config: &[u8]) {
    let published_directory = test_directory.join(format!("models/{REPOSITORY_ID}"));
    fs::create_dir_all(&published_directory).expect("published fixture directory should exist");
    for (relative_path, payload) in [
        ("config.json", model_config),
        ("tokenizer.json", TOKENIZER_CONFIG),
        ("model-00001.safetensors", MODEL_WEIGHTS),
        ("model.safetensors.index.json", MODEL_INDEX),
    ] {
        fs::write(published_directory.join(relative_path), payload)
            .expect("published fixture file should be written");
    }
}
