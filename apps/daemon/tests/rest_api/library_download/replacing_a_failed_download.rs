//! A failed download must not hold the Library against other models (#1897): the REST journeys
//! for replacing it, for asking again for the failed model, and for a model asked for behind a
//! paused download, which waits on the list without starting.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use axum::{Router, http::StatusCode};
use tokio::time::timeout;

use super::*;

/// One model's download fails on every attempt (checksum mismatch); the
/// catalog also lists a second model. Returns the application and the failed
/// model's staged-bytes folder.
async fn application_with_a_failed_download(test_directory: &Path) -> (Router, PathBuf) {
    let catalog = two_model_catalog();
    let coordinator = build_test_coordinator(
        test_directory,
        Arc::clone(&catalog),
        Arc::new(ScriptedHub::checksum_mismatch()),
    );
    let application = build_application_with_library_download(
        ScriptedExecutor::unavailable(),
        catalog,
        coordinator,
    );
    assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
    wait_for_download_state(&application, "failed").await;
    let failed_staging_directory =
        test_directory.join(format!("models/.incomplete/{REPOSITORY_ID}"));
    assert!(
        failed_staging_directory.is_dir(),
        "the failed download kept its staged bytes"
    );
    (application, failed_staging_directory)
}

#[tokio::test]
async fn should_start_another_model_after_a_failed_download_through_rest() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let (application, failed_staging_directory) =
            application_with_a_failed_download(test_directory.path()).await;

        let other_model_response = start_download_of(&application, OTHER_REPOSITORY_ID).await;

        // The failed job is not running, so it no longer holds the Library.
        assert_eq!(other_model_response, StatusCode::ACCEPTED);
        wait_for_download_model(&application, OTHER_REPOSITORY_ID).await;
        assert!(
            !failed_staging_directory.exists(),
            "the failed model's staged bytes are discarded with its job"
        );
        // The failed model stays listed as failed, so the person can retry it.
        let download_document = get_download_document(&application).await;
        assert_eq!(
            download_document["queue"]["failed"][0]["huggingface_id"],
            REPOSITORY_ID
        );
        // Stop the replacing download so no transfer outlives this test.
        let cancel_response = post_without_body(&application, "/v1/library/download/cancel").await;
        assert!(cancel_response.status().is_success());
        wait_for_download_state(&application, "idle").await;
    })
    .await
    .expect("the failed-then-other-model journey should remain bounded");
}

#[tokio::test]
async fn should_keep_a_failed_download_when_the_requested_model_is_not_in_the_catalog() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let (application, failed_staging_directory) =
            application_with_a_failed_download(test_directory.path()).await;

        let unknown_model_response =
            start_download_of(&application, "astronomical-test/not-in-catalog").await;

        assert!(unknown_model_response.is_client_error());
        assert!(
            failed_staging_directory.is_dir(),
            "a refused request deletes nothing"
        );
        assert_eq!(get_download_document(&application).await["state"], "failed");
    })
    .await
    .expect("the unknown-model journey should remain bounded");
}

#[tokio::test]
async fn should_answer_current_for_the_failed_model_and_keep_its_bytes_for_resume() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let (application, failed_staging_directory) =
            application_with_a_failed_download(test_directory.path()).await;

        let same_model_response = start_download(&application).await;

        // Asking again for the current model starts nothing new: it stays
        // the current job, which the person resumes to keep its bytes.
        assert_eq!(same_model_response, StatusCode::OK);
        assert!(
            failed_staging_directory.is_dir(),
            "the same model resumes, keeping its bytes"
        );
    })
    .await
    .expect("the same-model journey should remain bounded");
}

#[tokio::test]
async fn should_list_another_model_behind_a_paused_download_without_starting_it() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let catalog = two_model_catalog();
        let coordinator = build_test_coordinator(
            test_directory.path(),
            Arc::clone(&catalog),
            Arc::new(ScriptedHub::delayed(Duration::from_millis(250))),
        );
        let application = build_application_with_library_download(
            ScriptedExecutor::unavailable(),
            catalog,
            coordinator,
        );
        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        wait_for_download_state(&application, "downloading").await;
        let pause_response = post_without_body(&application, "/v1/library/download/pause").await;
        assert_eq!(pause_response.status(), StatusCode::OK);

        let other_model_response = start_download_of(&application, OTHER_REPOSITORY_ID).await;

        // Pausing pauses the whole list: the other model waits its turn and
        // the paused download keeps the Library.
        assert_eq!(other_model_response, StatusCode::ACCEPTED);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let paused_job = get_download_document(&application).await;
        assert_eq!(paused_job["state"], "paused");
        assert_eq!(paused_job["huggingface_id"], REPOSITORY_ID);
        assert_eq!(
            paused_job["queue"]["waiting"][0]["huggingface_id"],
            OTHER_REPOSITORY_ID
        );
    })
    .await
    .expect("the paused journey should remain bounded");
}

/// A real-world failure is usually the network, not a bad digest: the connection drops, the job
/// ends `failed` with `download_failed`, and another model must still be able to start.
#[tokio::test]
async fn should_start_another_model_after_a_network_failure_through_rest() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let catalog = two_model_catalog();
        let coordinator = build_test_coordinator(
            test_directory.path(),
            Arc::clone(&catalog),
            Arc::new(ScriptedHub::network_failure()),
        );
        let application = build_application_with_library_download(
            ScriptedExecutor::unavailable(),
            catalog,
            coordinator,
        );
        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        let failed_job = wait_for_download_state(&application, "failed").await;
        assert_eq!(failed_job["error_code"], "download_failed");

        let other_model_response = start_download_of(&application, OTHER_REPOSITORY_ID).await;

        assert_eq!(other_model_response, StatusCode::ACCEPTED);
        wait_for_download_model(&application, OTHER_REPOSITORY_ID).await;
        wait_for_download_state(&application, "failed").await;
    })
    .await
    .expect("the network-failure journey should remain bounded");
}
