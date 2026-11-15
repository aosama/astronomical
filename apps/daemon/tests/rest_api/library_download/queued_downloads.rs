//! The download list through REST: models asked for while another downloads
//! wait their turn, start one after another, can be taken off the list, and
//! are refused at once when the list would not fit on the disk.

use std::{io, path::Path, sync::Arc, time::Duration};

use axum::{Router, http::StatusCode};
use tokio::time::timeout;

use super::*;

const SLOW_PAYLOAD_DELAY: Duration = Duration::from_millis(250);

/// A free-space reading that fits one test model with room to spare, but not
/// two.
struct OneModelCapacity;

impl DiskCapacityQuery for OneModelCapacity {
    fn available_space_bytes(&self, _existing_same_volume_path: &Path) -> io::Result<u64> {
        Ok(u64::try_from(repository_bytes() * 3 / 2).expect("the fixture size fits in u64"))
    }
}

fn two_model_application(
    test_directory: &Path,
    hub: ScriptedHub,
    capacity_query: Arc<dyn DiskCapacityQuery>,
) -> Router {
    let catalog = two_model_catalog();
    let coordinator = build_test_coordinator_with_capacity(
        test_directory,
        Arc::clone(&catalog),
        Arc::new(hub),
        capacity_query,
    );
    build_application_with_library_download(ScriptedExecutor::unavailable(), catalog, coordinator)
}

async fn post_json(
    application: &Router,
    path: &str,
    huggingface_id: &str,
) -> (StatusCode, serde_json::Value) {
    let response = application
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(format!(
                    "{{\"huggingface_id\":\"{huggingface_id}\"}}"
                )))
                .expect("the request should be valid"),
        )
        .await
        .expect("the response should be available");
    let status = response.status();
    (status, response_json(response).await)
}

async fn cancel_and_wait_for_idle(application: &Router) {
    let cancel_response = post_without_body(application, "/v1/library/download/cancel").await;
    assert!(cancel_response.status().is_success());
    wait_for_download_state(application, "idle").await;
}

#[tokio::test]
async fn should_list_a_second_model_and_download_it_after_the_first() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let application = two_model_application(
            test_directory.path(),
            ScriptedHub::delayed(SLOW_PAYLOAD_DELAY),
            Arc::new(AvailableCapacity),
        );
        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);

        let (status, queued_document) =
            post_json(&application, "/v1/library/download", OTHER_REPOSITORY_ID).await;

        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(queued_document["state"], "queued");
        assert_eq!(queued_document["queue_position"], 1);
        let listed_document = get_download_document(&application).await;
        assert_eq!(listed_document["huggingface_id"], REPOSITORY_ID);
        assert_eq!(
            listed_document["queue"]["waiting"][0]["huggingface_id"],
            OTHER_REPOSITORY_ID
        );
        assert_eq!(
            listed_document["queue"]["waiting"][0]["approximate_size_bytes"],
            repository_bytes()
        );
        wait_for_download_model(&application, OTHER_REPOSITORY_ID).await;
        wait_for_download_state(&application, "idle").await;
        assert!(
            test_directory
                .path()
                .join("models")
                .join(REPOSITORY_ID)
                .is_dir()
        );
        assert!(
            test_directory
                .path()
                .join("models")
                .join(OTHER_REPOSITORY_ID)
                .is_dir()
        );
        assert_eq!(
            get_download_document(&application).await["queue"]["waiting"],
            serde_json::json!([])
        );
    })
    .await
    .expect("the two-model journey should remain bounded");
}

#[tokio::test]
async fn should_keep_the_place_of_a_model_asked_for_twice_and_answer_current_for_the_running_one() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let application = two_model_application(
            test_directory.path(),
            ScriptedHub::delayed(SLOW_PAYLOAD_DELAY),
            Arc::new(AvailableCapacity),
        );
        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        post_json(&application, "/v1/library/download", OTHER_REPOSITORY_ID).await;

        let (again_status, again_document) =
            post_json(&application, "/v1/library/download", OTHER_REPOSITORY_ID).await;
        let (current_status, current_document) =
            post_json(&application, "/v1/library/download", REPOSITORY_ID).await;

        assert_eq!(again_status, StatusCode::ACCEPTED);
        assert_eq!(again_document["queue_position"], 1);
        assert_eq!(current_status, StatusCode::OK);
        assert_eq!(current_document["state"], "current");
        assert_eq!(
            get_download_document(&application).await["queue"]["waiting"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );
        post_json(
            &application,
            "/v1/library/download/queue/remove",
            OTHER_REPOSITORY_ID,
        )
        .await;
        cancel_and_wait_for_idle(&application).await;
    })
    .await
    .expect("the repeated-request journey should remain bounded");
}

#[tokio::test]
async fn should_keep_a_failed_model_listed_and_go_on_to_the_next() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let application = two_model_application(
            test_directory.path(),
            ScriptedHub::checksum_mismatch(),
            Arc::new(AvailableCapacity),
        );
        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        post_json(&application, "/v1/library/download", OTHER_REPOSITORY_ID).await;

        wait_for_download_model(&application, OTHER_REPOSITORY_ID).await;
        let failed_document = wait_for_download_state(&application, "failed").await;

        assert_eq!(failed_document["huggingface_id"], OTHER_REPOSITORY_ID);
        assert_eq!(
            failed_document["queue"]["failed"][0]["huggingface_id"],
            REPOSITORY_ID
        );
        assert_eq!(
            failed_document["queue"]["failed"][0]["error_code"],
            "checksum_mismatch"
        );
    })
    .await
    .expect("the failed-then-next journey should remain bounded");
}

#[tokio::test]
async fn should_take_a_waiting_model_off_the_list_and_refuse_an_unlisted_one() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let application = two_model_application(
            test_directory.path(),
            ScriptedHub::delayed(SLOW_PAYLOAD_DELAY),
            Arc::new(AvailableCapacity),
        );
        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        post_json(&application, "/v1/library/download", OTHER_REPOSITORY_ID).await;

        let (removed_status, removed_document) = post_json(
            &application,
            "/v1/library/download/queue/remove",
            OTHER_REPOSITORY_ID,
        )
        .await;
        let (unlisted_status, unlisted_document) = post_json(
            &application,
            "/v1/library/download/queue/remove",
            OTHER_REPOSITORY_ID,
        )
        .await;

        assert_eq!(removed_status, StatusCode::OK);
        assert_eq!(removed_document["queue"]["waiting"], serde_json::json!([]));
        assert_eq!(unlisted_status, StatusCode::NOT_FOUND);
        assert_eq!(unlisted_document["error"]["code"], "not_in_queue");
        cancel_and_wait_for_idle(&application).await;
    })
    .await
    .expect("the removal journey should remain bounded");
}

#[tokio::test]
async fn should_refuse_a_model_that_would_not_fit_on_the_disk_beside_the_list() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let application = two_model_application(
            test_directory.path(),
            ScriptedHub::delayed(SLOW_PAYLOAD_DELAY),
            Arc::new(OneModelCapacity),
        );
        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);

        let (refused_status, refused_document) =
            post_json(&application, "/v1/library/download", OTHER_REPOSITORY_ID).await;

        assert_eq!(refused_status, StatusCode::CONFLICT);
        assert_eq!(refused_document["error"]["code"], "insufficient_disk");
        assert!(
            refused_document["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains(" GB"))
        );
        assert_eq!(
            get_download_document(&application).await["queue"]["waiting"],
            serde_json::json!([])
        );
        cancel_and_wait_for_idle(&application).await;
    })
    .await
    .expect("the disk refusal journey should remain bounded");
}

#[tokio::test]
async fn should_start_the_next_model_when_the_current_one_is_cancelled() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let application = two_model_application(
            test_directory.path(),
            ScriptedHub::delayed(SLOW_PAYLOAD_DELAY),
            Arc::new(AvailableCapacity),
        );
        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        post_json(&application, "/v1/library/download", OTHER_REPOSITORY_ID).await;

        let cancel_response = post_without_body(&application, "/v1/library/download/cancel").await;

        assert!(cancel_response.status().is_success());
        wait_for_download_model(&application, OTHER_REPOSITORY_ID).await;
        cancel_and_wait_for_idle(&application).await;
    })
    .await
    .expect("the cancel-then-next journey should remain bounded");
}

/// The list lives beside the models, so a restart keeps it: the recovered
/// download holds the Library as paused, and once it is resumed and done,
/// the model listed before the restart downloads next.
#[tokio::test]
async fn should_keep_the_list_across_a_restart_and_carry_on_after_the_resumed_download() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let application = two_model_application(
            test_directory.path(),
            ScriptedHub::delayed(SLOW_PAYLOAD_DELAY),
            Arc::new(AvailableCapacity),
        );
        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        wait_for_download_state(&application, "downloading").await;
        post_json(&application, "/v1/library/download", OTHER_REPOSITORY_ID).await;
        let pause_response = post_without_body(&application, "/v1/library/download/pause").await;
        assert_eq!(pause_response.status(), StatusCode::OK);

        let restarted_application = two_model_application(
            test_directory.path(),
            ScriptedHub::new(),
            Arc::new(AvailableCapacity),
        );
        let restarted_document = get_download_document(&restarted_application).await;
        let resume_response =
            post_without_body(&restarted_application, "/v1/library/download/resume").await;

        assert_eq!(restarted_document["state"], "paused");
        assert_eq!(
            restarted_document["queue"]["waiting"][0]["huggingface_id"],
            OTHER_REPOSITORY_ID
        );
        assert_eq!(resume_response.status(), StatusCode::ACCEPTED);
        wait_for_download_model(&restarted_application, OTHER_REPOSITORY_ID).await;
        wait_for_download_state(&restarted_application, "idle").await;
        assert!(
            test_directory
                .path()
                .join("models")
                .join(OTHER_REPOSITORY_ID)
                .is_dir()
        );
    })
    .await
    .expect("the restart journey should remain bounded");
}
