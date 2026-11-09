//! Remove from disk REST journeys: a model Download Models installed moves to
//! the Trash (a recording fake here, never the real Trash), and every refusal
//! leaves the folder where it is.

use std::path::PathBuf;
use std::sync::Mutex;

use astronomical_daemon::{
    ModelFolderTrash, ModelFolderTrashError, build_application_with_library_download_and_trash,
};
use axum::{body::Body, http::Request};
use tokio::time::{Duration, timeout};
use tower::ServiceExt;

use super::*;

const CATALOG_READY_POLL_ATTEMPTS: u32 = 400;

/// Records each folder it was asked to move and, when it succeeds, deletes it
/// the way a move out of the models folder would.
struct RecordingTrash {
    moved_folders: Mutex<Vec<PathBuf>>,
    outcome: Result<(), ModelFolderTrashError>,
}

impl RecordingTrash {
    fn succeeding() -> Arc<Self> {
        Arc::new(Self {
            moved_folders: Mutex::new(Vec::new()),
            outcome: Ok(()),
        })
    }

    fn unavailable() -> Arc<Self> {
        Arc::new(Self {
            moved_folders: Mutex::new(Vec::new()),
            outcome: Err(ModelFolderTrashError::Unavailable),
        })
    }

    fn moved_folders(&self) -> Vec<PathBuf> {
        self.moved_folders
            .lock()
            .expect("the recording lock should not be poisoned")
            .clone()
    }
}

impl ModelFolderTrash for RecordingTrash {
    fn is_available(&self) -> bool {
        self.outcome != Err(ModelFolderTrashError::Unavailable)
    }

    fn move_to_trash(&self, model_folder: &Path) -> Result<(), ModelFolderTrashError> {
        self.moved_folders
            .lock()
            .expect("the recording lock should not be poisoned")
            .push(model_folder.to_path_buf());
        if self.outcome.is_ok() {
            std::fs::remove_dir_all(model_folder).expect("the fake Trash should remove the folder");
        }
        self.outcome
    }
}

fn removal_application(
    test_directory: &Path,
    hub: Arc<ScriptedHub>,
    model_folder_trash: Arc<RecordingTrash>,
) -> Router {
    let download_catalog = test_catalog();
    let coordinator = build_test_coordinator(test_directory, Arc::clone(&download_catalog), hub);
    build_application_with_library_download_and_trash(
        ScriptedExecutor::unavailable(),
        download_catalog,
        coordinator,
        model_folder_trash,
    )
}

async fn remove_model(application: &Router, request_body: &str) -> axum::response::Response {
    remove_model_with_origin(application, request_body, None).await
}

async fn remove_model_with_origin(
    application: &Router,
    request_body: &str,
    origin: Option<&str>,
) -> axum::response::Response {
    let mut removal_request = Request::post("/v1/library/remove")
        .header("content-type", "application/json")
        .header("host", "127.0.0.1:6733");
    if let Some(origin) = origin {
        removal_request = removal_request.header("origin", origin);
    }
    application
        .clone()
        .oneshot(
            removal_request
                .body(Body::from(request_body.to_owned()))
                .expect("the removal request should be valid"),
        )
        .await
        .expect("the removal response should be available")
}

async fn first_catalog_entry(application: &Router) -> serde_json::Value {
    let catalog_response = application
        .clone()
        .oneshot(
            Request::get("/v1/library/catalog")
                .body(Body::empty())
                .expect("catalog request should be valid"),
        )
        .await
        .expect("catalog response should be available");
    response_json(catalog_response).await["entries"][0].clone()
}

async fn download_until_removable(application: &Router) {
    download_until_catalog_field(application, "removable_from_disk").await;
}

async fn download_until_catalog_field(application: &Router, catalog_field: &str) {
    assert_eq!(start_download(application).await, StatusCode::ACCEPTED);
    wait_for_download_state(application, "idle").await;
    for _poll_attempt in 0..CATALOG_READY_POLL_ATTEMPTS {
        if first_catalog_entry(application).await[catalog_field] == true {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the downloaded model never reported {catalog_field}");
}

fn installed_model_folder(test_directory: &Path) -> PathBuf {
    test_directory.join("models").join(REPOSITORY_ID)
}

#[tokio::test]
async fn should_move_a_downloaded_model_to_the_trash_and_stop_listing_it_on_this_mac() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let recording_trash = RecordingTrash::succeeding();
        let application = removal_application(
            test_directory.path(),
            Arc::new(ScriptedHub::new()),
            Arc::clone(&recording_trash),
        );
        download_until_removable(&application).await;

        let removal_response = remove_model(
            &application,
            &format!(r#"{{"huggingface_id":"{REPOSITORY_ID}"}}"#),
        )
        .await;

        assert_eq!(removal_response.status(), StatusCode::OK);
        assert_eq!(
            recording_trash.moved_folders(),
            vec![installed_model_folder(test_directory.path())]
        );
        let catalog_entry = first_catalog_entry(&application).await;
        assert_eq!(catalog_entry["ready_on_this_mac"], false);
        assert_eq!(catalog_entry["removable_from_disk"], false);
        let publisher_folder = installed_model_folder(test_directory.path())
            .parent()
            .expect("the model folder should have a publisher folder")
            .to_path_buf();
        assert!(
            !publisher_folder.exists(),
            "the publisher folder left empty by the removal is tidied away"
        );
        assert!(
            test_directory.path().join("models").is_dir(),
            "the models folder itself stays"
        );
    })
    .await
    .expect("the removal journey must finish within five seconds");
}

#[tokio::test]
async fn should_refuse_a_folder_that_download_models_did_not_install() {
    let test_directory = tempfile::tempdir().expect("temporary directory should be available");
    let placed_folder = installed_model_folder(test_directory.path());
    std::fs::create_dir_all(&placed_folder).expect("the placed folder should be created");
    std::fs::write(placed_folder.join("config.json"), MODEL_CONFIG)
        .expect("the placed model file should be written");
    let recording_trash = RecordingTrash::succeeding();
    let application = removal_application(
        test_directory.path(),
        Arc::new(ScriptedHub::new()),
        Arc::clone(&recording_trash),
    );

    let removal_response = remove_model(
        &application,
        &format!(r#"{{"huggingface_id":"{REPOSITORY_ID}"}}"#),
    )
    .await;

    assert_eq!(removal_response.status(), StatusCode::CONFLICT);
    assert_eq!(
        response_json(removal_response).await["error"]["code"],
        "not_installed_by_download_models"
    );
    assert!(recording_trash.moved_folders().is_empty());
    assert!(placed_folder.exists(), "a copy the person placed stays");
}

#[tokio::test]
async fn should_refuse_a_model_outside_the_catalog_without_touching_anything() {
    let test_directory = tempfile::tempdir().expect("temporary directory should be available");
    let recording_trash = RecordingTrash::succeeding();
    let application = removal_application(
        test_directory.path(),
        Arc::new(ScriptedHub::new()),
        Arc::clone(&recording_trash),
    );

    let removal_response = remove_model(
        &application,
        r#"{"huggingface_id":"astronomical-test/not-in-the-catalog"}"#,
    )
    .await;

    assert_eq!(removal_response.status(), StatusCode::NOT_FOUND);
    assert!(recording_trash.moved_folders().is_empty());
}

#[tokio::test]
async fn should_reject_a_request_that_names_a_path() {
    let test_directory = tempfile::tempdir().expect("temporary directory should be available");
    let recording_trash = RecordingTrash::succeeding();
    let application = removal_application(
        test_directory.path(),
        Arc::new(ScriptedHub::new()),
        Arc::clone(&recording_trash),
    );

    let removal_response = remove_model(
        &application,
        &format!(r#"{{"huggingface_id":"{REPOSITORY_ID}","path":"/"}}"#),
    )
    .await;

    assert!(removal_response.status().is_client_error());
    assert!(recording_trash.moved_folders().is_empty());
}

#[tokio::test]
async fn should_keep_the_model_and_say_so_when_this_mac_has_no_trash_command() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let recording_trash = RecordingTrash::unavailable();
        let application = removal_application(
            test_directory.path(),
            Arc::new(ScriptedHub::new()),
            Arc::clone(&recording_trash),
        );
        download_until_catalog_field(&application, "ready_on_this_mac").await;
        assert_eq!(
            first_catalog_entry(&application).await["removable_from_disk"],
            false,
            "a Mac without the trash command is never offered Remove from disk"
        );

        let removal_response = remove_model(
            &application,
            &format!(r#"{{"huggingface_id":"{REPOSITORY_ID}"}}"#),
        )
        .await;

        assert_eq!(removal_response.status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            response_json(removal_response).await["error"]["code"],
            "trash_unavailable"
        );
        assert!(installed_model_folder(test_directory.path()).exists());
        assert_eq!(
            first_catalog_entry(&application).await["ready_on_this_mac"],
            true
        );
    })
    .await
    .expect("the unavailable-Trash journey must finish within five seconds");
}

#[tokio::test]
async fn should_refuse_while_the_model_is_still_downloading() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let recording_trash = RecordingTrash::succeeding();
        let application = removal_application(
            test_directory.path(),
            Arc::new(ScriptedHub::delayed(Duration::from_secs(2))),
            Arc::clone(&recording_trash),
        );
        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        wait_for_download_state(&application, "downloading").await;

        let removal_response = remove_model(
            &application,
            &format!(r#"{{"huggingface_id":"{REPOSITORY_ID}"}}"#),
        )
        .await;

        assert_eq!(removal_response.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(removal_response).await["error"]["code"],
            "download_in_progress"
        );
        assert!(recording_trash.moved_folders().is_empty());
    })
    .await
    .expect("the in-progress journey must finish within five seconds");
}

#[tokio::test]
async fn should_refuse_a_removal_from_another_sites_page_and_keep_the_model() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let recording_trash = RecordingTrash::succeeding();
        let application = removal_application(
            test_directory.path(),
            Arc::new(ScriptedHub::new()),
            Arc::clone(&recording_trash),
        );
        download_until_removable(&application).await;

        let removal_response = remove_model_with_origin(
            &application,
            &format!(r#"{{"huggingface_id":"{REPOSITORY_ID}"}}"#),
            Some("http://rebound.example:6733"),
        )
        .await;

        assert_eq!(removal_response.status(), StatusCode::FORBIDDEN);
        assert!(recording_trash.moved_folders().is_empty());
        assert!(installed_model_folder(test_directory.path()).exists());
    })
    .await
    .expect("the foreign-origin journey must finish within five seconds");
}

#[tokio::test]
async fn should_accept_a_removal_from_this_instances_own_page() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let recording_trash = RecordingTrash::succeeding();
        let application = removal_application(
            test_directory.path(),
            Arc::new(ScriptedHub::new()),
            Arc::clone(&recording_trash),
        );
        download_until_removable(&application).await;

        let removal_response = remove_model_with_origin(
            &application,
            &format!(r#"{{"huggingface_id":"{REPOSITORY_ID}"}}"#),
            Some("http://127.0.0.1:6733"),
        )
        .await;

        assert_eq!(removal_response.status(), StatusCode::OK);
        assert_eq!(recording_trash.moved_folders().len(), 1);
    })
    .await
    .expect("the own-page journey must finish within five seconds");
}

#[tokio::test]
async fn should_say_a_catalog_model_without_a_folder_is_not_on_this_mac() {
    let test_directory = tempfile::tempdir().expect("temporary directory should be available");
    let recording_trash = RecordingTrash::succeeding();
    let application = removal_application(
        test_directory.path(),
        Arc::new(ScriptedHub::new()),
        Arc::clone(&recording_trash),
    );

    let removal_response = remove_model(
        &application,
        &format!(r#"{{"huggingface_id":"{REPOSITORY_ID}"}}"#),
    )
    .await;

    assert_eq!(removal_response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        response_json(removal_response).await["error"]["code"],
        "not_on_this_mac"
    );
    assert!(recording_trash.moved_folders().is_empty());
}
