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

const OWN_PROVENANCE: &str =
    r#"{"schema_version":1,"provider_model_id":"astronomical-test/example-qwen","revision":"r"}"#;

/// A model folder Download Models published, written directly so a journey can
/// pair it with a refresh or worker state that a real download could not reach.
fn place_published_model(test_directory: &Path) -> PathBuf {
    let model_folder = installed_model_folder(test_directory);
    std::fs::create_dir_all(&model_folder).expect("the model folder should be created");
    std::fs::write(model_folder.join("config.json"), MODEL_CONFIG)
        .expect("the model file should be written");
    std::fs::write(
        model_folder.join(".astronomical-library-provenance.json"),
        OWN_PROVENANCE,
    )
    .expect("the provenance record should be written");
    model_folder
}

fn removal_application_with(
    test_directory: &Path,
    executor: ScriptedExecutor,
    discovery_refresh: Arc<dyn DownloadPublicationRefresh>,
    model_folder_trash: Arc<RecordingTrash>,
) -> Router {
    let download_catalog = test_catalog();
    let hub = Arc::new(ScriptedHub::new());
    let coordinator = Arc::new(LibraryDownloadCoordinator::new(
        Arc::clone(&download_catalog),
        test_directory.join("models"),
        Arc::new(AvailableCapacity),
        hub.clone(),
        hub,
        discovery_refresh,
        DaemonPerformanceAttributionLog::open(test_directory, false)
            .expect("disabled attribution should construct"),
    ));
    build_application_with_library_download_and_trash(
        executor,
        download_catalog,
        coordinator,
        model_folder_trash,
    )
}

fn executor_with_worker(
    ready_model_id: Option<&str>,
    loading_model: Option<Option<&str>>,
) -> ScriptedExecutor {
    let mut executor = ScriptedExecutor::unavailable();
    executor.health_snapshot.ready_model_id = ready_model_id.map(str::to_owned);
    executor.health_snapshot.model_load_in_flight =
        loading_model.map(
            |loading_model_id| astronomical_ipc_protocol::ModelLoadInFlight {
                model_id: loading_model_id.map(str::to_owned),
            },
        );
    executor
}

#[tokio::test]
async fn should_refuse_while_a_model_from_the_folder_is_loaded_or_loading() {
    for (executor, expected_code) in [
        (
            executor_with_worker(Some("example-qwen"), None),
            "model_loaded",
        ),
        (
            executor_with_worker(Some("other-model"), Some(None)),
            "model_loading",
        ),
        (
            executor_with_worker(None, Some(Some("example-qwen"))),
            "model_loading",
        ),
    ] {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let model_folder = place_published_model(test_directory.path());
        let recording_trash = RecordingTrash::succeeding();
        let application = removal_application_with(
            test_directory.path(),
            executor,
            Arc::new(ValidatingDiscoveryRefresh),
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
            expected_code
        );
        assert!(recording_trash.moved_folders().is_empty());
        assert!(model_folder.is_dir(), "a busy model's folder stays");
    }
}

#[tokio::test]
async fn should_report_removal_even_when_the_discovery_refresh_fails() {
    let test_directory = tempfile::tempdir().expect("temporary directory should be available");
    let model_folder = place_published_model(test_directory.path());
    let recording_trash = RecordingTrash::succeeding();
    let application = removal_application_with(
        test_directory.path(),
        ScriptedExecutor::unavailable(),
        Arc::new(FailingDiscoveryRefresh),
        Arc::clone(&recording_trash),
    );

    let removal_response = remove_model(
        &application,
        &format!(r#"{{"huggingface_id":"{REPOSITORY_ID}"}}"#),
    )
    .await;

    assert_eq!(
        removal_response.status(),
        StatusCode::OK,
        "the folder is gone; the next discovery catches up"
    );
    assert_eq!(recording_trash.moved_folders(), vec![model_folder.clone()]);
    assert!(!model_folder.exists());
}

#[tokio::test]
async fn should_keep_a_publisher_folder_that_still_holds_something() {
    let test_directory = tempfile::tempdir().expect("temporary directory should be available");
    let model_folder = place_published_model(test_directory.path());
    let publisher_folder = model_folder
        .parent()
        .expect("the model has a publisher folder")
        .to_path_buf();
    std::fs::write(publisher_folder.join(".DS_Store"), b"finder state")
        .expect("the Finder file should be written");
    let recording_trash = RecordingTrash::succeeding();
    let application = removal_application_with(
        test_directory.path(),
        ScriptedExecutor::unavailable(),
        Arc::new(ValidatingDiscoveryRefresh),
        Arc::clone(&recording_trash),
    );

    let removal_response = remove_model(
        &application,
        &format!(r#"{{"huggingface_id":"{REPOSITORY_ID}"}}"#),
    )
    .await;

    assert_eq!(removal_response.status(), StatusCode::OK);
    assert!(
        publisher_folder.join(".DS_Store").is_file(),
        "only an empty publisher folder is removed"
    );
}

async fn send_with_origin(
    application: &Router,
    method: &str,
    request_path: &str,
    host: &str,
    origin: &str,
) -> StatusCode {
    let request_builder = Request::builder()
        .method(method)
        .uri(request_path)
        .header("content-type", "application/json")
        .header("host", host)
        .header("origin", origin);
    let request_body = if method == "GET" {
        Body::empty()
    } else {
        Body::from(format!(r#"{{"huggingface_id":"{REPOSITORY_ID}"}}"#))
    };
    application
        .clone()
        .oneshot(
            request_builder
                .body(request_body)
                .expect("the request should be valid"),
        )
        .await
        .expect("the response should be available")
        .status()
}

/// Astro drives downloads from the Thin Talk canvas, whose page origin is the
/// app's own `thintalk-asset://shell`; it may download but never remove.
#[tokio::test]
async fn should_let_the_thin_talk_canvas_drive_downloads_but_not_remove_models() {
    let test_directory = tempfile::tempdir().expect("temporary directory should be available");
    let application = removal_application(
        test_directory.path(),
        Arc::new(ScriptedHub::new()),
        RecordingTrash::succeeding(),
    );
    let canvas_origin = "thintalk-asset://shell";

    assert_eq!(
        send_with_origin(
            &application,
            "GET",
            "/v1/library/download",
            "127.0.0.1:6733",
            canvas_origin
        )
        .await,
        StatusCode::OK
    );
    assert_eq!(
        send_with_origin(
            &application,
            "POST",
            "/v1/library/download",
            "127.0.0.1:6733",
            canvas_origin
        )
        .await,
        StatusCode::ACCEPTED
    );
    assert_eq!(
        send_with_origin(
            &application,
            "POST",
            "/v1/library/remove",
            "127.0.0.1:6733",
            canvas_origin
        )
        .await,
        StatusCode::FORBIDDEN
    );
}

/// DNS rebinding: the hostile domain resolves to this Mac, so the browser
/// names it in `Host` as well as `Origin` and treats the request as same-site.
#[tokio::test]
async fn should_refuse_a_rebound_domain_that_names_itself_in_host_and_origin() {
    let test_directory = tempfile::tempdir().expect("temporary directory should be available");
    let application = removal_application(
        test_directory.path(),
        Arc::new(ScriptedHub::new()),
        RecordingTrash::succeeding(),
    );
    for request_path in ["/v1/library/download", "/v1/library/remove"] {
        assert_eq!(
            send_with_origin(
                &application,
                "POST",
                request_path,
                "rebound.example:6733",
                "http://rebound.example:6733"
            )
            .await,
            StatusCode::FORBIDDEN,
            "{request_path} answered a rebound domain"
        );
    }
}
