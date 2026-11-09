//! `DownloadStart` over the daemon IPC door the CLI uses, against a real
//! coordinator and a scripted network. These pin what a person relies on when
//! a large download outlasts one command: asking again attaches to the running
//! download, a paused one resumes, and a restart loses no staged progress.

use std::path::Path;

use astronomical_config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};
use astronomical_daemon::{
    DaemonIpcGenerationContext, DaemonIpcService, DaemonPerformanceAttributionLog,
    ImageGenerationExecutor, start_daemon_ipc_service,
};
use astronomical_ipc_protocol::{DaemonIpcClient, DaemonRequest, DaemonResponse};
use std::sync::atomic::AtomicU64;

use super::*;

/// One service over `coordinator`, answering on the instance socket in
/// `state_directory`.
async fn start_ipc_service(
    state_directory: &Path,
    download_catalog: Arc<DownloadCatalog>,
    coordinator: Arc<LibraryDownloadCoordinator>,
) -> DaemonIpcService {
    let instance_paths = AstronomicalInstancePaths::for_state_directory(
        state_directory.to_path_buf(),
        AstronomicalRuntimeInstance::Development,
    );
    let generation_context = DaemonIpcGenerationContext {
        executor: Arc::new(ScriptedExecutor::unavailable()) as Arc<dyn ImageGenerationExecutor>,
        reloadable_config: None,
        next_chat_request_id: Arc::new(AtomicU64::new(1)),
        download_catalog,
        library_download_coordinator: Some(coordinator),
        instance_paths: instance_paths.clone(),
    };
    let attribution_log = DaemonPerformanceAttributionLog::open(state_directory, false)
        .expect("disabled attribution should construct");
    start_daemon_ipc_service(&instance_paths, generation_context, &attribution_log)
        .await
        .expect("the IPC service should start on the instance socket")
}

/// One request on a fresh connection, as each CLI poll makes it.
async fn ask(service: &DaemonIpcService, daemon_request: DaemonRequest) -> DaemonResponse {
    let mut daemon_client = DaemonIpcClient::connect(service.socket_path().to_path_buf())
        .await
        .expect("the client should connect to the running daemon IPC service");
    daemon_client
        .send_request(&daemon_request)
        .await
        .expect("the request should transmit");
    daemon_client
        .next_response()
        .await
        .expect("the reply read should not fail at the transport layer")
        .expect("the daemon should answer the request")
}

fn download_start() -> DaemonRequest {
    DaemonRequest::DownloadStart {
        model_id: REPOSITORY_ID.to_owned(),
    }
}

fn assert_download_started(daemon_response: &DaemonResponse, context: &str) {
    assert_eq!(
        daemon_response,
        &DaemonResponse::DownloadStarted {
            huggingface_id: REPOSITORY_ID.to_owned()
        },
        "{context}"
    );
}

/// Polls the job the way the CLI does until the daemon reports no job.
async fn wait_until_no_job(service: &DaemonIpcService) {
    for _poll_attempt in 0..200 {
        if let DaemonResponse::DownloadStatus { job: None } =
            ask(service, DaemonRequest::DownloadStatus).await
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the download never finished");
}

/// Polls until the job exists and has reached `Downloading`.
async fn wait_until_downloading(service: &DaemonIpcService) {
    for _poll_attempt in 0..200 {
        if let DaemonResponse::DownloadStatus { job: Some(job) } =
            ask(service, DaemonRequest::DownloadStatus).await
            && job.state == "downloading"
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the download never reached the downloading state");
}

fn model_config_exists(test_directory: &Path) -> bool {
    test_directory
        .join(format!("models/{REPOSITORY_ID}/config.json"))
        .is_file()
}

/// Waits for the real outcome: the model published into the models folder.
/// "No job" alone is not enough, because the very first poll can run before
/// the background task has created its job at all.
async fn wait_until_published(test_directory: &Path) {
    for _poll_attempt in 0..300 {
        if model_config_exists(test_directory) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the model was never published");
}

#[tokio::test]
async fn should_start_a_fresh_download_through_download_start() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let catalog = test_catalog();
        let coordinator = build_test_coordinator(
            test_directory.path(),
            Arc::clone(&catalog),
            Arc::new(ScriptedHub::new()),
        );
        let service = start_ipc_service(test_directory.path(), catalog, coordinator).await;

        let start_response = ask(&service, download_start()).await;

        assert_download_started(&start_response, "a fresh model must start");
        wait_until_published(test_directory.path()).await;
        wait_until_no_job(&service).await;
        service.shutdown().await.expect("clean shutdown");
    })
    .await
    .expect("the fresh-start journey should remain bounded");
}

/// The rerun-after-timeout case. Before this behavior existed the second
/// request was refused ("another Library download is active") and a download
/// that outlasted one command could not be followed to its end.
#[tokio::test]
async fn should_attach_to_a_download_that_is_already_running() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let catalog = test_catalog();
        let coordinator = build_test_coordinator(
            test_directory.path(),
            Arc::clone(&catalog),
            Arc::new(ScriptedHub::delayed(Duration::from_millis(250))),
        );
        let service = start_ipc_service(test_directory.path(), catalog, coordinator).await;
        assert_download_started(
            &ask(&service, download_start()).await,
            "the first run starts it",
        );
        wait_until_downloading(&service).await;

        let second_response = ask(&service, download_start()).await;
        let third_response = ask(&service, download_start()).await;

        assert_download_started(&second_response, "a rerun must attach, not be refused");
        assert_download_started(&third_response, "attaching must be repeatable");
        wait_until_published(test_directory.path()).await;
        wait_until_no_job(&service).await;
        service.shutdown().await.expect("clean shutdown");
    })
    .await
    .expect("the attach journey should remain bounded");
}

#[tokio::test]
async fn should_resume_a_paused_download_through_download_start() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let catalog = test_catalog();
        let coordinator = build_test_coordinator(
            test_directory.path(),
            Arc::clone(&catalog),
            Arc::new(ScriptedHub::delayed(Duration::from_millis(250))),
        );
        let service =
            start_ipc_service(test_directory.path(), catalog, Arc::clone(&coordinator)).await;
        assert_download_started(&ask(&service, download_start()).await, "start");
        wait_until_downloading(&service).await;
        coordinator
            .pause()
            .await
            .expect("the running download should pause");

        let resume_response = ask(&service, download_start()).await;

        assert_download_started(&resume_response, "a paused download must resume");
        wait_until_published(test_directory.path()).await;
        wait_until_no_job(&service).await;
        service.shutdown().await.expect("clean shutdown");
    })
    .await
    .expect("the pause-and-resume journey should remain bounded");
}

/// A first daemon that is interrupted mid-download, leaving its job and staged
/// bytes in `test_directory` for the next daemon to find.
async fn interrupt_a_running_download(test_directory: &Path) {
    let catalog = test_catalog();
    let coordinator = build_test_coordinator(
        test_directory,
        Arc::clone(&catalog),
        Arc::new(ScriptedHub::delayed(Duration::from_millis(250))),
    );
    let service = start_ipc_service(test_directory, catalog, Arc::clone(&coordinator)).await;
    assert_download_started(&ask(&service, download_start()).await, "start");
    wait_until_downloading(&service).await;
    wait_until_bytes_are_staged(test_directory).await;
    coordinator
        .pause()
        .await
        .expect("the download should be interrupted");
    service
        .shutdown()
        .await
        .expect("the first daemon shuts down");
}

/// The job reports "downloading" before its first chunk reaches the staging
/// folder, so a pause right after that report could leave nothing staged; the
/// restart journeys need staged bytes to prove a paused download keeps them.
async fn wait_until_bytes_are_staged(test_directory: &Path) {
    let staging_directory = test_directory.join(format!("models/.incomplete/{REPOSITORY_ID}"));
    for _poll_attempt in 0..500 {
        if staging_directory.is_dir() && staged_byte_count(&staging_directory) > 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the running download never staged a byte");
}

/// A new daemon over the same folder, recovering at startup as production
/// does before it serves anything.
async fn start_a_restarted_daemon(test_directory: &Path) -> DaemonIpcService {
    let catalog = test_catalog();
    let coordinator = build_test_coordinator(
        test_directory,
        Arc::clone(&catalog),
        Arc::new(ScriptedHub::new()),
    );
    coordinator
        .recover_startup_state()
        .await
        .expect("startup recovery should accept the interrupted job");
    start_ipc_service(test_directory, catalog, coordinator).await
}

/// The daemon quits mid-download (the menu app is closed, the Mac sleeps and
/// the daemon is restarted): a new daemon over the same folder must still hold
/// the job, and continue it when asked again.
#[tokio::test]
async fn should_resume_after_a_daemon_restart_through_download_start() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        interrupt_a_running_download(test_directory.path()).await;
        let restarted_service = start_a_restarted_daemon(test_directory.path()).await;

        let job_before_asking = ask(&restarted_service, DaemonRequest::DownloadStatus).await;
        let resume_response = ask(&restarted_service, download_start()).await;

        let DaemonResponse::DownloadStatus {
            job: Some(recovered_job),
        } = job_before_asking
        else {
            panic!("the restarted daemon must still hold the interrupted job");
        };
        assert_eq!(recovered_job.huggingface_id, REPOSITORY_ID);
        assert_download_started(&resume_response, "the restarted daemon must resume the job");
        wait_until_published(test_directory.path()).await;
        wait_until_no_job(&restarted_service).await;
        restarted_service.shutdown().await.expect("clean shutdown");
    })
    .await
    .expect("the restart-resume journey should remain bounded");
}

/// Two terminals on a fresh install both run the command at the same moment:
/// both see no job, one start wins, and the other must attach to it instead of
/// failing the person's first command with "another download is active".
#[tokio::test]
async fn should_let_two_simultaneous_requests_share_one_fresh_download() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let catalog = test_catalog();
        let coordinator = build_test_coordinator(
            test_directory.path(),
            Arc::clone(&catalog),
            Arc::new(ScriptedHub::delayed(Duration::from_millis(250))),
        );
        let service = start_ipc_service(test_directory.path(), catalog, coordinator).await;

        let (first_response, second_response) = tokio::join!(
            ask(&service, download_start()),
            ask(&service, download_start())
        );

        assert_download_started(&first_response, "the first simultaneous request");
        assert_download_started(&second_response, "the second simultaneous request");
        wait_until_published(test_directory.path()).await;
        wait_until_no_job(&service).await;
        service.shutdown().await.expect("clean shutdown");
    })
    .await
    .expect("the simultaneous-start journey should remain bounded");
}

/// Polls the job the way the CLI does until it reports `Failed`.
async fn wait_until_failed(service: &DaemonIpcService) {
    for _poll_attempt in 0..STATE_POLL_ATTEMPTS {
        if let DaemonResponse::DownloadStatus { job: Some(job) } =
            ask(service, DaemonRequest::DownloadStatus).await
            && job.state == "failed"
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the download never failed");
}

/// Polls until the job names `huggingface_id`. The started task records its
/// job a moment after the start is answered, so one poll is not enough.
async fn wait_until_job_names(service: &DaemonIpcService, huggingface_id: &str) {
    for _poll_attempt in 0..STATE_POLL_ATTEMPTS {
        if let DaemonResponse::DownloadStatus { job: Some(job) } =
            ask(service, DaemonRequest::DownloadStatus).await
            && job.huggingface_id == huggingface_id
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the download of {huggingface_id} never held the Library");
}

/// The reported journey (#1897) through the CLI's own door: one model's
/// download fails its checksum, and `astronomical download` of another model
/// must start instead of answering that a download is already active.
#[tokio::test]
async fn should_start_another_model_over_a_failed_download_through_download_start() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let catalog = two_model_catalog();
        let coordinator = build_test_coordinator(
            test_directory.path(),
            Arc::clone(&catalog),
            Arc::new(ScriptedHub::checksum_mismatch()),
        );
        let service = start_ipc_service(test_directory.path(), catalog, coordinator).await;
        assert_download_started(&ask(&service, download_start()).await, "the first model");
        wait_until_failed(&service).await;

        let other_model_response = ask(
            &service,
            DaemonRequest::DownloadStart {
                model_id: OTHER_REPOSITORY_ID.to_owned(),
            },
        )
        .await;

        assert_eq!(
            other_model_response,
            DaemonResponse::DownloadStarted {
                huggingface_id: OTHER_REPOSITORY_ID.to_owned()
            },
            "another model must replace the failed download"
        );
        wait_until_job_names(&service, OTHER_REPOSITORY_ID).await;
        service.shutdown().await.expect("clean shutdown");
    })
    .await
    .expect("the failed-then-other-model journey should remain bounded");
}

/// The upgrade path: a download failed under the previous daemon (or the previous version,
/// whose job file format is the same), the Mac restarts or the app updates, and the person asks
/// for another model. The new daemon must recover the failed job at startup and still let the
/// other model start and publish.
#[tokio::test]
async fn should_start_another_model_over_a_failed_download_left_before_a_restart() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let first_catalog = two_model_catalog();
        let first_coordinator = build_test_coordinator(
            test_directory.path(),
            Arc::clone(&first_catalog),
            Arc::new(ScriptedHub::checksum_mismatch()),
        );
        let first_service =
            start_ipc_service(test_directory.path(), first_catalog, first_coordinator).await;
        assert_download_started(
            &ask(&first_service, download_start()).await,
            "the first model",
        );
        wait_until_failed(&first_service).await;
        first_service
            .shutdown()
            .await
            .expect("the first daemon shuts down");

        let restarted_catalog = two_model_catalog();
        let restarted_coordinator = build_test_coordinator(
            test_directory.path(),
            Arc::clone(&restarted_catalog),
            Arc::new(ScriptedHub::new()),
        );
        restarted_coordinator
            .recover_startup_state()
            .await
            .expect("startup recovery should accept the failed job");
        let restarted_service = start_ipc_service(
            test_directory.path(),
            restarted_catalog,
            restarted_coordinator,
        )
        .await;
        wait_until_failed(&restarted_service).await;

        let other_model_response = ask(
            &restarted_service,
            DaemonRequest::DownloadStart {
                model_id: OTHER_REPOSITORY_ID.to_owned(),
            },
        )
        .await;

        assert_eq!(
            other_model_response,
            DaemonResponse::DownloadStarted {
                huggingface_id: OTHER_REPOSITORY_ID.to_owned()
            }
        );
        wait_until_job_names(&restarted_service, OTHER_REPOSITORY_ID).await;
        restarted_service.shutdown().await.expect("clean shutdown");
    })
    .await
    .expect("the restart journey should remain bounded");
}

/// Startup clears the empty staging folders earlier versions left behind (#1929), and keeps a
/// paused download's staged bytes so it can still be resumed.
#[tokio::test]
async fn should_clear_old_empty_staging_folders_at_startup() {
    timeout(Duration::from_secs(10), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        interrupt_a_running_download(test_directory.path()).await;
        let leftover_directory = test_directory.path().join("models/.incomplete/araysuter");
        std::fs::create_dir_all(&leftover_directory)
            .expect("the leftover folder should be created");

        let restarted_service = start_a_restarted_daemon(test_directory.path()).await;

        assert!(
            !leftover_directory.exists(),
            "the empty leftover folder is removed"
        );
        let staged_byte_count = staged_byte_count(
            &test_directory
                .path()
                .join(format!("models/.incomplete/{REPOSITORY_ID}")),
        );
        assert!(
            staged_byte_count > 0,
            "the paused download keeps its staged bytes"
        );
        restarted_service.shutdown().await.expect("clean shutdown");
    })
    .await
    .expect("the startup cleanup journey should remain bounded");
}

/// Total bytes staged under `staging_directory`, counted recursively.
fn staged_byte_count(staging_directory: &Path) -> u64 {
    let mut total_bytes = 0;
    for staging_entry in std::fs::read_dir(staging_directory).expect("staging should be readable") {
        let staging_entry = staging_entry.expect("staging entry");
        let entry_metadata = staging_entry.metadata().expect("staging entry metadata");
        total_bytes += if entry_metadata.is_dir() {
            staged_byte_count(&staging_entry.path())
        } else {
            entry_metadata.len()
        };
    }
    total_bytes
}

/// Startup cleanup is housekeeping: when it cannot run (a symlinked staging folder, which the
/// store refuses to follow), startup still succeeds.
#[tokio::test]
async fn should_start_even_when_the_staging_folder_cannot_be_tidied() {
    let test_directory = tempfile::tempdir().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    let outside_directory = test_directory.path().join("outside");
    std::fs::create_dir_all(&models_directory).expect("the models folder should be created");
    std::fs::create_dir_all(&outside_directory).expect("the outside folder should be created");
    std::os::unix::fs::symlink(&outside_directory, models_directory.join(".incomplete"))
        .expect("the symlink should be created");
    let coordinator = build_test_coordinator(
        test_directory.path(),
        test_catalog(),
        Arc::new(ScriptedHub::new()),
    );

    coordinator
        .recover_startup_state()
        .await
        .expect("a failed tidy-up must not stop startup");

    assert!(outside_directory.is_dir());
}
