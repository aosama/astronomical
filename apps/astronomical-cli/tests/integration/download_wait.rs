//! How the CLI follows a model download: live rate and time remaining, a wait
//! that ends on lack of progress and never on a fixed clock, and failure text
//! that says how to resume. Driven through `models download`, the plain form of
//! the wait that `image`, `respond` and `embed` all share, against a stub
//! daemon that scripts the job the daemon would report on each poll.

use std::{path::PathBuf, time::Duration};

use astronomical_cli::{ModelsCommand, ModelsDependencies, ModelsError};
use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::DaemonDownloadJob;
use tokio::time::timeout;

use crate::integration::support::environment::{
    SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory,
};
use crate::integration::support::stub_daemon::{
    StubCatalogEntry, StubDaemonConfig, spawn_stub_daemon, stub_download_job,
};

const MODEL: &str = "test/downloaded-model";
const MEGABYTE: u64 = 1_048_576;
/// The stub catalog reports every entry as about 2 GB (1.86 in the CLI's
/// 2^30 gigabytes).
const TOTAL_BYTES: u64 = 2_000_000_000;
const POLL_INTERVAL: Duration = Duration::from_millis(20);

fn job(state: &str, bytes_completed: u64) -> Option<DaemonDownloadJob> {
    Some(stub_download_job(
        MODEL,
        state,
        bytes_completed,
        TOTAL_BYTES,
        None,
    ))
}

struct DownloadRun {
    outcome: Result<(), ModelsError>,
    stderr: String,
}

/// Runs `models download` against a stub that reports `download_jobs` on
/// successive polls and then flips the catalog to ready, so a script that
/// ends in `None` is a download that completed.
async fn run_download(
    test_name: &str,
    download_jobs: Vec<Option<DaemonDownloadJob>>,
    stall_bound: Duration,
) -> DownloadRun {
    let test_directory = fresh_test_directory("dlwait", test_name);
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            catalog_entries: vec![StubCatalogEntry::chat(MODEL, "downloaded-model", false)],
            catalog_entries_after_download: Some(vec![StubCatalogEntry::chat(
                MODEL,
                "downloaded-model",
                true,
            )]),
            download_jobs,
            ..Default::default()
        },
    );
    let (outcome, stderr) = follow_download(socket_path, stall_bound).await;
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
    DownloadRun { outcome, stderr }
}

async fn follow_download(
    socket_path: PathBuf,
    stall_bound: Duration,
) -> (Result<(), ModelsError>, String) {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut models_dependencies = ModelsDependencies {
        candidate_socket_paths: vec![socket_path],
        runtime_instance: AstronomicalRuntimeInstance::Development,
        stdout: &mut stdout,
        stderr: &mut stderr,
        request_timeout: TEST_TIMEOUT,
        download_stall_bound: stall_bound,
        download_poll_interval: POLL_INTERVAL,
    };
    let outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_models(
            &ModelsCommand::Download {
                model_id: MODEL.to_owned(),
            },
            &mut models_dependencies,
        ),
    )
    .await
    .expect("the download journey should finish inside the test timeout");
    (outcome, String::from_utf8_lossy(&stderr).into_owned())
}

fn failure_text(download_run: &DownloadRun) -> String {
    match &download_run.outcome {
        Err(models_error) => models_error.to_string(),
        Ok(()) => panic!("the download was expected to fail"),
    }
}

#[tokio::test]
async fn should_announce_the_download_size_before_any_bytes_move() {
    let download_run = run_download(
        "announce",
        vec![job("downloading", MEGABYTE), None],
        Duration::from_secs(5),
    )
    .await;

    download_run.outcome.expect("the download completes");
    assert!(
        download_run
            .stderr
            .contains("downloading test/downloaded-model (1.86 GB)"),
        "{}",
        download_run.stderr
    );
}

#[tokio::test]
async fn should_show_the_transfer_rate_and_time_remaining_while_bytes_flow() {
    let mut download_jobs = Vec::new();
    for step in 1..=8_u64 {
        download_jobs.push(job("downloading", step * 100 * MEGABYTE));
    }
    download_jobs.push(None);

    let download_run = run_download("rate-eta", download_jobs, Duration::from_secs(5)).await;

    download_run.outcome.expect("the download completes");
    assert!(
        download_run.stderr.contains("/s"),
        "{}",
        download_run.stderr
    );
    assert!(
        download_run.stderr.contains("left"),
        "{}",
        download_run.stderr
    );
    assert!(
        download_run.stderr.contains("%)"),
        "{}",
        download_run.stderr
    );
}

/// The reason this exists: a download far longer than the stall bound, which
/// keeps moving, must finish. The earlier fixed two-minute cap ended exactly
/// these downloads.
#[tokio::test]
async fn should_let_a_slow_download_finish_however_long_it_takes_while_it_keeps_moving() {
    let mut download_jobs = Vec::new();
    for step in 1..=40_u64 {
        download_jobs.push(job("downloading", step * MEGABYTE));
    }
    download_jobs.push(None);
    let stall_bound = Duration::from_millis(150);

    let download_run = run_download("slow-but-moving", download_jobs, stall_bound).await;

    // 40 polls at 20 ms is about 800 ms, over five times the stall bound.
    download_run
        .outcome
        .expect("a moving download must not be cut off");
}

#[tokio::test]
async fn should_give_up_when_the_bytes_stop_and_say_how_to_continue() {
    let download_run = run_download(
        "stalled",
        vec![job("downloading", 500 * MEGABYTE)],
        Duration::from_millis(100),
    )
    .await;

    let failure = failure_text(&download_run);

    assert!(failure.contains("made no progress"), "{failure}");
    assert!(failure.contains("while downloading"), "{failure}");
    assert!(
        failure.contains("run the same command again to continue"),
        "{failure}"
    );
    assert!(failure.contains("bytes already on disk"), "{failure}");
}

#[tokio::test]
async fn should_name_a_paused_download_when_it_stops_there() {
    let download_run = run_download(
        "paused",
        vec![job("paused", 500 * MEGABYTE)],
        Duration::from_millis(100),
    )
    .await;

    let failure = failure_text(&download_run);

    assert!(failure.contains("while paused"), "{failure}");
}

#[tokio::test]
async fn should_give_verifying_more_patience_than_downloading() {
    let mut verifying_jobs = Vec::new();
    for _poll in 0..12 {
        verifying_jobs.push(job("verifying", TOTAL_BYTES));
    }
    verifying_jobs.push(None);

    // 12 silent polls is about 240 ms: past the 50 ms stall bound, inside the
    // tenfold bound verifying gets.
    let download_run = run_download("verifying", verifying_jobs, Duration::from_millis(50)).await;

    download_run
        .outcome
        .expect("a long checksum must not be mistaken for a stall");
}

#[tokio::test]
async fn should_tell_the_user_to_rerun_to_resume_after_a_failed_download() {
    let failed_job = Some(stub_download_job(
        MODEL,
        "failed",
        700 * MEGABYTE,
        TOTAL_BYTES,
        Some("the connection was lost"),
    ));

    let download_run = run_download("failed", vec![failed_job], Duration::from_secs(5)).await;

    let failure = failure_text(&download_run);
    assert!(failure.contains("the connection was lost"), "{failure}");
    assert!(
        failure.contains("run the same command again to resume"),
        "{failure}"
    );
}

#[tokio::test]
async fn should_survive_the_byte_count_moving_backwards_after_a_daemon_restart() {
    // The daemon restarted mid-download and reconciled against the staged
    // files, so the reported byte count dropped before climbing again.
    let download_jobs = vec![
        job("downloading", 900 * MEGABYTE),
        job("downloading", 1_000 * MEGABYTE),
        job("downloading", 800 * MEGABYTE),
        job("downloading", 900 * MEGABYTE),
        job("downloading", 1_000 * MEGABYTE),
        None,
    ];

    let download_run = run_download("backwards", download_jobs, Duration::from_secs(5)).await;

    download_run
        .outcome
        .expect("the reconciled byte count is still progress");
    assert!(
        download_run.stderr.contains("%)"),
        "{}",
        download_run.stderr
    );
}

#[tokio::test]
async fn should_not_leave_the_end_of_a_longer_line_behind_a_shorter_one() {
    let download_jobs = vec![
        job("downloading", 100 * MEGABYTE),
        job("downloading", 200 * MEGABYTE),
        job("verifying", TOTAL_BYTES),
        None,
    ];

    let download_run = run_download("padding", download_jobs, Duration::from_secs(5)).await;

    download_run.outcome.expect("the download completes");
    let mut line_lengths = Vec::new();
    for progress_line in download_run.stderr.split('\r') {
        let trimmed_line = progress_line.trim_end_matches('\n');
        if trimmed_line.contains("GB /") {
            line_lengths.push(trimmed_line.chars().count());
        }
    }
    assert!(line_lengths.len() >= 3, "{line_lengths:?}");
    let longest = line_lengths.iter().max().copied().unwrap_or(0);
    assert_eq!(
        *line_lengths.last().expect("a last line"),
        longest,
        "{line_lengths:?}"
    );
}

fn stub_config(download_jobs: Vec<Option<DaemonDownloadJob>>) -> StubDaemonConfig {
    StubDaemonConfig {
        catalog_entries: vec![StubCatalogEntry::chat(MODEL, "downloaded-model", false)],
        catalog_entries_after_download: Some(vec![StubCatalogEntry::chat(
            MODEL,
            "downloaded-model",
            true,
        )]),
        download_jobs,
        ..Default::default()
    }
}

/// An app update restarts the daemon in the middle of a long transfer. The
/// wait must ride it out, and on the daemon's return ask it to continue: the
/// new daemon parks the job as paused, and the stub only reports the model
/// ready once it has been asked to download.
#[tokio::test]
async fn should_ride_out_a_daemon_restart_in_the_middle_of_a_download() {
    let test_directory = fresh_test_directory("dlwait", "restart-mid-wait");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let first_daemon = spawn_stub_daemon(
        socket_path.clone(),
        stub_config(vec![job("downloading", 100 * MEGABYTE)]),
    );

    let mut restarted_daemon = None;
    let restart_the_daemon = async {
        tokio::time::sleep(Duration::from_millis(150)).await;
        first_daemon.abort();
        let _ = first_daemon.await;
        // Long enough that several polls find nobody home.
        tokio::time::sleep(Duration::from_millis(200)).await;
        std::fs::remove_file(&socket_path).expect("the dead daemon's socket file should exist");
        restarted_daemon = Some(spawn_stub_daemon(
            socket_path.clone(),
            stub_config(vec![job("downloading", 300 * MEGABYTE), None]),
        ));
    };
    let ((), (outcome, stderr)) = tokio::join!(
        restart_the_daemon,
        follow_download(socket_path.clone(), Duration::from_secs(5))
    );
    restarted_daemon
        .expect("the daemon should have been restarted")
        .abort();
    let _ = std::fs::remove_dir_all(&test_directory);

    outcome.expect("the download should finish after the daemon comes back");
    assert!(
        stderr.contains("waiting for the daemon to come back"),
        "{stderr}"
    );
}

#[tokio::test]
async fn should_give_up_when_the_daemon_never_comes_back() {
    let test_directory = fresh_test_directory("dlwait", "daemon-gone");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let only_daemon = spawn_stub_daemon(
        socket_path.clone(),
        stub_config(vec![job("downloading", 100 * MEGABYTE)]),
    );

    let stop_the_daemon = async {
        tokio::time::sleep(Duration::from_millis(150)).await;
        only_daemon.abort();
        let _ = only_daemon.await;
    };
    let (_, (outcome, _stderr)) = tokio::join!(
        stop_the_daemon,
        follow_download(socket_path.clone(), Duration::from_millis(300))
    );
    let _ = std::fs::remove_dir_all(&test_directory);

    let failure = outcome.expect_err("a daemon that never returns must end the wait");
    assert!(
        failure.to_string().contains("daemon was unreachable"),
        "{failure}"
    );
}
