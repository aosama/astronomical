//! `respond` while the daemon loads the model: it sends nothing on the generation's
//! connection until the load ends, and reading the weights can take minutes.
//! A load report counts as hearing from the daemon; only silence, or a report that
//! stays put past the stall bound, means it has gone away.

use std::{io::Write, path::PathBuf, time::Duration};

use astronomical_cli::{RespondDependencies, RespondError};
use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::ModelLoadInFlight;
use tokio::time::timeout;

use crate::integration::support::environment::{
    DOWNLOAD_POLL_INTERVAL, SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory, respond_arguments,
};
use crate::integration::support::stub_daemon::{
    StubDaemonConfig, StubInstalledModel, spawn_stub_daemon,
};

/// Short enough that a load outlasting it proves the bound is on silence, and long
/// enough for the journey's own model and catalog reads on a loaded machine.
const SILENCE_BOUND: Duration = Duration::from_millis(500);
const LOAD_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// Several silence bounds long, so an answer arriving proves the wait was extended.
const LOAD_DURATION_PAST_THE_BOUND: Duration = Duration::from_millis(1600);
/// Far longer than any test waits, standing in for a load that never finishes.
const LOAD_THAT_NEVER_FINISHES: Duration = Duration::from_secs(60);

fn load_dependencies<'a>(
    candidate_socket_paths: Vec<PathBuf>,
    stdout: &'a mut Vec<u8>,
    stderr: &'a mut Vec<u8>,
) -> RespondDependencies<'a> {
    RespondDependencies {
        candidate_socket_paths,
        runtime_instance: AstronomicalRuntimeInstance::Development,
        stdout: stdout as &mut dyn Write,
        stderr: stderr as &mut dyn Write,
        request_timeout: SILENCE_BOUND,
        download_stall_bound: TEST_TIMEOUT,
        download_poll_interval: DOWNLOAD_POLL_INTERVAL,
        model_load_poll_interval: LOAD_POLL_INTERVAL,
        model_load_stall_bound: TEST_TIMEOUT,
    }
}

fn stub_config_loading_model(
    chat_first_frame_delay: Duration,
    model_load_reports: Vec<ModelLoadInFlight>,
) -> StubDaemonConfig {
    StubDaemonConfig {
        installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
        default_model_id: Some("test/local-chatter".to_owned()),
        chat_fragments: vec!["Hello".to_owned()],
        chat_first_frame_delay,
        model_load_reports,
        ..Default::default()
    }
}

struct RespondJourney {
    respond_outcome: Result<(), RespondError>,
    stdout: String,
    stderr: String,
}

async fn run_respond_against(test_name: &str, stub_config: StubDaemonConfig) -> RespondJourney {
    run_respond_with_stall_bound(test_name, stub_config, TEST_TIMEOUT).await
}

async fn run_respond_with_stall_bound(
    test_name: &str,
    stub_config: StubDaemonConfig,
    model_load_stall_bound: Duration,
) -> RespondJourney {
    let test_directory = fresh_test_directory("load", test_name);
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), stub_config);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies = load_dependencies(vec![socket_path], &mut stdout, &mut stderr);
    respond_dependencies.model_load_stall_bound = model_load_stall_bound;

    let respond_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_respond(
            &respond_arguments("Say hello", None, false),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");

    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
    RespondJourney {
        respond_outcome,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
    }
}

fn stuck_load_report() -> ModelLoadInFlight {
    ModelLoadInFlight::for_model("test/local-chatter")
}

/// Weights read under memory pressure stay reported for minutes while the daemon is
/// healthy: the false failure this wait exists to end.
#[tokio::test]
async fn should_keep_waiting_while_a_report_stays_put_inside_the_stall_bound() {
    let journey = run_respond_against(
        "patient",
        stub_config_loading_model(LOAD_DURATION_PAST_THE_BOUND, vec![stuck_load_report()]),
    )
    .await;

    journey
        .respond_outcome
        .expect("a load still being reported must reach its answer");
    assert_eq!(journey.stdout, "Hello");
    assert!(
        journey.stderr.contains("Loading the model"),
        "the wait should say what it is waiting for: {:?}",
        journey.stderr
    );
    assert!(
        journey.stderr.ends_with('\n'),
        "the live line should end before the answer finishes: {:?}",
        journey.stderr
    );
}

#[tokio::test]
async fn should_give_up_when_a_report_stays_put_past_the_stall_bound() {
    let journey = run_respond_with_stall_bound(
        "stuck",
        stub_config_loading_model(LOAD_THAT_NEVER_FINISHES, vec![stuck_load_report()]),
        Duration::from_millis(300),
    )
    .await;

    assert!(
        matches!(
            journey.respond_outcome,
            Err(RespondError::DaemonStoppedResponding)
        ),
        "a daemon that repeats one report forever has wedged: {:?}",
        journey.respond_outcome
    );
    assert!(journey.stderr.ends_with('\n'), "{:?}", journey.stderr);
}

#[tokio::test]
async fn should_give_up_on_silence_when_nothing_reports_a_load() {
    let journey = run_respond_against(
        "silent",
        stub_config_loading_model(LOAD_THAT_NEVER_FINISHES, Vec::new()),
    )
    .await;

    assert!(
        matches!(
            journey.respond_outcome,
            Err(RespondError::DaemonStoppedResponding)
        ),
        "with no load reported, the bound must apply as before: {:?}",
        journey.respond_outcome
    );
    assert!(
        journey.stderr.is_empty(),
        "nothing was loading, so nothing should claim to be: {:?}",
        journey.stderr
    );
}

#[tokio::test]
async fn should_say_nothing_about_loading_for_an_answer_with_no_load() {
    let journey = run_respond_against(
        "fast",
        stub_config_loading_model(Duration::ZERO, Vec::new()),
    )
    .await;

    journey
        .respond_outcome
        .expect("an answer with no load should complete");
    assert_eq!(journey.stdout, "Hello");
    assert!(journey.stderr.is_empty(), "{:?}", journey.stderr);
}
