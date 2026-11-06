//! `embed` while the daemon has nothing to answer with yet: it sends nothing on the
//! connection while the request waits behind a generation or a model loads, and either
//! can outlast the silence bound.

use std::{io::Write, time::Duration};

use astronomical_cli::{EmbedDependencies, EmbedError};
use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::ModelLoadInFlight;
use tokio::time::timeout;

use super::{embed_arguments, resident_default_stub_config};
use crate::integration::support::environment::{
    DAEMON_WORK_POLL_INTERVAL, DOWNLOAD_POLL_INTERVAL, FIRST_FRAME_DELAY_PAST_THE_BOUND,
    FIRST_FRAME_THAT_NEVER_ARRIVES, SILENCE_BOUND, SOCKET_FILE_NAME, TEST_TIMEOUT,
    fresh_test_directory, rising_progress_counts,
};
use crate::integration::support::stub_daemon::{
    StubDaemonConfig, StubEmbeddingsOutcome, spawn_stub_daemon,
};

/// How high the count stands has no meaning, so a high one must earn nothing.
const STUCK_COUNT: u64 = 400;

struct EmbedJourney {
    embed_outcome: Result<(), EmbedError>,
    stdout: String,
    stderr: String,
}

async fn run_embed_against(test_name: &str, stub_config: StubDaemonConfig) -> EmbedJourney {
    let test_directory = fresh_test_directory("ewait", test_name);
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), stub_config);
    let mut stdin = std::io::empty();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut embed_dependencies = EmbedDependencies {
        candidate_socket_paths: vec![socket_path],
        runtime_instance: AstronomicalRuntimeInstance::Development,
        stdin: &mut stdin,
        stdout: &mut stdout as &mut dyn Write,
        stderr: &mut stderr as &mut dyn Write,
        request_timeout: SILENCE_BOUND,
        download_stall_bound: TEST_TIMEOUT,
        download_poll_interval: DOWNLOAD_POLL_INTERVAL,
        daemon_work_poll_interval: DAEMON_WORK_POLL_INTERVAL,
        model_load_stall_bound: TEST_TIMEOUT,
    };

    let embed_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_embed(
            &embed_arguments(Some("Two households"), None, None),
            &mut embed_dependencies,
        ),
    )
    .await
    .expect("the embed journey should finish inside the test timeout");

    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
    EmbedJourney {
        embed_outcome,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
    }
}

fn stub_config_answering_after(answer_delay: Duration) -> StubDaemonConfig {
    StubDaemonConfig {
        embeddings_first_frame_delay: answer_delay,
        ..resident_default_stub_config(StubEmbeddingsOutcome::Completed)
    }
}

/// The request waits behind a long generation: the false failure this wait exists to end.
#[tokio::test]
async fn should_keep_waiting_for_embeddings_while_requests_make_progress() {
    let journey = run_embed_against(
        "rising",
        StubDaemonConfig {
            request_progress_counts: rising_progress_counts(),
            ..stub_config_answering_after(FIRST_FRAME_DELAY_PAST_THE_BOUND)
        },
    )
    .await;

    journey
        .embed_outcome
        .expect("embeddings behind a moving request must reach their answer");
    assert!(
        journey.stdout.contains("\"embedding\""),
        "{:?}",
        journey.stdout
    );
    assert!(journey.stderr.is_empty(), "{:?}", journey.stderr);
}

#[tokio::test]
async fn should_keep_waiting_for_embeddings_while_a_model_loads() {
    let journey = run_embed_against(
        "load",
        StubDaemonConfig {
            model_load_reports: vec![Some(ModelLoadInFlight::for_model("test/local-embedder"))],
            ..stub_config_answering_after(FIRST_FRAME_DELAY_PAST_THE_BOUND)
        },
    )
    .await;

    journey
        .embed_outcome
        .expect("embeddings behind a reported load must reach their answer");
    assert!(
        journey.stdout.contains("\"embedding\""),
        "{:?}",
        journey.stdout
    );
    assert!(
        journey.stderr.contains("Loading the model"),
        "the wait should say what it is waiting for: {:?}",
        journey.stderr
    );
    assert!(journey.stderr.ends_with('\n'), "{:?}", journey.stderr);
}

/// The stub answers after the load ends and progress carries the wait the rest of the way.
#[tokio::test]
async fn should_end_the_load_line_when_the_load_ends_before_the_embeddings() {
    let reported_load = Some(ModelLoadInFlight::for_model("test/local-embedder"));
    let journey = run_embed_against(
        "load-ends",
        StubDaemonConfig {
            model_load_reports: vec![reported_load.clone(), reported_load, None],
            request_progress_counts: rising_progress_counts(),
            ..stub_config_answering_after(FIRST_FRAME_DELAY_PAST_THE_BOUND)
        },
    )
    .await;

    journey
        .embed_outcome
        .expect("progress after the load must reach the answer");
    assert_eq!(journey.stderr, "\rLoading the model\n");
}

/// A count that never moves says nothing about the worker, however high it is.
#[tokio::test]
async fn should_give_up_on_embeddings_when_the_count_never_changes() {
    let journey = run_embed_against(
        "never",
        StubDaemonConfig {
            request_progress_counts: vec![STUCK_COUNT],
            ..stub_config_answering_after(FIRST_FRAME_THAT_NEVER_ARRIVES)
        },
    )
    .await;

    assert!(matches!(
        journey.embed_outcome,
        Err(EmbedError::DaemonStoppedResponding)
    ));
    assert!(journey.stdout.is_empty(), "{:?}", journey.stdout);
}

/// An answer inside the bound must not poll its way into any extra output.
#[tokio::test]
async fn should_say_nothing_extra_for_embeddings_that_answer_at_once() {
    let journey = run_embed_against("fast", stub_config_answering_after(Duration::ZERO)).await;

    journey
        .embed_outcome
        .expect("embeddings with nothing to wait on should complete");
    assert!(journey.stderr.is_empty(), "{:?}", journey.stderr);
}
