//! The `astronomical models` verb: list, supported, default, download. The
//! daemon side is the shared stub speaking the real framed protocol.

use std::{io::Write, path::PathBuf};

use astronomical_cli::{CliCommand, ModelsCommand, ModelsDependencies, UsageError, run_models};
use tokio::time::timeout;

use super::stub_daemon::{
    StubCatalogEntry, StubDaemonConfig, StubInstalledModel, spawn_stub_daemon, stub_download_job,
};
use super::test_support::{
    DOWNLOAD_POLL_INTERVAL, SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory, parse,
};

fn models_dependencies<'a>(
    candidate_socket_paths: Vec<PathBuf>,
    stdout: &'a mut Vec<u8>,
    stderr: &'a mut Vec<u8>,
) -> ModelsDependencies<'a> {
    ModelsDependencies {
        candidate_socket_paths,
        stdout: stdout as &mut dyn Write,
        stderr: stderr as &mut dyn Write,
        request_timeout: TEST_TIMEOUT,
        download_stage_bound: TEST_TIMEOUT,
        download_poll_interval: DOWNLOAD_POLL_INTERVAL,
    }
}

#[test]
fn should_parse_models_subcommands() {
    assert_eq!(
        parse(&["models", "list"]),
        Ok(CliCommand::Models(ModelsCommand::List))
    );
    assert_eq!(
        parse(&["models", "supported"]),
        Ok(CliCommand::Models(ModelsCommand::Supported))
    );
    assert_eq!(
        parse(&["models", "default"]),
        Ok(CliCommand::Models(ModelsCommand::Default {
            model_id: None
        }))
    );
    assert_eq!(
        parse(&["models", "default", "test/model"]),
        Ok(CliCommand::Models(ModelsCommand::Default {
            model_id: Some("test/model".to_owned())
        }))
    );
    assert_eq!(
        parse(&["models", "download", "test/model"]),
        Ok(CliCommand::Models(ModelsCommand::Download {
            model_id: "test/model".to_owned()
        }))
    );
}

#[test]
fn should_reject_models_without_a_subcommand_as_a_usage_error() {
    assert!(matches!(
        parse(&["models"]),
        Err(UsageError::ModelsSubcommandRequired)
    ));
}

#[test]
fn should_reject_models_with_an_unknown_subcommand_as_a_usage_error() {
    assert!(matches!(
        parse(&["models", "reboot"]),
        Err(UsageError::UnknownModelsSubcommand(subcommand)) if subcommand == "reboot"
    ));
}

#[test]
fn should_reject_models_download_without_a_model_id_as_a_usage_error() {
    assert!(matches!(
        parse(&["models", "download"]),
        Err(UsageError::ModelsDownloadModelRequired)
    ));
}

/// Stub with two installed models (one resident) plus a three-entry catalog
/// covering the ready / downloading / absent states.
fn catalog_stub_config() -> StubDaemonConfig {
    StubDaemonConfig {
        installed_models: vec![
            StubInstalledModel::chat("test/local-chatter", true),
            StubInstalledModel::embeddings("test/local-embedder", false),
        ],
        default_model_id: Some("test/local-chatter".to_owned()),
        catalog_entries: vec![
            StubCatalogEntry::chat("test/ready-model", "ready-model", true),
            StubCatalogEntry {
                download_state: Some("downloading".to_owned()),
                ..StubCatalogEntry::chat("test/half-model", "half-model", false)
            },
            StubCatalogEntry::chat("test/absent-model", "absent-model", false),
        ],
        ..Default::default()
    }
}

#[tokio::test]
async fn should_list_installed_models_with_the_resident_marker() {
    let test_directory = fresh_test_directory("models", "list");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), catalog_stub_config());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut models_dependencies = models_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let models_outcome = timeout(
        TEST_TIMEOUT,
        run_models(&ModelsCommand::List, &mut models_dependencies),
    )
    .await
    .expect("the models journey should finish inside the test timeout");
    models_outcome.expect("models list should complete against the stub daemon");
    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout.contains("test/local-chatter"),
        "the list should include the chat model: {rendered_stdout:?}"
    );
    assert!(
        rendered_stdout.contains("test/local-embedder"),
        "the list should include the embeddings model: {rendered_stdout:?}"
    );
    let resident_line = rendered_stdout
        .lines()
        .find(|line| line.contains("test/local-chatter"))
        .expect("the resident model line should exist");
    assert!(
        resident_line.starts_with('*'),
        "the resident model line should carry the marker: {rendered_stdout:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_render_the_catalog_with_local_states() {
    let test_directory = fresh_test_directory("models", "supported");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), catalog_stub_config());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut models_dependencies = models_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let models_outcome = timeout(
        TEST_TIMEOUT,
        run_models(&ModelsCommand::Supported, &mut models_dependencies),
    )
    .await
    .expect("the models journey should finish inside the test timeout");
    models_outcome.expect("models supported should complete against the stub daemon");
    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout.contains("ready-model") && rendered_stdout.contains("ready"),
        "a ready entry should render as ready: {rendered_stdout:?}"
    );
    assert!(
        rendered_stdout.contains("half-model") && rendered_stdout.contains("downloading"),
        "an entry with an active download should render its state: {rendered_stdout:?}"
    );
    assert!(
        rendered_stdout.contains("absent-model") && rendered_stdout.contains("not on this Mac"),
        "an absent entry should render as not on this Mac: {rendered_stdout:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_show_and_set_the_default_model() {
    let test_directory = fresh_test_directory("models", "default");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), catalog_stub_config());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();

    // Show the current effective default.
    {
        let mut models_dependencies =
            models_dependencies(vec![socket_path.clone()], &mut stdout, &mut stderr);
        let models_outcome = timeout(
            TEST_TIMEOUT,
            run_models(
                &ModelsCommand::Default { model_id: None },
                &mut models_dependencies,
            ),
        )
        .await
        .expect("the models journey should finish inside the test timeout");
        models_outcome.expect("models default (show) should complete");
        assert!(
            String::from_utf8_lossy(&stdout).contains("default model: test/local-chatter"),
            "the show should print the daemon's effective default: {stdout:?}"
        );
    }
    // Persist a new default and confirm the show reflects it. The id must
    // come from the catalog: setting a default fetches the model first.
    stdout.clear();
    {
        let mut models_dependencies =
            models_dependencies(vec![socket_path.clone()], &mut stdout, &mut stderr);
        let models_outcome = timeout(
            TEST_TIMEOUT,
            run_models(
                &ModelsCommand::Default {
                    model_id: Some("test/ready-model".to_owned()),
                },
                &mut models_dependencies,
            ),
        )
        .await
        .expect("the models journey should finish inside the test timeout");
        models_outcome.expect("models default (set) should complete");
        assert!(
            String::from_utf8_lossy(&stdout).contains("default model: test/ready-model"),
            "the set should echo the persisted default: {stdout:?}"
        );
    }
    stdout.clear();
    {
        let mut models_dependencies =
            models_dependencies(vec![socket_path.clone()], &mut stdout, &mut stderr);
        let models_outcome = timeout(
            TEST_TIMEOUT,
            run_models(
                &ModelsCommand::Default { model_id: None },
                &mut models_dependencies,
            ),
        )
        .await
        .expect("the models journey should finish inside the test timeout");
        models_outcome.expect("models default (show again) should complete");
        assert!(
            String::from_utf8_lossy(&stdout).contains("default model: test/ready-model"),
            "the show after the set should reflect the new default: {stdout:?}"
        );
    }
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_download_a_missing_model_before_persisting_the_default() {
    let test_directory = fresh_test_directory("models", "default-downloads");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let missing_entry = StubCatalogEntry::chat("test/downloaded-model", "downloaded-model", false);
    let ready_entry = StubCatalogEntry::chat("test/downloaded-model", "downloaded-model", true);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            catalog_entries: vec![missing_entry],
            catalog_entries_after_download: Some(vec![ready_entry]),
            download_jobs: vec![
                Some(stub_download_job(
                    "test/downloaded-model",
                    "downloading",
                    500_000_000,
                    2_000_000_000,
                    None,
                )),
                None,
            ],
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut models_dependencies = models_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let models_outcome = timeout(
        TEST_TIMEOUT,
        run_models(
            &ModelsCommand::Default {
                model_id: Some("test/downloaded-model".to_owned()),
            },
            &mut models_dependencies,
        ),
    )
    .await
    .expect("the models journey should finish inside the test timeout");
    models_outcome.expect("setting the default should download the missing model first");
    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout.contains("default model: test/downloaded-model"),
        "the default should be persisted after the download: {rendered_stdout:?}"
    );
    let rendered_stderr = String::from_utf8_lossy(&stderr);
    assert!(
        rendered_stderr.contains("downloading test/downloaded-model"),
        "the download should report live progress on stderr: {rendered_stderr:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_download_a_missing_model_with_live_progress() {
    let test_directory = fresh_test_directory("models", "download");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let missing_entry = StubCatalogEntry::chat("test/downloaded-model", "downloaded-model", false);
    let ready_entry = StubCatalogEntry::chat("test/downloaded-model", "downloaded-model", true);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            catalog_entries: vec![missing_entry.clone()],
            catalog_entries_after_download: Some(vec![ready_entry]),
            download_jobs: vec![
                Some(stub_download_job(
                    "test/downloaded-model",
                    "downloading",
                    500_000_000,
                    2_000_000_000,
                    None,
                )),
                None,
            ],
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut models_dependencies = models_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let models_outcome = timeout(
        TEST_TIMEOUT,
        run_models(
            &ModelsCommand::Download {
                model_id: "test/downloaded-model".to_owned(),
            },
            &mut models_dependencies,
        ),
    )
    .await
    .expect("the models journey should finish inside the test timeout");
    models_outcome.expect("the download journey should complete");
    let rendered_stderr = String::from_utf8_lossy(&stderr);
    assert!(
        rendered_stderr.contains("test/downloaded-model is available"),
        "the download journey should announce availability on stderr: {rendered_stderr:?}"
    );
    assert!(
        rendered_stderr.contains("0.5 GB / 2 GB"),
        "the live progress should report decimal GB: {rendered_stderr:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_fail_to_download_a_model_outside_the_catalog() {
    let test_directory = fresh_test_directory("models", "download-unknown");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), catalog_stub_config());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut models_dependencies = models_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let models_outcome = timeout(
        TEST_TIMEOUT,
        run_models(
            &ModelsCommand::Download {
                model_id: "test/not-in-catalog".to_owned(),
            },
            &mut models_dependencies,
        ),
    )
    .await
    .expect("the models journey should finish inside the test timeout");
    assert!(
        matches!(&models_outcome,
            Err(astronomical_cli::ModelsError::ModelUnavailable { reason })
                if reason.contains("test/not-in-catalog")),
        "downloading a model outside the catalog must fail with the model id: {models_outcome:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}
