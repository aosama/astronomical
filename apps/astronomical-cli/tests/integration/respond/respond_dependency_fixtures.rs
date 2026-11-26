//! The dependency set and default stub daemon every `respond` journey starts
//! from. Zero test cases live here.

use std::{io::Write, path::PathBuf};

use astronomical_cli::RespondDependencies;
use config::AstronomicalRuntimeInstance;

use crate::integration::support::environment::{DOWNLOAD_POLL_INTERVAL, TEST_TIMEOUT};
use crate::integration::support::stub_daemon::{StubDaemonConfig, StubInstalledModel};

/// Journeys that need a different silence bound or poll interval overwrite
/// those fields on the returned value instead of rebuilding the whole set.
pub(super) fn respond_dependencies<'a>(
    candidate_socket_paths: Vec<PathBuf>,
    stdout: &'a mut Vec<u8>,
    stderr: &'a mut Vec<u8>,
) -> RespondDependencies<'a> {
    RespondDependencies {
        candidate_socket_paths,
        runtime_instance: AstronomicalRuntimeInstance::Development,
        stdout: stdout as &mut dyn Write,
        stderr: stderr as &mut dyn Write,
        request_timeout: TEST_TIMEOUT,
        download_stall_bound: TEST_TIMEOUT,
        download_poll_interval: DOWNLOAD_POLL_INTERVAL,
        daemon_work_poll_interval: DOWNLOAD_POLL_INTERVAL,
        model_load_stall_bound: TEST_TIMEOUT,
    }
}

/// Stub that has one resident chat model and names it as the effective
/// default: the no-flag resolution path.
pub(super) fn resident_default_stub_config() -> StubDaemonConfig {
    StubDaemonConfig {
        installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
        default_model_id: Some("test/local-chatter".to_owned()),
        chat_fragments: vec!["Hello".to_owned(), " world".to_owned()],
        ..Default::default()
    }
}
