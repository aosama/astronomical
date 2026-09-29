//! Helpers shared by the hermetic CLI verb tests: argument parsing, short
//! timeouts, and uniquely named per-test temporary directories.

use std::{
    ffi::OsString,
    path::PathBuf,
    process,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use astronomical_cli::{CliCommand, UsageError, parse_command};

/// Bound for every protocol stage in these tests: generous for a stub
/// daemon on loopback, short enough that a hung exchange fails the test
/// well inside the repo's 120-second ceiling.
pub const TEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Wait between download status polls: fast so download-wait tests stay
/// snappy against the stub daemon.
pub const DOWNLOAD_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Socket file name the stub daemon binds inside its test directory.
pub const SOCKET_FILE_NAME: &str = "ipc.sock";

/// Parses arguments as the CLI would see them after the binary name.
pub fn parse(arguments: &[&str]) -> Result<CliCommand, UsageError> {
    let process_arguments =
        std::iter::once(OsString::from("astronomical")).chain(arguments.iter().map(OsString::from));
    parse_command(process_arguments)
}

/// A unique temporary directory per test: pid plus nanos so parallel tests
/// and repeated runs never collide. The caller removes it when done. Names
/// stay short because unix socket paths must fit in `sockaddr_un`.
pub fn fresh_test_directory(verb_name: &str, test_name: &str) -> PathBuf {
    let nanos_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should provide time after the epoch")
        .as_nanos();
    let test_directory = std::env::temp_dir().join(format!(
        "ast-{verb_name}-{}-{}-{test_name}",
        process::id(),
        nanos_since_epoch % 1_000_000_000
    ));
    std::fs::create_dir_all(&test_directory).expect("the test directory should be creatable");
    test_directory
}
