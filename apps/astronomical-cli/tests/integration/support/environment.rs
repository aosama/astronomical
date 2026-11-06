//! Helpers shared by the CLI verb journeys: short timeouts and
//! uniquely named per-test temporary directories.

use std::{
    path::PathBuf,
    process,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use astronomical_cli::RespondArguments;

/// Bound for every protocol stage in these tests: generous for a stub
/// daemon on loopback, short enough that a hung exchange fails the test
/// well inside the repo's 120-second ceiling.
pub const TEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Wait between download status polls: fast so download-wait tests stay
/// snappy against the stub daemon.
pub const DOWNLOAD_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Short enough that a wait outlasting it proves the bound is on silence, and long
/// enough for a journey's own model and catalog reads on a loaded machine.
pub const SILENCE_BOUND: Duration = Duration::from_millis(500);
pub const DAEMON_WORK_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// Several silence bounds long, so an answer arriving proves the wait was extended.
pub const FIRST_FRAME_DELAY_PAST_THE_BOUND: Duration = Duration::from_millis(1600);
/// Far longer than any test waits, standing in for a first frame that never comes.
pub const FIRST_FRAME_THAT_NEVER_ARRIVES: Duration = Duration::from_secs(60);

/// A progress count that moves on every status poll, for a scripted stub to report.
/// At one count per poll, 400 rises for longer than any journey waits.
pub fn rising_progress_counts() -> Vec<u64> {
    let mut rising_counts = Vec::new();
    for scripted_count in 1..=400 {
        rising_counts.push(scripted_count);
    }
    rising_counts
}

/// Socket file name the stub daemon binds inside its test directory.
pub const SOCKET_FILE_NAME: &str = "ipc.sock";

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

/// Builds `respond` arguments with only the prompt, model, and stream mode
/// set, for tests that exercise one dimension at a time.
pub fn respond_arguments(
    prompt: &str,
    model_id: Option<&str>,
    no_stream: bool,
) -> RespondArguments {
    RespondArguments {
        prompt: prompt.to_owned(),
        images: Vec::new(),
        model_id: model_id.map(str::to_owned),
        instructions: None,
        thinking_budget: None,
        schema_path: None,
        no_stream,
    }
}

/// Builds `respond` arguments with the generation controls set: the
/// instructions and the thinking budget.
pub fn respond_arguments_with(
    prompt: &str,
    instructions: Option<&str>,
    thinking_budget: Option<u16>,
) -> RespondArguments {
    RespondArguments {
        prompt: prompt.to_owned(),
        images: Vec::new(),
        model_id: None,
        instructions: instructions.map(str::to_owned),
        thinking_budget,
        schema_path: None,
        no_stream: false,
    }
}

/// Builds `respond` arguments with the structured-output schema file set.
pub fn respond_arguments_with_schema(
    prompt: &str,
    schema_path: Option<PathBuf>,
) -> RespondArguments {
    RespondArguments {
        prompt: prompt.to_owned(),
        images: Vec::new(),
        model_id: None,
        instructions: None,
        thinking_budget: None,
        schema_path,
        no_stream: false,
    }
}
