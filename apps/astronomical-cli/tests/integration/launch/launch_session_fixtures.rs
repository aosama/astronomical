//! What every harness launch journey shares: a fake harness program on a
//! scratch PATH and a `prepare_launch` call wired to a stub daemon. Zero
//! test cases live here.

use std::{
    ffi::OsString,
    io::Cursor,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

use astronomical_cli::{
    LaunchArguments, LaunchDependencies, LaunchError, PreparedLaunch, prepare_launch,
};
use tempfile::TempDir;

use crate::integration::support::stub_loopback_http::chat_models_json;

/// Bounds every call to the stub daemon, so a launch journey that waits on a
/// stub that never answers fails within seconds instead of hanging the suite.
pub const HTTP_TIMEOUT: Duration = Duration::from_secs(2);

/// Writes an executable `program_name` script into `parent_directory`, so
/// `prepare_launch` finds it on a PATH made of just that directory.
pub fn install_fake_harness(
    parent_directory: &Path,
    program_name: &str,
    script_text: &str,
) -> PathBuf {
    let program_path = parent_directory.join(program_name);
    std::fs::write(&program_path, script_text).expect("fake harness script");
    let mut permissions = std::fs::metadata(&program_path)
        .expect("fake harness metadata")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&program_path, permissions).expect("fake harness executable bit");
    program_path
}

/// The PATH value that exposes only the fake harness installed in the scratch directory.
pub fn path_value_of(temporary_directory: &TempDir) -> OsString {
    temporary_directory.path().as_os_str().to_os_string()
}

/// The library that forces the model picker: two chat models with different context windows.
pub fn two_chat_models_json() -> String {
    chat_models_json(&[
        ("library-chat-model", 131_072),
        ("second-library-chat-model", 32_768),
    ])
}

/// Runs `prepare_launch` against the stub daemon at `bind_address` with the
/// given terminal answers, and returns the prepared launch with everything it
/// wrote to stderr, which the journeys assert on.
pub fn prepare_session(
    tool_slug: Option<&str>,
    model_id: Option<&str>,
    bind_address: std::net::SocketAddr,
    path_value: OsString,
    is_interactive: bool,
    stdin_text: &str,
) -> Result<(PreparedLaunch, String), LaunchError> {
    let mut stdin = Cursor::new(stdin_text.as_bytes().to_vec());
    let mut stderr_bytes = Vec::new();
    let prepared_launch = prepare_launch(
        LaunchArguments {
            tool_slug: tool_slug.map(str::to_owned),
            model_id: model_id.map(str::to_owned),
            harness_arguments: Vec::new(),
        },
        LaunchDependencies {
            candidate_bind_addresses: vec![bind_address],
            path_value,
            is_interactive,
            stdin: &mut stdin,
            stderr: &mut stderr_bytes,
            http_timeout: HTTP_TIMEOUT,
        },
    )?;
    let stderr_text = String::from_utf8(stderr_bytes).expect("stderr utf8");
    Ok((prepared_launch, stderr_text))
}
