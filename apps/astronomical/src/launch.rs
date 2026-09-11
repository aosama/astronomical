//! Launch journey: resolve tool, find the running loopback instance, pick a
//! chat model only when needed, then prepare a process-scoped harness exec.

use std::{
    ffi::OsString,
    io::{BufRead, Write},
    net::SocketAddr,
    path::PathBuf,
    time::Duration,
};

use crate::{
    arguments::LaunchArguments,
    errors::LaunchError,
    http::get_loopback_json,
    models::chat_models_from_models_document,
    opencode::{OPENCODE_CONFIG_CONTENT_VARIABLE, opencode_config_content},
    prompt::{select_chat_model, warn_if_context_window_is_narrow},
    tools::resolve_launch_tool,
};

const STATUS_PATH: &str = "/v1/status";
const MODELS_PATH: &str = "/v1/models";
pub const DEFAULT_LOOPBACK_TIMEOUT: Duration = Duration::from_secs(3);

/// Collaborators tests replace so journeys do not exec a real harness.
pub struct LaunchDependencies<'a> {
    /// Ordered loopback candidates; the first one answering status wins.
    pub candidate_bind_addresses: Vec<SocketAddr>,
    pub path_value: OsString,
    pub is_interactive: bool,
    pub stdin: &'a mut dyn BufRead,
    pub stderr: &'a mut dyn Write,
    pub http_timeout: Duration,
}

/// Process image the CLI will exec. Tests inspect this instead of replacing
/// the test process.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedLaunch {
    pub program: PathBuf,
    pub extra_environment: Vec<(String, String)>,
}

pub fn prepare_launch(
    launch_arguments: LaunchArguments,
    launch_dependencies: LaunchDependencies<'_>,
) -> Result<PreparedLaunch, LaunchError> {
    let resolved_launch_tool = resolve_launch_tool(
        launch_arguments.tool_slug.as_deref(),
        &launch_dependencies.path_value,
    )?;
    let chosen_bind_address = first_healthy_instance(
        &launch_dependencies.candidate_bind_addresses,
        launch_dependencies.http_timeout,
    )?;
    let models_document = get_loopback_json(
        chosen_bind_address,
        MODELS_PATH,
        launch_dependencies.http_timeout,
    )
    .map_err(|_| LaunchError::ModelListUnavailable)?;
    let chat_models = chat_models_from_models_document(&models_document)?;
    let selected_chat_model = select_chat_model(
        &chat_models,
        launch_arguments.model_id.as_deref(),
        launch_dependencies.is_interactive,
        launch_dependencies.stdin,
        launch_dependencies.stderr,
    )?;
    warn_if_context_window_is_narrow(&selected_chat_model, launch_dependencies.stderr);
    let config_content = opencode_config_content(chosen_bind_address, &selected_chat_model)
        .map_err(|serialize_error| {
            tracing::debug!(error = %serialize_error, "OpenCode config could not be serialized");
            LaunchError::OpenCodeConfigFailed
        })?;
    Ok(PreparedLaunch {
        program: resolved_launch_tool.program,
        extra_environment: vec![(OPENCODE_CONFIG_CONTENT_VARIABLE.to_owned(), config_content)],
    })
}

/// Returns the first candidate whose `/v1/status` looks like Astronomical.
///
/// A stopped instance refuses loopback immediately, so probing both channels
/// costs nothing on the happy path. Status must include `application` so a
/// random listener on the same port is not treated as the product.
fn first_healthy_instance(
    candidate_bind_addresses: &[SocketAddr],
    http_timeout: Duration,
) -> Result<SocketAddr, LaunchError> {
    for candidate_bind_address in candidate_bind_addresses {
        let Ok(status_document) =
            get_loopback_json(*candidate_bind_address, STATUS_PATH, http_timeout)
        else {
            continue;
        };
        if status_document.get("application").is_none() {
            continue;
        }
        tracing::debug!(
            bind_address = %candidate_bind_address,
            "selected running Astronomical instance"
        );
        return Ok(*candidate_bind_address);
    }
    Err(LaunchError::AstronomicalUnavailable)
}
