use std::{ffi::OsString, path::PathBuf};

use astronomical_config::{
    AstronomicalConfigError, AstronomicalInstancePaths, AstronomicalRuntimeInstance,
};

const HELP_TEXT: &str = "Astronomical local model runner\n\nUsage: astronomicald [--instance stable|development] [--state-directory PATH | --application-group-container PATH]\n       astronomicald --help\n       astronomicald --version\n\nOptions:\n  --instance INSTANCE                  Runtime instance (default: development)\n  --state-directory PATH               Absolute writable state root for this invocation\n  --application-group-container PATH   Absolute app group container holding the instance's standard state\n  -h, --help                           Show this help\n  --version                            Show exact build identity\n";
const STATE_DIRECTORY_FLAG: &str = "--state-directory";
const APPLICATION_GROUP_CONTAINER_FLAG: &str = "--application-group-container";

pub(crate) enum DaemonCommand {
    Run(DaemonArguments),
    Help,
    Version,
}

pub(crate) struct DaemonArguments {
    runtime_instance: AstronomicalRuntimeInstance,
    state_location: StateLocation,
}

/// Where this invocation keeps the instance's state.
#[derive(Debug, PartialEq, Eq)]
enum StateLocation {
    /// The channel's default place for the current user.
    Default,
    /// A caller-chosen root, served as a custom instance on an ephemeral port.
    StateDirectory(PathBuf),
    /// The standard instance, kept in an app group container that the
    /// sandboxed chat app and menu-bar agent can both open.
    ApplicationGroupContainer(PathBuf),
}

impl DaemonArguments {
    pub(crate) fn parse(
        process_arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<DaemonCommand, DaemonArgumentError> {
        let supplied_arguments = process_arguments.into_iter().skip(1).collect::<Vec<_>>();
        if supplied_arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            return Ok(DaemonCommand::Help);
        }
        if supplied_arguments
            .iter()
            .any(|argument| argument == "--version")
        {
            return Ok(DaemonCommand::Version);
        }

        let mut runtime_instance = AstronomicalRuntimeInstance::Development;
        let mut state_location = StateLocation::Default;
        let mut has_runtime_instance_argument = false;
        let mut argument_index = 0;
        while argument_index < supplied_arguments.len() {
            let argument = &supplied_arguments[argument_index];
            let argument_value = supplied_arguments.get(argument_index + 1);
            if argument == "--instance" {
                if has_runtime_instance_argument {
                    return Err(DaemonArgumentError::RepeatedArgument("--instance"));
                }
                runtime_instance = parse_runtime_instance(argument_value)?;
                has_runtime_instance_argument = true;
            } else if argument == STATE_DIRECTORY_FLAG {
                require_unset_state_location(&state_location, STATE_DIRECTORY_FLAG)?;
                state_location = StateLocation::StateDirectory(parse_absolute_directory(
                    argument_value,
                    STATE_DIRECTORY_FLAG,
                )?);
            } else if argument == APPLICATION_GROUP_CONTAINER_FLAG {
                require_unset_state_location(&state_location, APPLICATION_GROUP_CONTAINER_FLAG)?;
                state_location = StateLocation::ApplicationGroupContainer(
                    parse_absolute_directory(argument_value, APPLICATION_GROUP_CONTAINER_FLAG)?,
                );
            } else {
                return Err(DaemonArgumentError::UnknownArgument(
                    argument.to_string_lossy().into_owned(),
                ));
            }
            argument_index += 2;
        }
        Ok(DaemonCommand::Run(Self {
            runtime_instance,
            state_location,
        }))
    }

    pub(crate) fn resolve_instance_paths(
        self,
    ) -> Result<AstronomicalInstancePaths, AstronomicalConfigError> {
        self.state_location.instance_paths(self.runtime_instance)
    }
}

impl StateLocation {
    fn instance_paths(
        self,
        runtime_instance: AstronomicalRuntimeInstance,
    ) -> Result<AstronomicalInstancePaths, AstronomicalConfigError> {
        match self {
            Self::StateDirectory(state_directory) => Ok(
                AstronomicalInstancePaths::for_state_directory(state_directory, runtime_instance),
            ),
            Self::ApplicationGroupContainer(application_group_container) => {
                Ok(AstronomicalInstancePaths::for_application_group_container(
                    application_group_container,
                    runtime_instance,
                ))
            }
            Self::Default => {
                AstronomicalInstancePaths::default_location_instance_paths(runtime_instance)
            }
        }
    }
}

fn parse_runtime_instance(
    argument_value: Option<&OsString>,
) -> Result<AstronomicalRuntimeInstance, DaemonArgumentError> {
    let raw_runtime_instance = argument_value
        .ok_or(DaemonArgumentError::MissingValue("--instance"))?
        .to_str()
        .ok_or(DaemonArgumentError::NonUtf8Instance)?;
    match raw_runtime_instance.parse::<AstronomicalRuntimeInstance>() {
        Ok(runtime_instance) => Ok(runtime_instance),
        Err(configuration_error) => Err(DaemonArgumentError::InvalidInstance(
            configuration_error.to_string(),
        )),
    }
}

/// One invocation has one state location: a second flag of either kind would
/// leave it unclear which root the instance lock and the models belong to.
fn require_unset_state_location(
    state_location: &StateLocation,
    supplied_flag: &'static str,
) -> Result<(), DaemonArgumentError> {
    match state_location {
        StateLocation::Default => Ok(()),
        StateLocation::StateDirectory(_) if supplied_flag == STATE_DIRECTORY_FLAG => {
            Err(DaemonArgumentError::RepeatedArgument(supplied_flag))
        }
        StateLocation::ApplicationGroupContainer(_)
            if supplied_flag == APPLICATION_GROUP_CONTAINER_FLAG =>
        {
            Err(DaemonArgumentError::RepeatedArgument(supplied_flag))
        }
        _ => Err(DaemonArgumentError::ConflictingStateLocations),
    }
}

fn parse_absolute_directory(
    argument_value: Option<&OsString>,
    directory_flag: &'static str,
) -> Result<PathBuf, DaemonArgumentError> {
    let directory =
        PathBuf::from(argument_value.ok_or(DaemonArgumentError::MissingValue(directory_flag))?);
    if !directory.is_absolute() || directory.parent().is_none() {
        return Err(DaemonArgumentError::InvalidStateDirectory(directory));
    }
    Ok(directory)
}

pub(crate) const fn help_text() -> &'static str {
    HELP_TEXT
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum DaemonArgumentError {
    #[error("missing value for {0}")]
    MissingValue(&'static str),
    #[error("runtime instance must be valid UTF-8")]
    NonUtf8Instance,
    #[error("{0}")]
    InvalidInstance(String),
    #[error("argument may be supplied only once: {0}")]
    RepeatedArgument(&'static str),
    #[error("state directory must be an absolute non-root path, got {0:?}")]
    InvalidStateDirectory(PathBuf),
    #[error(
        "--state-directory and --application-group-container name two state locations; supply one"
    )]
    ConflictingStateLocations,
    #[error("unrecognized argument: {0}")]
    UnknownArgument(String),
}

#[cfg(test)]
#[path = "daemon_arguments_tests.rs"]
mod daemon_arguments_tests;
