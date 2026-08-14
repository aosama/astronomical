use std::{ffi::OsString, path::PathBuf};

use astronomical_config::{
    AstronomicalConfigError, AstronomicalInstancePaths, AstronomicalRuntimeInstance,
};

const HELP_TEXT: &str = "Astronomical local model server\n\nUsage: astronomicald [--instance stable|development] [--state-directory PATH]\n       astronomicald --help\n       astronomicald --version\n\nOptions:\n  --instance INSTANCE      Runtime instance (default: development)\n  --state-directory PATH   Absolute writable state root for this invocation\n  -h, --help               Show this help\n  --version                Show exact build identity\n";

pub(crate) enum DaemonCommand {
    Run(DaemonArguments),
    Help,
    Version,
}

pub(crate) struct DaemonArguments {
    runtime_instance: AstronomicalRuntimeInstance,
    state_directory_override: Option<PathBuf>,
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
        let mut state_directory_override = None;
        let mut has_runtime_instance_argument = false;
        let mut argument_index = 0;
        while argument_index < supplied_arguments.len() {
            let argument = &supplied_arguments[argument_index];
            if argument == "--instance" {
                if has_runtime_instance_argument {
                    return Err(DaemonArgumentError::RepeatedArgument("--instance"));
                }
                let raw_runtime_instance = supplied_arguments
                    .get(argument_index + 1)
                    .ok_or(DaemonArgumentError::MissingValue("--instance"))?
                    .to_str()
                    .ok_or(DaemonArgumentError::NonUtf8Instance)?;
                runtime_instance = raw_runtime_instance.parse().map_err(
                    |configuration_error: AstronomicalConfigError| {
                        DaemonArgumentError::InvalidInstance(configuration_error.to_string())
                    },
                )?;
                has_runtime_instance_argument = true;
                argument_index += 2;
                continue;
            }
            if argument == "--state-directory" {
                if state_directory_override.is_some() {
                    return Err(DaemonArgumentError::RepeatedArgument("--state-directory"));
                }
                let state_directory = supplied_arguments
                    .get(argument_index + 1)
                    .ok_or(DaemonArgumentError::MissingValue("--state-directory"))?;
                let state_directory = PathBuf::from(state_directory);
                if !state_directory.is_absolute() || state_directory.parent().is_none() {
                    return Err(DaemonArgumentError::InvalidStateDirectory(state_directory));
                }
                state_directory_override = Some(state_directory);
                argument_index += 2;
                continue;
            }
            return Err(DaemonArgumentError::UnknownArgument(
                argument.to_string_lossy().into_owned(),
            ));
        }
        Ok(DaemonCommand::Run(Self {
            runtime_instance,
            state_directory_override,
        }))
    }

    pub(crate) fn resolve_instance_paths(
        self,
    ) -> Result<AstronomicalInstancePaths, AstronomicalConfigError> {
        match self.state_directory_override {
            Some(state_directory) => Ok(AstronomicalInstancePaths::for_state_directory(
                state_directory,
                self.runtime_instance,
            )),
            None => AstronomicalInstancePaths::for_current_user(self.runtime_instance),
        }
    }
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
    #[error("unrecognized argument: {0}")]
    UnknownArgument(String),
}
