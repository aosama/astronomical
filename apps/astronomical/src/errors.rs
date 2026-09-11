//! One-line user-facing failures for launch. Recovery actions stay in the
//! message so the CLI never dumps config recipes as the happy path.

use std::path::PathBuf;

use thiserror::Error;

/// Failures after arguments have already parsed.
#[derive(Debug, Error)]
pub enum LaunchError {
    #[error("Start Astronomical first.")]
    AstronomicalUnavailable,
    #[error("Astronomical is running but did not return a model list.")]
    ModelListUnavailable,
    #[error("No models in the Library yet. Open Astronomical and download one.")]
    NoChatModels,
    #[error("Install OpenCode: curl -fsSL https://opencode.ai/install | bash")]
    OpenCodeMissing,
    #[error("Unknown tool {requested_tool}. Try: astronomical launch opencode")]
    UnknownTool { requested_tool: String },
    #[error("Choose a model with --model; several chat models are in the Library.")]
    ModelPickerRequired,
    #[error("Model {requested_model_id} is not a chat model in the Library.")]
    RequestedModelMissing { requested_model_id: String },
    #[error("Select a model by number or id.")]
    InvalidModelSelection,
    #[error("could not prepare OpenCode config.")]
    OpenCodeConfigFailed,
    #[error("failed to start {program:?}: {cause}")]
    ToolStartFailed { program: PathBuf, cause: String },
}

/// Command-line usage failures. These exit 2.
#[derive(Debug, Error)]
pub enum UsageError {
    #[error("missing command. Try: astronomical launch opencode")]
    MissingCommand,
    #[error("unknown command: {0}")]
    UnknownCommand(String),
    #[error("unrecognized argument: {0}")]
    UnknownArgument(String),
    #[error("missing value for {0}")]
    MissingValue(&'static str),
    #[error("argument may be supplied only once: {0}")]
    RepeatedArgument(&'static str),
    #[error("launch accepts at most one tool name")]
    MultipleTools,
}
