//! One-line user-facing failures for launch. Recovery actions stay in the
//! message so the CLI never dumps config recipes as the happy path.

use std::path::PathBuf;

use crate::daemon_probe::DaemonProbeError;
use astronomical_ipc_protocol::EmbeddingsFailureReason;
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
    #[error("schema object needs --name NAME")]
    SchemaNameRequired,
    #[error(
        "schema object needs at least one property. Try: astronomical schema object --name Thing --string label"
    )]
    SchemaPropertyRequired,
    #[error("{0} must follow a property")]
    SchemaModifierWithoutProperty(String),
    #[error("property may be supplied only once: {0}")]
    SchemaDuplicateProperty(String),
    #[error("invalid property path: {0}")]
    SchemaInvalidPropertyPath(String),
    #[error("unknown schema target: {0}. Supported targets: object")]
    UnknownSchemaTarget(String),
    #[error("unknown validate target: {0}. Supported targets: config")]
    UnknownValidateTarget(String),
    #[error("unknown instance {0}: expected stable or development")]
    UnknownInstance(String),
    #[error("respond needs a prompt. Try: astronomical respond 'Hello'")]
    RespondPromptRequired,
    #[error("embed takes one input. Pass TEXT, --file PATH, or pipe stdin — not several.")]
    EmbedInputConflict,
}

/// Failures of the one-shot `respond` journey after arguments have parsed.
/// These exit 1.
#[derive(Error)]
pub enum RespondError {
    #[error("Astronomical isn't running — start it, then retry.")]
    DaemonNotRunning,
    #[error("Astronomical is running but no model is loaded — load one in the app, then retry.")]
    NoModelLoaded,
    #[error("Model {requested_model_id} isn't loaded; {ready_model_id} is.")]
    RequestedModelNotReady {
        requested_model_id: String,
        ready_model_id: String,
    },
    #[error("The daemon declined the request: {reason}")]
    GenerationRejected { reason: String },
    #[error("The model failed to finish the response.")]
    GenerationFailed,
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error("Could not write the answer to standard output: {cause}")]
    StdoutUnwritable { cause: String },
}

impl From<DaemonProbeError> for RespondError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning => RespondError::DaemonNotRunning,
            DaemonProbeError::DaemonStoppedResponding => RespondError::DaemonStoppedResponding,
        }
    }
}

// The debug form must carry the same user-facing guidance as the display
// form: journeys and tests report whichever rendering they captured.
impl std::fmt::Debug for RespondError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_string())
    }
}

/// Failures of the one-shot `embed` journey after arguments have parsed.
/// These exit 1.
#[derive(Error)]
pub enum EmbedError {
    #[error("Astronomical isn't running — start it, then retry.")]
    DaemonNotRunning,
    #[error(
        "Astronomical is running but no model is loaded — load one in the app, or pass --model, then retry."
    )]
    NoModelLoaded,
    #[error("The daemon declined the request: {reason}")]
    EmbeddingsRejected { reason: String },
    #[error("{}", crate::embed::embeddings_failure_reason_text(reason))]
    EmbeddingsFailed { reason: EmbeddingsFailureReason },
    #[error("embed needs input text. Try: astronomical embed 'Hello'")]
    EmbedInputRequired,
    #[error("Could not read {file_path}: {cause}")]
    InputFileUnreadable { file_path: PathBuf, cause: String },
    #[error("Could not read standard input: {cause}")]
    StdinUnreadable { cause: String },
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error("Could not write the vector document to standard output: {cause}")]
    StdoutUnwritable { cause: String },
}

impl From<DaemonProbeError> for EmbedError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning => EmbedError::DaemonNotRunning,
            DaemonProbeError::DaemonStoppedResponding => EmbedError::DaemonStoppedResponding,
        }
    }
}

// The debug form must carry the same user-facing guidance as the display
// form: journeys and tests report whichever rendering they captured.
impl std::fmt::Debug for EmbedError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_string())
    }
}
