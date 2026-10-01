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
#[derive(Debug, Error, PartialEq, Eq)]
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
    #[error("respond takes one prompt. Use either the positional PROMPT or --text, not both.")]
    RespondPromptConflict,
    #[error("--thinking-budget must be a whole number of tokens from 0 to 65535: {0}")]
    InvalidThinkingBudget(String),
    #[error("embed takes one input. Pass TEXT, --file PATH, or pipe stdin — not several.")]
    EmbedInputConflict,
    #[error("models needs a subcommand: list, supported, default, download")]
    ModelsSubcommandRequired,
    #[error("unknown models subcommand: {0}. Try: list, supported, default, download")]
    UnknownModelsSubcommand(String),
    #[error("models download needs a MODEL_ID. Try: astronomical models download Qwen3.5-2B-4bit")]
    ModelsDownloadModelRequired,
}

/// Failures of the one-shot `respond` journey after arguments have parsed.
/// These exit 1, except `ModelUnavailable`, which exits 2 as a usage error.
#[derive(Error)]
pub enum RespondError {
    #[error("Astronomical isn't running — start it, then retry.")]
    DaemonNotRunning,
    #[error("The Astronomical worker is not ready yet — retry in a moment.")]
    WorkerNotReady,
    #[error("{reason}")]
    ModelUnavailable { reason: String },
    #[error("the model download failed: {reason}")]
    DownloadFailed { reason: String },
    #[error("The daemon declined the request: {reason}")]
    GenerationRejected { reason: String },
    #[error("The model failed to finish the response: {reason}")]
    GenerationFailed { reason: String },
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error("Could not write the answer to standard output: {cause}")]
    StdoutUnwritable { cause: String },
    #[error("could not read image {path}: {cause}")]
    ImageReadFailed { path: PathBuf, cause: String },
    #[error("{path} is not a supported image; supported formats: {supported}")]
    UnsupportedImage { path: PathBuf, supported: String },
    #[error(
        "the combined image size is {actual_bytes} bytes, over the {maximum_bytes}-byte limit; \
         send smaller or fewer --image files"
    )]
    ImageTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("could not read schema {path}: {cause}")]
    SchemaReadFailed { path: PathBuf, cause: String },
    #[error("{path} is not valid UTF-8 text: {cause}")]
    SchemaNotUtf8 { path: PathBuf, cause: String },
    #[error(
        "the schema file is {actual_bytes} bytes, over the {maximum_bytes}-byte limit; \
         send a smaller --schema file"
    )]
    SchemaTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
}

impl From<DaemonProbeError> for RespondError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning => RespondError::DaemonNotRunning,
            DaemonProbeError::DaemonStoppedResponding => RespondError::DaemonStoppedResponding,
            DaemonProbeError::DaemonRejected { reason } => RespondError::DownloadFailed { reason },
        }
    }
}

impl From<crate::model_lifecycle::ModelLifecycleError> for RespondError {
    fn from(lifecycle_error: crate::model_lifecycle::ModelLifecycleError) -> Self {
        match lifecycle_error {
            crate::model_lifecycle::ModelLifecycleError::DaemonNotRunning => {
                RespondError::DaemonNotRunning
            }
            crate::model_lifecycle::ModelLifecycleError::WorkerNotReady => {
                RespondError::WorkerNotReady
            }
            crate::model_lifecycle::ModelLifecycleError::DaemonStoppedResponding => {
                RespondError::DaemonStoppedResponding
            }
            crate::model_lifecycle::ModelLifecycleError::ModelUnavailable { reason } => {
                RespondError::ModelUnavailable { reason }
            }
            crate::model_lifecycle::ModelLifecycleError::DownloadFailed { reason } => {
                RespondError::DownloadFailed { reason }
            }
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
/// These exit 1, except `ModelUnavailable`, which exits 2 as a usage error.
#[derive(Error)]
pub enum EmbedError {
    #[error("Astronomical isn't running — start it, then retry.")]
    DaemonNotRunning,
    #[error("The Astronomical worker is not ready yet — retry in a moment.")]
    WorkerNotReady,
    #[error("{reason}")]
    ModelUnavailable { reason: String },
    #[error("the model download failed: {reason}")]
    DownloadFailed { reason: String },
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
            DaemonProbeError::DaemonRejected { reason } => EmbedError::DownloadFailed { reason },
        }
    }
}

impl From<crate::model_lifecycle::ModelLifecycleError> for EmbedError {
    fn from(lifecycle_error: crate::model_lifecycle::ModelLifecycleError) -> Self {
        match lifecycle_error {
            crate::model_lifecycle::ModelLifecycleError::DaemonNotRunning => {
                EmbedError::DaemonNotRunning
            }
            crate::model_lifecycle::ModelLifecycleError::WorkerNotReady => {
                EmbedError::WorkerNotReady
            }
            crate::model_lifecycle::ModelLifecycleError::DaemonStoppedResponding => {
                EmbedError::DaemonStoppedResponding
            }
            crate::model_lifecycle::ModelLifecycleError::ModelUnavailable { reason } => {
                EmbedError::ModelUnavailable { reason }
            }
            crate::model_lifecycle::ModelLifecycleError::DownloadFailed { reason } => {
                EmbedError::DownloadFailed { reason }
            }
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

/// Failures of the `models` verb after arguments have parsed. These exit 1,
/// except `ModelUnavailable`, which exits 2 as a usage error.
#[derive(Error)]
pub enum ModelsError {
    #[error("Astronomical isn't running — start it, then retry.")]
    DaemonNotRunning,
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error("The Astronomical worker is not ready yet — retry in a moment.")]
    WorkerNotReady,
    #[error("{reason}")]
    ModelUnavailable { reason: String },
    #[error("the model download failed: {reason}")]
    DownloadFailed { reason: String },
    #[error("The daemon declined the request: {reason}")]
    DaemonRejected { reason: String },
    #[error("Could not write the model report to standard output: {cause}")]
    StdoutUnwritable { cause: String },
}

impl From<DaemonProbeError> for ModelsError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning => ModelsError::DaemonNotRunning,
            DaemonProbeError::DaemonStoppedResponding => ModelsError::DaemonStoppedResponding,
            DaemonProbeError::DaemonRejected { reason } => ModelsError::DaemonRejected { reason },
        }
    }
}

impl From<crate::model_lifecycle::ModelLifecycleError> for ModelsError {
    fn from(lifecycle_error: crate::model_lifecycle::ModelLifecycleError) -> Self {
        match lifecycle_error {
            crate::model_lifecycle::ModelLifecycleError::DaemonNotRunning => {
                ModelsError::DaemonNotRunning
            }
            crate::model_lifecycle::ModelLifecycleError::WorkerNotReady => {
                ModelsError::WorkerNotReady
            }
            crate::model_lifecycle::ModelLifecycleError::DaemonStoppedResponding => {
                ModelsError::DaemonStoppedResponding
            }
            crate::model_lifecycle::ModelLifecycleError::ModelUnavailable { reason } => {
                ModelsError::ModelUnavailable { reason }
            }
            crate::model_lifecycle::ModelLifecycleError::DownloadFailed { reason } => {
                ModelsError::DownloadFailed { reason }
            }
        }
    }
}

impl std::fmt::Debug for ModelsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_string())
    }
}

/// Failures of the `status` verb after arguments have parsed. These exit 1.
#[derive(Error)]
pub enum StatusError {
    #[error("Astronomical isn't running — start it, then retry.")]
    DaemonNotRunning,
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error("Could not write the status report to standard output: {cause}")]
    StdoutUnwritable { cause: String },
}

impl From<DaemonProbeError> for StatusError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning => StatusError::DaemonNotRunning,
            DaemonProbeError::DaemonStoppedResponding | DaemonProbeError::DaemonRejected { .. } => {
                StatusError::DaemonStoppedResponding
            }
        }
    }
}

impl std::fmt::Debug for StatusError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_string())
    }
}
