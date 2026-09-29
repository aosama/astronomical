//! Parsed `astronomical models` invocation.

/// One `astronomical models` subcommand.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelsCommand {
    /// Models installed on this Mac.
    List,
    /// The release catalog: what can be downloaded and its local state.
    Supported,
    /// Show the effective default model; with a MODEL_ID, persist it.
    Default { model_id: Option<String> },
    /// Start (or resume) a download and wait, with live progress.
    Download { model_id: String },
}
