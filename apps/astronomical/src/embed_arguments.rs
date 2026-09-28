//! Arguments for the ephemeral one-shot `astronomical embed` verb.

use std::path::PathBuf;

/// Parsed `astronomical embed` invocation before any IPC work. Exactly one
/// input source survives parsing: text, file, or stdin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbedArguments {
    /// The text to embed, exactly as supplied on the command line.
    pub text: Option<String>,
    /// File whose contents are embedded, when `--file` was given.
    pub file_path: Option<PathBuf>,
    /// Exact model identity to demand from the resident daemon, when given.
    pub model_id: Option<String>,
}
