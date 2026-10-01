//! Arguments for the ephemeral one-shot `astronomical respond` verb.

use std::path::PathBuf;

/// Parsed `astronomical respond` invocation before any IPC work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RespondArguments {
    /// The user's prompt, exactly as supplied on the command line.
    pub prompt: String,
    /// Raster images to attach to the prompt, in the order the user supplied them.
    pub images: Vec<PathBuf>,
    /// Exact model identity to demand from the resident daemon, when given.
    pub model_id: Option<String>,
    /// System-prompt-style guidance applied to the reply, when given.
    pub instructions: Option<String>,
    /// Cap on the tokens a thinking model may spend reasoning, when given.
    /// `None` lets the model think freely up to the output budget.
    pub thinking_budget: Option<u16>,
    /// Buffer the answer and print it once instead of streaming fragments.
    pub no_stream: bool,
}
