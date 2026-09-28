//! Arguments for the ephemeral one-shot `astronomical respond` verb.

/// Parsed `astronomical respond` invocation before any IPC work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RespondArguments {
    /// The user's prompt, exactly as supplied on the command line.
    pub prompt: String,
    /// Exact model identity to demand from the resident daemon, when given.
    pub model_id: Option<String>,
    /// Buffer the answer and print it once instead of streaming fragments.
    pub no_stream: bool,
}
