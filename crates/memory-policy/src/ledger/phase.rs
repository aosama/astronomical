//! What the model is doing, as far as memory is concerned.

/// The three states whose memory needs differ.
///
/// Reading a prompt works on a whole span of tokens at once, so one layer
/// needs far more MLX working buffers than writing a reply, which handles one
/// token. The span size is part of the phase because MLX working buffers are
/// measured per span size.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LedgerPhase {
    /// No request is running.
    Idle,
    /// A prompt is being read, at most this many tokens per layer step.
    ReadingPrompt { span_token_count: u32 },
    /// The reply is being written, one token per layer step.
    WritingReply,
}
