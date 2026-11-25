//! How each text engine governs memory around one forward.
//!
//! The engines govern in two different patterns. The resident engine asks
//! its memory governor before a forward runs and may refuse it, moving the
//! request to the streaming engine; the streaming engine starts a
//! measurement before the forward and tells its governor afterwards what the
//! forward used. Shared steps only say where a forward begins and ends; the
//! engine decides what that means, and carries what it needs from one to the
//! other in [`TextForwardMemoryGovernance::ForwardInProgress`].

use runtime_integration::MlxMemorySnapshot;

use crate::decoder_state::RequestDecoderStateStack;
use crate::{InferenceEngineError, PerformanceAttribution};

use super::execution_core::TextExecutionCoreOwner;

/// What a governed forward runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextForwardKind {
    /// A stretch of prompt or injected input, read in one forward.
    PromptChunk { token_count: usize },
    /// One reply token.
    ReplyToken,
}

pub trait TextForwardMemoryGovernance: TextExecutionCoreOwner {
    /// What the engine carries from the start of a forward to its end.
    type ForwardInProgress;

    /// A forward is about to run with the request's decoder state as it is.
    fn begin_governed_forward(
        &mut self,
        forward_kind: TextForwardKind,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<Self::ForwardInProgress, InferenceEngineError>;

    /// The forward ran and its stored state is now `stored_state_bytes`;
    /// returns the memory reading taken after it.
    fn complete_governed_forward(
        &self,
        forward_in_progress: Self::ForwardInProgress,
        stored_state_bytes: u64,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxMemorySnapshot, InferenceEngineError>;
}
