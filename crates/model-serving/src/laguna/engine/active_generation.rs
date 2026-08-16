//! Request-owned Laguna state retained between bounded engine advances.

use astronomical_ipc_protocol::{RequestId, WorkerPromptWorkReuse};
use astronomical_runtime_integration::MlxArray;

use crate::laguna::LagunaDecoderState;
use crate::{PerformanceAttribution, PersistentPromptCacheBlockKey};

/// One active Laguna request whose prompt and decode work advance incrementally.
pub(super) struct LagunaActiveGeneration {
    /// Protocol identity used to reject advances or cancellation for another request.
    pub(super) request_id: RequestId,
    /// Model-owned attention and recurrent state after the latest completed advance.
    pub(super) decoder_state: LagunaDecoderState,
    /// Previously sampled token that the next decode forward must consume.
    pub(super) next_input_token_ids: Vec<u32>,
    /// User-visible token allowance that remains after completed decode advances.
    pub(super) remaining_output_tokens: u16,
    /// Original user-configured output bound retained for attribution reports.
    pub(super) configured_maximum_output_tokens: u16,
    /// Optional per-operation timing and byte attribution owned by this request.
    pub(super) performance_attribution: PerformanceAttribution,
    /// Logical context length used to recompose the decode-time memory budget.
    pub(super) context_token_count: u64,
    /// Complete rendered prompt retained until every uncached chunk is processed.
    pub(super) prompt_token_ids: Vec<u32>,
    /// First prompt token not yet represented by the live decoder state.
    pub(super) next_prompt_token_position: usize,
    /// Durable parent key used when the next complete prompt-cache block is published.
    pub(super) last_published_block_key: Option<PersistentPromptCacheBlockKey>,
    /// Evaluated logits from the terminal prompt chunk, consumed for first-token sampling.
    pub(super) terminal_prompt_logits: Option<MlxArray>,
    /// Eligible and restored model work reported consistently on every progress boundary.
    pub(super) prompt_work_reuse: WorkerPromptWorkReuse,
}
