//! The ownership view of MLX active memory the Prism engine reports.
//!
//! The dense folded-ternary model owns its whole weight payload, so active
//! memory is attributed to the model core first and the live decoder state
//! second; there is no expert owner.

use crate::RequestDecoderStateStack;
use crate::memory::telemetry::MlxActiveMemoryBreakdown;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;

/// The breakdown of `active_memory_bytes`, charging the live request's
/// decoder state when a request is running and nothing to context otherwise.
pub(super) fn prism_active_memory_breakdown(
    model: &PrismHadamardTextModel,
    active_memory_bytes: u64,
    live_request_decoder_state: Option<&RequestDecoderStateStack>,
) -> MlxActiveMemoryBreakdown {
    let context_state_payload_bytes =
        live_request_decoder_state.map_or(0, RequestDecoderStateStack::payload_byte_count);
    MlxActiveMemoryBreakdown::reconcile(
        active_memory_bytes,
        0,
        model.pack().total_payload_bytes(),
        context_state_payload_bytes,
    )
}

#[cfg(test)]
#[path = "active_memory_breakdown_tests.rs"]
mod active_memory_breakdown_tests;
