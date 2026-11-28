//! What a model occupies and could keep, as the memory ledger sees it: the
//! core weights, the weights that may be kept or streamed, and the room one
//! step's streamed weights need while a prompt is read and while a reply is
//! written.

use memory_policy::ledger::ModelMemoryShape;

use crate::ModelMemoryGeometry;

/// Layers in flight while a sparse model's prompt is read: the one being
/// multiplied and the one read ahead of it.
const SPARSE_READING_STREAM_SLOT_COUNT: u64 = 2;

/// A sparse model's shape: its experts are what may be kept or streamed.
/// Reading a prompt holds whole layers in flight; writing a reply holds only
/// the experts one forward's rows chose.
///
/// A drafted forward routes each of its rows, so its page is up to that many
/// tokens' experts and never more than the layer. Sized for one token, the
/// slot was 27 MB on the 35B-A3B while a four-row draft check streamed a
/// 28-expert page of 50 MB, and a reply at an 8 GiB limit was refused 1 MB
/// short.
pub(super) fn sparse_model_memory_shape(
    model_geometry: ModelMemoryGeometry,
    reply_forward_row_count: u64,
) -> ModelMemoryShape {
    let writing_reply_stream_slot_bytes = model_geometry
        .largest_routed_expert_page_bytes
        .saturating_mul(reply_forward_row_count)
        .min(model_geometry.largest_complete_expert_layer_bytes);
    ModelMemoryShape {
        core_weight_bytes: model_geometry.model_core_payload_bytes,
        streamable_weight_bytes: model_geometry.complete_expert_payload_bytes,
        reading_prompt_stream_slot_bytes: model_geometry
            .largest_complete_expert_layer_bytes
            .saturating_mul(SPARSE_READING_STREAM_SLOT_COUNT),
        writing_reply_stream_slot_bytes,
    }
}

/// A dense model's shape: the decoder layers its streamer owns are what may
/// be kept or streamed, through the same layer slots in both phases.
pub(super) fn dense_model_memory_shape(
    core_weight_bytes: u64,
    decoder_layer_payload_bytes: &[u64],
    streaming_slot_bytes: u64,
) -> ModelMemoryShape {
    let mut streamable_weight_bytes = 0_u64;
    for layer_payload_bytes in decoder_layer_payload_bytes {
        streamable_weight_bytes = streamable_weight_bytes.saturating_add(*layer_payload_bytes);
    }
    ModelMemoryShape {
        core_weight_bytes,
        streamable_weight_bytes,
        reading_prompt_stream_slot_bytes: streaming_slot_bytes,
        writing_reply_stream_slot_bytes: streaming_slot_bytes,
    }
}

#[cfg(test)]
#[path = "model_memory_shape_tests.rs"]
mod tests;
