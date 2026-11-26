//! How many dense decoder layers stay kept after a release, and how many
//! fit in the room the memory governor has.
//!
//! The streamer applies these counts; keeping the arithmetic apart lets it
//! be tested without loading a model (issues #1985 and #2221).

/// How many leading layers fit in `room_bytes`. Layers are kept whole and
/// in order, so the count stops at the first layer that does not fit.
pub fn leading_layer_count_within(layer_payload_bytes: &[u64], room_bytes: u64) -> usize {
    let mut kept_layer_count = 0;
    let mut kept_layer_payload_bytes = 0_u64;
    for one_layer_payload_bytes in layer_payload_bytes {
        kept_layer_payload_bytes =
            kept_layer_payload_bytes.saturating_add(*one_layer_payload_bytes);
        if kept_layer_payload_bytes > room_bytes {
            break;
        }
        kept_layer_count += 1;
    }
    kept_layer_count
}

/// Kept layers left after giving trailing ones back until at least
/// `required_bytes` are freed, and the bytes that frees. Layers go whole,
/// last first, so the leading layers a forward runs first stay kept.
pub fn kept_layer_count_after_release(
    layer_payload_bytes: &[u64],
    current_kept_layer_count: usize,
    required_bytes: u64,
) -> (usize, u64) {
    let mut kept_layer_count = current_kept_layer_count.min(layer_payload_bytes.len());
    let mut released_bytes = 0_u64;
    while released_bytes < required_bytes && kept_layer_count > 0 {
        kept_layer_count -= 1;
        released_bytes = released_bytes.saturating_add(layer_payload_bytes[kept_layer_count]);
    }
    (kept_layer_count, released_bytes)
}

/// Layers in flight while a forward streams: the one computing and the one
/// being read for it, never more than the layers that actually stream.
pub fn streaming_slot_count_in_use(
    decoder_layer_count: usize,
    kept_layer_count: usize,
    streaming_slot_count: u64,
) -> u64 {
    let streamed_layer_count =
        u64::try_from(decoder_layer_count.saturating_sub(kept_layer_count)).unwrap_or(u64::MAX);
    streamed_layer_count.min(streaming_slot_count)
}

#[cfg(test)]
#[path = "dense_layer_counts_tests.rs"]
mod tests;
