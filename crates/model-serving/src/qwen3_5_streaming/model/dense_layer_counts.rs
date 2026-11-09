//! How many dense decoder layers stay kept after a release or a re-plan.
//!
//! The streamer and the live limit path apply these counts; keeping the
//! arithmetic apart lets it be tested without loading a model (issue #1985).

/// Which way a re-plan may move the kept layer count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KeptLayerReplan {
    /// After a raise, a load, or a finished request: keep what the plan
    /// allows, reading back layers that were given up.
    MatchPlan,
    /// Before a lower limit is installed: only give layers up. Reading a
    /// layer back here would spend an SSD read on bytes the lower limit is
    /// about to need, and the plan for the lower limit never needs more.
    NeverGrow,
}

/// The kept count a re-plan settles on.
pub(crate) const fn kept_layer_count_after_replan(
    planned_kept_layer_count: usize,
    current_kept_layer_count: usize,
    kept_layer_replan: KeptLayerReplan,
) -> usize {
    match kept_layer_replan {
        KeptLayerReplan::MatchPlan => planned_kept_layer_count,
        KeptLayerReplan::NeverGrow if planned_kept_layer_count < current_kept_layer_count => {
            planned_kept_layer_count
        }
        KeptLayerReplan::NeverGrow => current_kept_layer_count,
    }
}

/// Kept layers left after giving trailing ones back until at least
/// `required_bytes` are freed, and the bytes that frees. Layers go whole,
/// last first, so the leading layers a forward runs first stay kept.
pub(crate) fn kept_layer_count_after_release(
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
pub(crate) fn streaming_slot_count_in_use(
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
