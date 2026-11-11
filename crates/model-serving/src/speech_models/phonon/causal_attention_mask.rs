//! The additive mask of causal attention with an optional window.

/// Row-major `[new, stored + new]`, 0 where a query may look and minus
/// infinity where it may not. Positions are relative to the oldest stored
/// key: query `i` sits at `stored + i`, and sees key `j` when `j` is not in
/// its future and, with a window, when it is fewer than `window` positions
/// back (the query itself counts as one).
pub(crate) fn causal_mask_values(
    stored_count: usize,
    new_count: usize,
    attention_window: Option<usize>,
) -> Vec<f32> {
    let key_count = stored_count + new_count;
    let mut mask_values = Vec::with_capacity(new_count * key_count);
    for query_index in 0..new_count {
        let query_position = stored_count + query_index;
        for key_position in 0..key_count {
            let is_in_past = key_position <= query_position;
            let is_in_window = attention_window
                .is_none_or(|window| query_position.saturating_sub(key_position) < window);
            mask_values.push(if is_in_past && is_in_window {
                0.0
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    mask_values
}

/// Whether one new query against `stored` keys needs a mask at all: with no
/// window, or a window the keys fit in, it may look at every key.
pub(crate) fn single_query_sees_every_key(
    stored_count: usize,
    attention_window: Option<usize>,
) -> bool {
    attention_window.is_none_or(|window| stored_count < window)
}

#[cfg(test)]
#[path = "causal_attention_mask_tests.rs"]
mod tests;
