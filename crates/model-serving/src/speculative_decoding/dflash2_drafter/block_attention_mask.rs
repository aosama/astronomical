//! Which keys each drafted block row may attend to.
//!
//! Keys are the kept context rows followed by the block's own rows. A block
//! row sees every block row (DFlash2 drafts the block at once, so the block is
//! not causal unless the checkpoint says so) and the context rows inside its
//! sliding window, counted by row index exactly as the reference does: block
//! row `i` sits at index `context_rows + i`, and sees context row `j` when
//! `context_rows + i - j < window`.

/// Row-major `[block_rows, context_rows + block_rows]`; true where the query
/// may attend. `sliding_window` is None for a full-attention layer.
pub fn block_attention_visibility(
    context_row_count: i32,
    block_row_count: i32,
    sliding_window: Option<i32>,
    block_is_causal: bool,
) -> Vec<bool> {
    let key_count = context_row_count + block_row_count;
    let mut visibility = Vec::with_capacity((block_row_count * key_count) as usize);
    for block_row_index in 0..block_row_count {
        let query_index = context_row_count + block_row_index;
        for key_index in 0..key_count {
            let is_visible = if key_index < context_row_count {
                match sliding_window {
                    Some(window) => query_index - key_index < window,
                    None => true,
                }
            } else {
                !block_is_causal || key_index <= query_index
            };
            visibility.push(is_visible);
        }
    }
    visibility
}
