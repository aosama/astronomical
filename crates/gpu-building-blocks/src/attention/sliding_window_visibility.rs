//! Architecture-neutral sliding-window causal visibility.
//!
//! A query at absolute position `q` may attend to a key at absolute position
//! `k` only when `q >= k` and `q < k + window`. Positions are absolute token
//! indices, never local query ranks.

/// Errors from sliding-window visibility construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SlidingWindowVisibilityError {
    /// Window size was zero.
    ZeroWindowSize,
    /// Query or key token count was zero.
    ZeroTokenCount { description: &'static str },
}

/// Returns whether one absolute query position may attend one absolute key position.
pub fn sliding_window_position_is_visible(
    query_absolute_position: u32,
    key_absolute_position: u32,
    window_size: u32,
) -> Result<bool, SlidingWindowVisibilityError> {
    if window_size == 0 {
        return Err(SlidingWindowVisibilityError::ZeroWindowSize);
    }
    Ok(query_absolute_position >= key_absolute_position
        && query_absolute_position < key_absolute_position.saturating_add(window_size))
}

/// Builds a row-major visibility table from absolute query and key ranges.
pub fn sliding_window_visibility_table(
    first_query_absolute_position: u32,
    query_token_count: u32,
    first_key_absolute_position: u32,
    key_token_count: u32,
    window_size: u32,
) -> Result<Vec<Vec<bool>>, SlidingWindowVisibilityError> {
    if window_size == 0 {
        return Err(SlidingWindowVisibilityError::ZeroWindowSize);
    }
    if query_token_count == 0 {
        return Err(SlidingWindowVisibilityError::ZeroTokenCount {
            description: "query token count must be positive",
        });
    }
    if key_token_count == 0 {
        return Err(SlidingWindowVisibilityError::ZeroTokenCount {
            description: "key token count must be positive",
        });
    }
    let mut visibility_rows = Vec::with_capacity(query_token_count as usize);
    for query_offset in 0..query_token_count {
        let query_absolute_position = first_query_absolute_position.saturating_add(query_offset);
        let mut visibility_columns = Vec::with_capacity(key_token_count as usize);
        for key_offset in 0..key_token_count {
            let key_absolute_position = first_key_absolute_position.saturating_add(key_offset);
            visibility_columns.push(sliding_window_position_is_visible(
                query_absolute_position,
                key_absolute_position,
                window_size,
            )?);
        }
        visibility_rows.push(visibility_columns);
    }
    Ok(visibility_rows)
}
