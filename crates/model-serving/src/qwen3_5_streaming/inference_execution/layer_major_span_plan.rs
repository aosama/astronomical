//! Which prompt spans one layer-major prefill runs (issue #1985).
//!
//! The spans are exactly the chunks the chunk-by-chunk engine would run from
//! the cursor, found with the same resolver, so the prompt cache sees the
//! same boundaries. The group stops at the prompt's end, or, while the cache
//! captures, after the span that ends one token short of a prompt that does
//! not fill its last block: that last token runs on its own so the partial
//! tail block is captured where the next request's lookup expects it.

use crate::persistent_cache::{
    persistent_prompt_cache_boundary_completed_prefill_chunk_tokens,
    persistent_prompt_cache_partial_tail_capture_end, resolve_prompt_prefill_chunk_span,
};
use crate::qwen3_5_streaming::model::layer_major_prefill::LayerMajorSpan;

/// The spans one layer-major prefill runs and the cursor it advances to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LayerMajorSpanPlan {
    pub(crate) group_end_token: usize,
    pub(crate) spans: Vec<LayerMajorSpan>,
}

/// Plans the spans from `prefill_cursor`, or `None` when fewer than two
/// remain: a single span is exactly what the chunk path already runs.
///
/// `capture_block_token_count` is the prompt cache's block size when this
/// request captures, and `None` otherwise.
pub(crate) fn layer_major_span_plan(
    prefill_cursor: usize,
    prompt_token_count: usize,
    chunk_token_count: usize,
    capture_block_token_count: Option<usize>,
) -> Option<LayerMajorSpanPlan> {
    let tail_capture_end = capture_block_token_count.map(|block_token_count| {
        persistent_prompt_cache_partial_tail_capture_end(prompt_token_count, block_token_count)
    });
    let mut span_ends = Vec::new();
    let mut span_start = prefill_cursor;
    while let Some(resolved_span) = resolve_prompt_prefill_chunk_span(
        span_start,
        prompt_token_count,
        span_start
            .saturating_add(chunk_token_count.max(1))
            .min(prompt_token_count),
        capture_block_token_count.is_some(),
        capture_block_token_count.is_some(),
        capture_block_token_count.unwrap_or(0),
    ) {
        span_start = resolved_span.requested_prefill_chunk_end;
        span_ends.push(span_start);
        if span_start == prompt_token_count || Some(span_start) == tail_capture_end {
            break;
        }
    }
    if span_ends.len() < 2 {
        return None;
    }
    Some(spans_with_boundaries(
        prefill_cursor,
        &span_ends,
        capture_block_token_count,
    ))
}

/// Turns absolute span ends into spans relative to the cursor, with the
/// prompt-cache boundaries each one captures.
fn spans_with_boundaries(
    prefill_cursor: usize,
    span_ends: &[usize],
    capture_block_token_count: Option<usize>,
) -> LayerMajorSpanPlan {
    let group_end_token = span_ends.last().copied().unwrap_or(prefill_cursor);
    let mut spans = Vec::with_capacity(span_ends.len());
    let mut span_start = prefill_cursor;
    for span_end in span_ends {
        let span_token_count = span_end - span_start;
        let mut interior_boundary_tokens =
            capture_block_token_count.map_or_else(Vec::new, |block_token_count| {
                persistent_prompt_cache_boundary_completed_prefill_chunk_tokens(
                    span_start,
                    *span_end,
                    block_token_count,
                )
            });
        let ends_on_boundary = interior_boundary_tokens.last() == Some(&span_token_count);
        if ends_on_boundary {
            interior_boundary_tokens.pop();
        }
        spans.push(LayerMajorSpan {
            start_token: span_start - prefill_cursor,
            end_token: span_end - prefill_cursor,
            interior_boundary_tokens,
            ends_at_inner_boundary: ends_on_boundary && *span_end != group_end_token,
        });
        span_start = *span_end;
    }
    LayerMajorSpanPlan {
        group_end_token,
        spans,
    }
}

#[cfg(test)]
#[path = "layer_major_span_plan_tests.rs"]
mod tests;
