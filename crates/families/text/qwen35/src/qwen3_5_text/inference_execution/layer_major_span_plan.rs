//! Which prompt spans one layer-at-a-time prefill runs (issues #1985 and
//! #2221).
//!
//! The spans are cut at the configured size and at the astro SSD prompt cache's
//! boundaries, with the resolver the astro SSD prompt cache's lookup expects. A prompt that
//! fits one span is a plan of one span: it is read the same way as a long
//! one, one layer at a time. The group stops at the prompt's end, or, while the astro SSD prompt cache
//! captures, after the span that ends one token short of a prompt that does
//! not fill its last block: that last token runs on its own so the partial
//! tail block is captured where the next request's lookup expects it.

use crate::qwen3_5_text::model::layer_major_prefill::LayerMajorSpan;
use engine_core::astro_ssd_prompt_cache::{
    astro_ssd_prompt_cache_boundary_completed_prefill_chunk_tokens,
    astro_ssd_prompt_cache_partial_tail_capture_end, resolve_prompt_prefill_chunk_span,
};

/// The spans one layer-major prefill runs and the cursor it advances to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerMajorSpanPlan {
    pub group_end_token: usize,
    pub spans: Vec<LayerMajorSpan>,
}

impl LayerMajorSpanPlan {
    /// The most tokens one span runs, which bounds one forward's working
    /// memory.
    pub fn largest_span_token_count(&self) -> usize {
        let mut largest_span_token_count = 0;
        for span in &self.spans {
            largest_span_token_count =
                largest_span_token_count.max(span.end_token - span.start_token);
        }
        largest_span_token_count
    }

    /// The same plan cut down to its first span, read from `prefill_cursor`.
    ///
    /// Neural Engine programs choose their engines one prompt chunk at a
    /// time and time each chunk to learn whether they help, so while they
    /// are attached a prompt is read a span at a time: each span through
    /// every layer before the next. That costs nothing here, because the
    /// programs attach only while every layer is kept in memory and no layer
    /// is read from storage again for the next span.
    #[must_use]
    pub fn limited_to_first_span(mut self, prefill_cursor: usize) -> Self {
        self.spans.truncate(1);
        if let Some(first_span) = self.spans.first_mut() {
            self.group_end_token = prefill_cursor + first_span.end_token;
            // The span now ends the group, whose end is checkpointed as a
            // group's end is, not as a boundary inside it.
            first_span.ends_at_inner_boundary = false;
        }
        self
    }
}

/// Plans the spans from `prefill_cursor`, or `None` when no token remains.
///
/// `capture_block_token_count` is the astro SSD prompt cache's block size when this
/// request captures, and `None` otherwise.
pub fn layer_major_span_plan(
    prefill_cursor: usize,
    prompt_token_count: usize,
    chunk_token_count: usize,
    capture_block_token_count: Option<usize>,
) -> Option<LayerMajorSpanPlan> {
    // A zero block holds no boundary; the astro SSD prompt cache contract never makes one,
    // and the boundary arithmetic divides by it.
    let capture_block_token_count =
        capture_block_token_count.filter(|block_token_count| *block_token_count > 0);
    let tail_capture_end = capture_block_token_count.map(|block_token_count| {
        astro_ssd_prompt_cache_partial_tail_capture_end(prompt_token_count, block_token_count)
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
    if span_ends.is_empty() {
        return None;
    }
    Some(spans_with_boundaries(
        prefill_cursor,
        &span_ends,
        capture_block_token_count,
    ))
}

/// Turns absolute span ends into spans relative to the cursor, with the
/// astro SSD prompt cache boundaries each one captures.
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
                astro_ssd_prompt_cache_boundary_completed_prefill_chunk_tokens(
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
