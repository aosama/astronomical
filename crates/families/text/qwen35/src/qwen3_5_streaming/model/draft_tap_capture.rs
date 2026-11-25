//! Keeping the hidden rows a hosted drafter reads, as a forward makes them.
//!
//! A DFlash drafter conditions on the target's hidden rows after a few
//! chosen layers (its taps). The streaming model's layer loops are the one
//! place those rows exist, so a caller that hosts a drafter asks for the
//! taps before a forward and takes them after it. A forward nobody asked
//! taps of keeps nothing and costs one borrow a layer.
//!
//! A prompt is read one layer at a time over several spans, so a tap's
//! rows arrive span by span, and every span of one layer before any span
//! of the next. The capture keeps each tap's most recent rows up to a row
//! limit: taps are about 50 MB a thousand tokens, and a drafter gains
//! nothing from rows beyond its window (capping its context at 2,048 rows
//! measured within the spread of no cap at all).

use std::cell::RefCell;

use mlx_bridge::MlxArray;

/// The taps asked of the forward in progress, and the rows kept so far.
#[derive(Debug, Default)]
pub struct DraftTapCapture {
    /// `None` while no caller wants taps.
    requested: RefCell<Option<RequestedTaps>>,
}

#[derive(Debug)]
struct RequestedTaps {
    /// The layers whose output rows are kept, in the drafter's order.
    tap_layer_ids: Vec<i32>,
    /// The most rows kept a tap; older spans give way to newer ones.
    row_limit: usize,
    /// For each requested layer, its kept spans in position order.
    kept_spans_by_tap: Vec<Vec<KeptSpan>>,
}

/// One span's rows after one tapped layer.
#[derive(Debug)]
struct KeptSpan {
    first_position: i32,
    row_count: usize,
    /// `[1, rows, hidden]`.
    rows: MlxArray,
}

/// What a forward kept: every tap's spans cover the same positions.
#[derive(Debug)]
pub struct CapturedTaps {
    /// The position of each tap's first kept row.
    pub first_position: i32,
    /// For each tap, in the drafter's order, its spans in position order.
    pub span_rows_by_tap: Vec<Vec<MlxArray>>,
}

impl DraftTapCapture {
    /// Asks the forward about to run to keep the rows after
    /// `tap_layer_ids`, at most the newest `row_limit` of them a tap.
    pub fn begin(&self, tap_layer_ids: &[i32], row_limit: usize) {
        let mut kept_spans_by_tap = Vec::new();
        kept_spans_by_tap.resize_with(tap_layer_ids.len(), Vec::new);
        *self.requested.borrow_mut() = Some(RequestedTaps {
            tap_layer_ids: tap_layer_ids.to_vec(),
            row_limit: row_limit.max(1),
            kept_spans_by_tap,
        });
    }

    /// Keeps `layer_rows` (`[1, rows, hidden]`, the rows from
    /// `first_position` on) when `layer_index` is a requested tap.
    ///
    /// A span that is run again (a failed span of a prompt, or a forward
    /// served twice) replaces what it and every later span left. Never
    /// fails the forward: rows that cannot be kept leave the tap short, and
    /// `take` then hands on nothing, so the reply is written without
    /// drafting rather than not at all.
    pub fn keep_layer_rows(&self, layer_index: usize, first_position: i32, layer_rows: &MlxArray) {
        let mut requested = self.requested.borrow_mut();
        let Some(requested_taps) = requested.as_mut() else {
            return;
        };
        let row_limit = requested_taps.row_limit;
        let row_count =
            usize::try_from(layer_rows.shape().get(1).copied().unwrap_or(0)).unwrap_or(0);
        for (tap_order, tap_layer_id) in requested_taps.tap_layer_ids.iter().enumerate() {
            if usize::try_from(*tap_layer_id) != Ok(layer_index) {
                continue;
            }
            let kept_spans = &mut requested_taps.kept_spans_by_tap[tap_order];
            kept_spans.retain(|kept_span| kept_span.first_position < first_position);
            let Ok(rows) = layer_rows.retain() else {
                continue;
            };
            kept_spans.push(KeptSpan {
                first_position,
                row_count,
                rows,
            });
            drop_spans_beyond_the_row_limit(kept_spans, row_limit);
        }
    }

    /// Ends the request and returns what was kept; `None` when no taps
    /// were asked for, a requested layer never ran, or the taps do not
    /// cover the same positions, in which case the caller has nothing it
    /// can hand to a drafter.
    pub fn take(&self) -> Option<CapturedTaps> {
        let requested_taps = self.requested.borrow_mut().take()?;
        let first_tap_spans = requested_taps.kept_spans_by_tap.first()?;
        let first_position = first_tap_spans.first()?.first_position;
        let kept_row_count = kept_row_count(first_tap_spans);
        let mut span_rows_by_tap = Vec::with_capacity(requested_taps.kept_spans_by_tap.len());
        for kept_spans in requested_taps.kept_spans_by_tap {
            let covers_the_same_rows = kept_spans.first().map(|span| span.first_position)
                == Some(first_position)
                && kept_row_count == self::kept_row_count(&kept_spans);
            if !covers_the_same_rows {
                return None;
            }
            let mut span_rows = Vec::with_capacity(kept_spans.len());
            for kept_span in kept_spans {
                span_rows.push(kept_span.rows);
            }
            span_rows_by_tap.push(span_rows);
        }
        Some(CapturedTaps {
            first_position,
            span_rows_by_tap,
        })
    }
}

fn kept_row_count(kept_spans: &[KeptSpan]) -> usize {
    let mut row_count = 0;
    for kept_span in kept_spans {
        row_count += kept_span.row_count;
    }
    row_count
}

/// Lets the oldest spans go while the newer ones alone reach the limit.
fn drop_spans_beyond_the_row_limit(kept_spans: &mut Vec<KeptSpan>, row_limit: usize) {
    let mut kept_rows = kept_row_count(kept_spans);
    let mut dropped_span_count = 0;
    for kept_span in kept_spans.iter() {
        if kept_rows - kept_span.row_count < row_limit {
            break;
        }
        kept_rows -= kept_span.row_count;
        dropped_span_count += 1;
    }
    kept_spans.drain(..dropped_span_count);
}

#[cfg(test)]
#[path = "draft_tap_capture_tests.rs"]
mod draft_tap_capture_tests;
