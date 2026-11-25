//! Keeping the hidden rows a hosted drafter reads, as a forward makes them.
//!
//! A DFlash drafter conditions on the target's hidden rows after a few
//! chosen layers (its taps). The streaming model's layer loop is the one
//! place those rows exist, so a caller that hosts a drafter asks for the
//! taps before a forward and takes them after it. A forward nobody asked
//! taps of keeps nothing and costs one borrow a layer.

use std::cell::RefCell;

use mlx_bridge::MlxArray;

/// The taps asked of the next forward, and the rows kept so far.
#[derive(Debug, Default)]
pub struct DraftTapCapture {
    /// `None` while no caller wants taps.
    requested: RefCell<Option<RequestedTaps>>,
}

#[derive(Debug)]
struct RequestedTaps {
    /// The layers whose output rows are kept, in the drafter's order.
    tap_layer_ids: Vec<i32>,
    /// One slot a requested layer, filled as its layer completes.
    captured_rows: Vec<Option<MlxArray>>,
}

impl DraftTapCapture {
    /// Asks the next forward to keep the rows after `tap_layer_ids`.
    pub fn begin(&self, tap_layer_ids: &[i32]) {
        let mut captured_rows = Vec::new();
        captured_rows.resize_with(tap_layer_ids.len(), Option::default);
        *self.requested.borrow_mut() = Some(RequestedTaps {
            tap_layer_ids: tap_layer_ids.to_vec(),
            captured_rows,
        });
    }

    /// Keeps `layer_output` (`[1, rows, hidden]`) when `layer_index` is a
    /// requested tap. A forward that is run twice (refused, then exact)
    /// overwrites the first run's rows with the second's.
    ///
    /// Never fails the forward: a row that cannot be kept leaves its slot
    /// empty, and `take` then hands on nothing, so the reply is written
    /// without drafting rather than not at all.
    pub fn keep_layer_output(&self, layer_index: usize, layer_output: &MlxArray) {
        let mut requested = self.requested.borrow_mut();
        let Some(requested_taps) = requested.as_mut() else {
            return;
        };
        for (tap_order, tap_layer_id) in requested_taps.tap_layer_ids.iter().enumerate() {
            if usize::try_from(*tap_layer_id) == Ok(layer_index) {
                requested_taps.captured_rows[tap_order] = layer_output.retain().ok();
            }
        }
    }

    /// Ends the request and returns one array a requested layer, in the
    /// drafter's order; `None` when no taps were asked for or a requested
    /// layer never ran, in which case the caller has nothing to hand on.
    pub fn take(&self) -> Option<Vec<MlxArray>> {
        let requested_taps = self.requested.borrow_mut().take()?;
        let mut taps = Vec::with_capacity(requested_taps.captured_rows.len());
        for captured_row in requested_taps.captured_rows {
            taps.push(captured_row?);
        }
        Some(taps)
    }
}

#[cfg(test)]
#[path = "draft_tap_capture_tests.rs"]
mod draft_tap_capture_tests;
