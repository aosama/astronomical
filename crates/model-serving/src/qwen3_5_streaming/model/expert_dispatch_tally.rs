//! Which way each layer's expert feed-forward was served, counted where the
//! choice is made.
//!
//! A forward's speed is decided layer by layer: a layer whose route stays on
//! the graphics processor adds nothing but its own work, while a layer whose
//! route has to reach the host first stops the whole forward until every
//! layer before it has run. The second kind is invisible from outside (the
//! forward still returns the right logits, only later), so slowness used to
//! be found by timing whole replies and guessing. This tally names the way
//! each layer went and adds up the time spent waiting for routes, so a
//! measurement reads the cause instead of inferring it.

use std::cell::Cell;
use std::time::Duration;

/// The way one layer's expert feed-forward was served.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExpertDispatchPath {
    /// A prompt forward multiplied the complete page it had in hand; the
    /// route never left the graphics processor.
    CompletePageInHand,
    /// The layer's table was multiplied with the route remapped on the
    /// graphics processor, with no wait.
    TableWithoutHostRoute,
    /// The route was copied to the host and every routed expert was in the
    /// layer's complete table.
    HostRouteTableHit,
    /// The route was copied to the host and the layer's table is not
    /// complete: the routed experts come from the table, from storage, or
    /// from both.
    HostRouteTableMiss,
    /// None of the paths above took the layer.
    Unhandled,
}

/// How many layers went each way since the counts were last taken.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertDispatchCounts {
    pub complete_page_in_hand_layer_count: u64,
    pub table_without_host_route_layer_count: u64,
    pub host_route_table_hit_layer_count: u64,
    pub host_route_table_miss_layer_count: u64,
    pub unhandled_layer_count: u64,
    /// Time spent copying routes to the host. Each copy waits for every
    /// layer before it, so this is forward time, not copy time alone.
    pub host_route_wait_nanoseconds: u64,
}

impl ExpertDispatchCounts {
    /// Layers that stopped the forward to bring their route to the host.
    #[must_use]
    pub const fn host_route_layer_count(&self) -> u64 {
        self.host_route_table_hit_layer_count
            + self.host_route_table_miss_layer_count
            + self.unhandled_layer_count
    }
}

/// The running counts of one model.
#[derive(Debug, Default)]
pub(crate) struct ExpertDispatchTally {
    counts: Cell<ExpertDispatchCounts>,
}

impl ExpertDispatchTally {
    pub(crate) fn record_path(&self, dispatch_path: ExpertDispatchPath) {
        let mut counts = self.counts.get();
        let path_layer_count = match dispatch_path {
            ExpertDispatchPath::CompletePageInHand => &mut counts.complete_page_in_hand_layer_count,
            ExpertDispatchPath::TableWithoutHostRoute => {
                &mut counts.table_without_host_route_layer_count
            }
            ExpertDispatchPath::HostRouteTableHit => &mut counts.host_route_table_hit_layer_count,
            ExpertDispatchPath::HostRouteTableMiss => &mut counts.host_route_table_miss_layer_count,
            ExpertDispatchPath::Unhandled => &mut counts.unhandled_layer_count,
        };
        *path_layer_count = path_layer_count.saturating_add(1);
        self.counts.set(counts);
    }

    pub(crate) fn record_host_route_wait(&self, host_route_wait: Duration) {
        let mut counts = self.counts.get();
        counts.host_route_wait_nanoseconds = counts
            .host_route_wait_nanoseconds
            .saturating_add(u64::try_from(host_route_wait.as_nanos()).unwrap_or(u64::MAX));
        self.counts.set(counts);
    }

    /// The counts so far, leaving the tally at zero.
    pub(crate) fn take(&self) -> ExpertDispatchCounts {
        self.counts.take()
    }
}

#[cfg(test)]
#[path = "expert_dispatch_tally_tests.rs"]
mod tests;
