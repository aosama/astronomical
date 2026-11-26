//! How a model's reply tokens got their missing experts, counted where the
//! choice is made.
//!
//! A reply token's missing experts either land straight in their table or
//! take the copying path, and the copying path is several times slower.
//! From outside both look the same, a token that took a little longer, so
//! this tally names which way each fetching layer went and why the direct
//! way was refused when it was. A measurement can then say how much of a
//! token's time is still the copying path's, and what would have to change
//! to take it.

use std::cell::Cell;
use std::time::Duration;

/// Why a fetching layer's missing experts did or did not go straight into
/// its table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectExpertFetchOutcome {
    /// Read from storage straight into the table.
    FetchedIntoTable,
    /// The reply's plan keeps nothing on this layer, or tables have no
    /// slots at all, so there is no table to read into.
    LayerKeepsNothing,
    /// The layer may keep experts, but its table could not take them: it
    /// does not exist yet, cannot grow or evict enough, or holds experts in
    /// an encoding the reader does not write.
    TableCouldNotTakeThem,
    /// The model has no direct reader, or the direct path is switched off.
    NoDirectReader,
}

/// Counts since they were last taken.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectExpertFetchCounts {
    pub fetched_into_table_layer_count: u64,
    pub fetched_into_table_expert_count: u64,
    /// Bytes of experts those fetches read from storage.
    pub fetched_into_table_bytes: u64,
    pub layer_keeps_nothing_count: u64,
    pub table_could_not_take_them_count: u64,
    pub no_direct_reader_count: u64,
    /// Time inside direct fetches: slot planning, the reads and bookkeeping.
    pub direct_fetch_nanoseconds: u64,
    /// The part of that time spent in the reads themselves.
    pub storage_read_nanoseconds: u64,
}

/// The running counts of one model.
#[derive(Debug, Default)]
pub struct DirectExpertFetchTally {
    counts: Cell<DirectExpertFetchCounts>,
}

impl DirectExpertFetchTally {
    pub fn record_outcome(&self, outcome: DirectExpertFetchOutcome) {
        let mut counts = self.counts.get();
        let outcome_count = match outcome {
            DirectExpertFetchOutcome::FetchedIntoTable => {
                &mut counts.fetched_into_table_layer_count
            }
            DirectExpertFetchOutcome::LayerKeepsNothing => &mut counts.layer_keeps_nothing_count,
            DirectExpertFetchOutcome::TableCouldNotTakeThem => {
                &mut counts.table_could_not_take_them_count
            }
            DirectExpertFetchOutcome::NoDirectReader => &mut counts.no_direct_reader_count,
        };
        *outcome_count = outcome_count.saturating_add(1);
        self.counts.set(counts);
    }

    /// Adds one completed direct fetch of `expert_count` experts.
    pub fn record_fetch(
        &self,
        expert_count: usize,
        fetched_bytes: u64,
        fetch_elapsed: Duration,
        storage_read_elapsed: Duration,
    ) {
        let mut counts = self.counts.get();
        counts.fetched_into_table_expert_count = counts
            .fetched_into_table_expert_count
            .saturating_add(u64::try_from(expert_count).unwrap_or(u64::MAX));
        counts.fetched_into_table_bytes = counts
            .fetched_into_table_bytes
            .saturating_add(fetched_bytes);
        counts.direct_fetch_nanoseconds = counts
            .direct_fetch_nanoseconds
            .saturating_add(u64::try_from(fetch_elapsed.as_nanos()).unwrap_or(u64::MAX));
        counts.storage_read_nanoseconds = counts
            .storage_read_nanoseconds
            .saturating_add(u64::try_from(storage_read_elapsed.as_nanos()).unwrap_or(u64::MAX));
        self.counts.set(counts);
    }

    /// The counts so far, leaving the tally at zero.
    pub fn take(&self) -> DirectExpertFetchCounts {
        self.counts.take()
    }
}

#[cfg(test)]
#[path = "direct_expert_fetch_tally_tests.rs"]
mod tests;
