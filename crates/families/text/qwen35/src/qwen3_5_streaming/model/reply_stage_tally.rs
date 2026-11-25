//! Where one reply's time went, by stage, kept while it is written.
//!
//! A reply's speed is one number from outside. Inside, a token is the sum
//! of kernel time, storage reads, hand-offs between the GPU and the host,
//! and host bookkeeping, and a change can only help the part it touches.
//! This tally keeps the parts the engine can see for itself, always on,
//! so a measurement reads them instead of adding timers for the occasion:
//!
//! - how many forwards the reply ran, and over how many rows;
//! - the in-graph lookups' time, split into the reads from storage and the
//!   bookkeeping around them (slot planning, demand counts, the slot map);
//! - a drafted round's stages: drafting, verifying on the target, choosing
//!   the target's tokens, and keeping rows and re-running the head;
//! - the hand-offs MLX made (`graph_host_step::evaluation_counts`).
//!
//! Timing a stage costs two clock reads, about fifty nanoseconds, against
//! stages of tens of microseconds and up.

use std::cell::Cell;
use std::time::Duration;

use graph_host_step::{EvaluationCounts, evaluation_counts};

/// One reply's stages. Durations are totals over the reply.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplyStageCounts {
    /// Forwards of reply rows: single tokens and verified drafted blocks,
    /// a forward served twice (refused, then exact) counted twice.
    pub reply_forward_count: u64,
    /// Rows those forwards ran: one a token, several a drafted block.
    pub reply_row_count: u64,
    /// Layers a lookup inside the graph served.
    pub lookup_count: u64,
    /// Time inside those lookups, reads included.
    pub lookup_nanoseconds: u64,
    /// The part of it spent reading experts from storage; the rest is the
    /// lookup's host bookkeeping.
    pub lookup_storage_read_nanoseconds: u64,
    pub draft_round_count: u64,
    /// The head's drafts, including the wait for the head's forward from
    /// the round before, which runs when its first logits are read.
    pub draft_nanoseconds: u64,
    /// The target's forward over the last chosen token and the drafts.
    pub verify_nanoseconds: u64,
    /// Choosing the target's token after every verified row.
    pub pick_nanoseconds: u64,
    /// Moving the state back to the kept rows and queuing the head's
    /// forward over them.
    pub keep_nanoseconds: u64,
    /// What MLX handed between the caller, the GPU and its CPU stream
    /// while the reply was written.
    pub hand_offs: EvaluationCounts,
}

impl ReplyStageCounts {
    /// The lookups' time that was not reading: their host bookkeeping.
    #[must_use]
    pub const fn lookup_bookkeeping_nanoseconds(&self) -> u64 {
        self.lookup_nanoseconds
            .saturating_sub(self.lookup_storage_read_nanoseconds)
    }
}

/// The stages of one drafted round.
#[derive(Clone, Copy, Debug, Default)]
pub struct DraftRoundStages {
    pub draft: Duration,
    pub verify: Duration,
    pub pick: Duration,
    pub keep: Duration,
}

/// The running tally of one model: the reply being written, and the last
/// one finished.
#[derive(Debug, Default)]
pub struct ReplyStageTally {
    counts: Cell<ReplyStageCounts>,
    /// `None` while no reply is being written: a request that ends while
    /// its prompt is still being read has no reply to close.
    hand_offs_when_the_reply_began: Cell<Option<EvaluationCounts>>,
    last_reply: Cell<Option<ReplyStageCounts>>,
}

/// What measuring a reply needs from its model.
#[derive(Debug, Default)]
pub struct ReplyMeasurement {
    /// Where the time of the reply being written went, and the last
    /// finished reply's.
    pub stage_tally: ReplyStageTally,
    /// On only while a measurement prices a route stop: every layer then
    /// copies its route to the host even when its table holds the layer
    /// whole.
    pub brings_every_route_to_the_host: Cell<bool>,
}

fn nanoseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

impl ReplyStageTally {
    /// Starts a reply's tally from nothing. The prompt's own evaluations
    /// stay out of it: they were made before this.
    pub fn begin_reply(&self) {
        self.counts.set(ReplyStageCounts::default());
        self.hand_offs_when_the_reply_began
            .set(Some(evaluation_counts()));
    }

    pub fn record_forward(&self, row_count: usize) {
        let mut counts = self.counts.get();
        counts.reply_forward_count += 1;
        counts.reply_row_count += u64::try_from(row_count).unwrap_or(u64::MAX);
        self.counts.set(counts);
    }

    /// One layer's lookup: all of its time, and the part spent reading.
    pub fn record_lookup(&self, lookup_elapsed: Duration, storage_read_elapsed: Duration) {
        let mut counts = self.counts.get();
        counts.lookup_count += 1;
        counts.lookup_nanoseconds += nanoseconds(lookup_elapsed);
        counts.lookup_storage_read_nanoseconds += nanoseconds(storage_read_elapsed);
        self.counts.set(counts);
    }

    pub fn record_draft_round(&self, stages: DraftRoundStages) {
        let mut counts = self.counts.get();
        counts.draft_round_count += 1;
        counts.draft_nanoseconds += nanoseconds(stages.draft);
        counts.verify_nanoseconds += nanoseconds(stages.verify);
        counts.pick_nanoseconds += nanoseconds(stages.pick);
        counts.keep_nanoseconds += nanoseconds(stages.keep);
        self.counts.set(counts);
    }

    /// Closes the reply's tally, keeps it as the last reply's and returns
    /// it; `None`, changing nothing, when no reply was begun.
    pub fn end_reply(&self) -> Option<ReplyStageCounts> {
        let hand_offs_when_the_reply_began = self.hand_offs_when_the_reply_began.take()?;
        let mut counts = self.counts.take();
        counts.hand_offs = evaluation_counts().since(hand_offs_when_the_reply_began);
        self.last_reply.set(Some(counts));
        Some(counts)
    }

    /// The last finished reply's stages, or `None` before any reply.
    pub fn last_reply(&self) -> Option<ReplyStageCounts> {
        self.last_reply.get()
    }
}

#[cfg(test)]
#[path = "reply_stage_tally_tests.rs"]
mod reply_stage_tally_tests;
