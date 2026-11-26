//! How often, and how lately, each slot of a layer's table was read: the
//! order the baseline decode policy gives experts up in.
//!
//! The expert read the fewest times since it was seated goes first, and
//! among equals the one read longest ago. Counts start again at every reply:
//! the experts a prompt read most are not the ones its reply reads most, and
//! counts carried over kept them past their use. Replays of a recorded
//! Shakespeare reply on Flash-Next at 34 GB put the gain at 2.6% fewer
//! reads from storage a token (18.72 to 18.24), and leaving the order of
//! equals to the sort cost 0.5 to 0.8 a token, so recency breaks ties.

/// One table's read bookkeeping, a slot per entry.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct SlotReads {
    read_count_by_slot: Vec<u64>,
    /// When each slot was last read or seated, on this table's own clock.
    last_read_by_slot: Vec<u64>,
    /// Advances at every read of the table and every seating, so a later
    /// event always has a larger value.
    table_clock: u64,
}

impl SlotReads {
    /// A table of `slot_count` slots, every one seated now with no reads.
    pub(super) fn seated(slot_count: usize) -> Self {
        Self {
            read_count_by_slot: vec![0; slot_count],
            last_read_by_slot: vec![0; slot_count],
            table_clock: 0,
        }
    }

    /// The table grew to `slot_count` slots; the new ones hold nothing yet.
    pub(super) fn grow_to(&mut self, slot_count: usize) {
        self.read_count_by_slot.resize(slot_count, 0);
        self.last_read_by_slot.resize(slot_count, 0);
    }

    /// An expert was seated in `slot`: no reads, and newer than every
    /// expert not read since.
    pub(super) fn seat(&mut self, slot: usize) {
        self.table_clock += 1;
        if let (Some(read_count), Some(last_read)) = (
            self.read_count_by_slot.get_mut(slot),
            self.last_read_by_slot.get_mut(slot),
        ) {
            *read_count = 0;
            *last_read = self.table_clock;
        }
    }

    /// The experts in `slots` were read by one routed step of the table.
    pub(super) fn read(&mut self, slots: &[usize]) {
        self.table_clock += 1;
        for &slot in slots {
            if let (Some(read_count), Some(last_read)) = (
                self.read_count_by_slot.get_mut(slot),
                self.last_read_by_slot.get_mut(slot),
            ) {
                *read_count = read_count.saturating_add(1);
                *last_read = self.table_clock;
            }
        }
    }

    /// A reply begins: every count starts again, while the order in which
    /// slots were last read stays, so equals still go oldest first.
    pub(super) fn start_reply(&mut self) {
        self.read_count_by_slot.fill(0);
    }

    /// The order the baseline policy gives `slot` up in: fewest reads,
    /// then read longest ago. A slot past the table counts as never given up.
    pub(super) fn giving_up_order(&self, slot: usize) -> (u64, u64) {
        match (
            self.read_count_by_slot.get(slot),
            self.last_read_by_slot.get(slot),
        ) {
            (Some(&read_count), Some(&last_read)) => (read_count, last_read),
            _ => (u64::MAX, u64::MAX),
        }
    }
}

#[cfg(test)]
#[path = "slot_reads_tests.rs"]
mod slot_reads_tests;
