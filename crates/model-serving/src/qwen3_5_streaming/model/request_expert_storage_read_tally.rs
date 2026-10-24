//! Per-request tally of expert payload bytes read from storage.
//!
//! The observed residency label (issue #1533) must reflect reads that
//! actually happened during the active request, independent of performance
//! attribution. Keeping the tally in its own type gives the label exactly one
//! writer at the storage boundary and one reset at request admission, rather
//! than a raw counter field that every streaming call site updates.

use std::cell::Cell;

#[derive(Debug, Default)]
pub(crate) struct RequestExpertStorageReadTally {
    expert_storage_read_byte_count: Cell<u64>,
}

impl RequestExpertStorageReadTally {
    /// Starts a fresh observation window for a newly admitted request.
    pub(crate) fn begin_request(&self) {
        self.expert_storage_read_byte_count.set(0);
    }

    /// Records one expert page streamed from storage.
    pub(crate) fn record_streamed_page(&self, streamed_page_payload_byte_count: u64) {
        self.expert_storage_read_byte_count.set(
            self.expert_storage_read_byte_count
                .get()
                .saturating_add(streamed_page_payload_byte_count),
        );
    }

    /// Expert payload bytes streamed from storage since the request began.
    pub(crate) fn expert_storage_read_byte_count(&self) -> u64 {
        self.expert_storage_read_byte_count.get()
    }
}

#[cfg(test)]
#[path = "request_expert_storage_read_tally_tests.rs"]
mod tests;
