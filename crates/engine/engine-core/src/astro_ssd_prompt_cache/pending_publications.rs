//! The `astro_ssd_prompt_cache` blocks handed to the background writer and
//! not yet on disk, shared by the writer, the engine thread that queues
//! blocks, and every reader that must see what was queued before it.
//!
//! Two rules ride on it. Admission bounds the host memory queued blocks hold:
//! a block waits for room unless the queue is empty, so a block larger than
//! the bound still goes through and nothing is ever dropped (a full queue
//! degrades to writing at the disk's pace, the old synchronous behaviour).
//! Draining is the read barrier: a prefix lookup, a block load or a clear
//! first waits until every queued block is written, so a request never misses
//! a block an earlier request queued.

use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

#[derive(Debug, Default)]
struct QueuedPublications {
    block_count: usize,
    payload_bytes: usize,
}

/// Counts of queued blocks, with a signal on every change.
#[derive(Debug, Default)]
pub struct AstroSsdPromptCachePendingPublications {
    queued_publications: Mutex<QueuedPublications>,
    queue_changed: Condvar,
}

impl AstroSsdPromptCachePendingPublications {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Counts one block of `payload_bytes` as queued, first waiting while
    /// other blocks are queued and adding this one would pass
    /// `maximum_queued_payload_bytes`.
    pub fn admit(&self, payload_bytes: usize, maximum_queued_payload_bytes: usize) {
        let mut queued_publications = self.lock_queued_publications();
        while queued_publications.block_count > 0
            && queued_publications
                .payload_bytes
                .saturating_add(payload_bytes)
                > maximum_queued_payload_bytes
        {
            queued_publications = self
                .queue_changed
                .wait(queued_publications)
                .unwrap_or_else(PoisonError::into_inner);
        }
        queued_publications.block_count += 1;
        queued_publications.payload_bytes = queued_publications
            .payload_bytes
            .saturating_add(payload_bytes);
    }

    /// Counts one admitted block of `payload_bytes` as finished, written or
    /// not, and wakes everyone waiting on the queue.
    pub fn complete(&self, payload_bytes: usize) {
        let mut queued_publications = self.lock_queued_publications();
        queued_publications.block_count = queued_publications.block_count.saturating_sub(1);
        queued_publications.payload_bytes = queued_publications
            .payload_bytes
            .saturating_sub(payload_bytes);
        drop(queued_publications);
        self.queue_changed.notify_all();
    }

    /// Returns once no block is queued.
    pub fn wait_until_drained(&self) {
        let mut queued_publications = self.lock_queued_publications();
        while queued_publications.block_count > 0 {
            queued_publications = self
                .queue_changed
                .wait(queued_publications)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    #[must_use]
    pub fn queued_block_count(&self) -> usize {
        self.lock_queued_publications().block_count
    }

    #[must_use]
    pub fn queued_payload_bytes(&self) -> usize {
        self.lock_queued_publications().payload_bytes
    }

    fn lock_queued_publications(&self) -> MutexGuard<'_, QueuedPublications> {
        self.queued_publications
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
#[path = "pending_publications_tests.rs"]
mod pending_publications_tests;
