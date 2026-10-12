//! In-memory progress observed by local user interfaces while payload bytes stream.
//!
//! Restart recovery derives authoritative progress from staged file lengths, so UI refreshes must
//! not force synchronized metadata transactions into the network hot path.

use std::sync::{Arc, RwLock};

use super::DownloadJob;

/// Process-local projection of the active download's newest received-byte count.
#[derive(Clone, Default)]
pub struct DownloadProgressSnapshot {
    current_job: Arc<RwLock<Option<DownloadJob>>>,
}

impl DownloadProgressSnapshot {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn publish(&self, download_job: DownloadJob) {
        *self
            .current_job
            .write()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) = Some(download_job);
    }

    pub fn clear(&self) {
        *self
            .current_job
            .write()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) = None;
    }

    #[must_use]
    pub fn current(&self) -> Option<DownloadJob> {
        self.current_job
            .read()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
            .clone()
    }
}
