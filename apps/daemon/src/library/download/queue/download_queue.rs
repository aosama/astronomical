//! The models waiting to download after the current one, in the order the
//! person asked for them, and the ones that could not finish.
//!
//! A single download already keeps several ranged requests in flight and
//! fills the connection, so models download one at a time: two at once would
//! split the same bandwidth and both would finish later. The list lets a
//! person press Get on several models in a row instead of waiting for each.

use serde::{Deserialize, Serialize};

use crate::library::DownloadJobPublicErrorCode;

/// The waiting and failed models, oldest request first.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadQueue {
    waiting_huggingface_ids: Vec<String>,
    failed_entries: Vec<FailedQueueEntry>,
}

/// A queued model whose download stopped with an error. It stays listed so
/// the person sees it and can try again or remove it; the models behind it
/// carry on.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FailedQueueEntry {
    pub huggingface_id: String,
    pub error_code: DownloadJobPublicErrorCode,
}

impl DownloadQueue {
    /// Adds a model to the end of the list and returns its 1-based place
    /// among the waiting models. A model already waiting keeps its place; a
    /// failed one leaves the failed list, since asking again means "try
    /// again".
    pub fn enqueue(&mut self, huggingface_id: &str) -> usize {
        self.failed_entries
            .retain(|failed_entry| failed_entry.huggingface_id != huggingface_id);
        if let Some(waiting_position) = self.waiting_position(huggingface_id) {
            return waiting_position;
        }
        self.waiting_huggingface_ids.push(huggingface_id.to_owned());
        self.waiting_huggingface_ids.len()
    }

    /// The 1-based place of a waiting model, or `None` when it is not waiting.
    #[must_use]
    pub fn waiting_position(&self, huggingface_id: &str) -> Option<usize> {
        self.waiting_huggingface_ids
            .iter()
            .position(|waiting_id| waiting_id == huggingface_id)
            .map(|waiting_index| waiting_index + 1)
    }

    /// Takes a model off the list, waiting or failed. Returns whether it was
    /// listed at all.
    pub fn remove(&mut self, huggingface_id: &str) -> bool {
        let listed_count_before = self.listed_count();
        self.waiting_huggingface_ids
            .retain(|waiting_id| waiting_id != huggingface_id);
        self.failed_entries
            .retain(|failed_entry| failed_entry.huggingface_id != huggingface_id);
        self.listed_count() != listed_count_before
    }

    /// Takes the oldest waiting model, the next to download.
    pub fn take_next(&mut self) -> Option<String> {
        if self.waiting_huggingface_ids.is_empty() {
            return None;
        }
        Some(self.waiting_huggingface_ids.remove(0))
    }

    /// Records a model whose download stopped with an error, replacing any
    /// earlier failure of the same model.
    pub fn record_failure(&mut self, huggingface_id: &str, error_code: DownloadJobPublicErrorCode) {
        self.failed_entries
            .retain(|failed_entry| failed_entry.huggingface_id != huggingface_id);
        self.failed_entries.push(FailedQueueEntry {
            huggingface_id: huggingface_id.to_owned(),
            error_code,
        });
    }

    #[must_use]
    pub fn waiting_huggingface_ids(&self) -> &[String] {
        &self.waiting_huggingface_ids
    }

    #[must_use]
    pub fn failed_entries(&self) -> &[FailedQueueEntry] {
        &self.failed_entries
    }

    #[must_use]
    pub fn has_waiting_models(&self) -> bool {
        !self.waiting_huggingface_ids.is_empty()
    }

    fn listed_count(&self) -> usize {
        self.waiting_huggingface_ids.len() + self.failed_entries.len()
    }
}
