//! The list of models waiting to download one after another.

pub(crate) mod download_queue;
pub(crate) mod download_queue_store;
pub(crate) mod queue_disk_admission;

#[cfg(test)]
mod download_queue_store_tests;
#[cfg(test)]
mod download_queue_tests;
#[cfg(test)]
mod queue_disk_admission_tests;

pub use download_queue::DownloadQueue;
pub use download_queue_store::{DownloadQueueStore, DownloadQueueStoreError};
pub use queue_disk_admission::{QueueDiskShortfall, admit_into_queue};
