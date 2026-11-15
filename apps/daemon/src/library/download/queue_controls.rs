//! The coordinator's side of the download list: adding a model, taking one
//! off, and starting the next model whenever the Library slot frees up.
//!
//! A child of the coordinator, like `installed_model_removal`, because every
//! change to the list must happen under the coordinator's command lock and
//! decide against its current job; a sibling module could reach neither.

use std::sync::Arc;

use crate::library::download::queue::{DownloadQueue, admit_into_queue};
use crate::library::{
    DiskCapacityQuery, DownloadCatalogEntry, DownloadJob, DownloadJobPublicErrorCode,
    DownloadJobState,
};

use super::{LibraryDownloadCoordinator, LibraryDownloadCoordinatorError};

/// What asking for a model did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DownloadEnqueueOutcome {
    /// The Library was free and the download began.
    Started,
    /// The model is the current download already.
    AlreadyCurrent,
    /// The model waits its turn; 1 means it downloads next.
    Queued { waiting_position: usize },
}

impl LibraryDownloadCoordinator {
    /// Starts the background loop that begins the next listed model each time
    /// a download ends, and once now so a list left over from before a
    /// restart carries on. The loop holds only a weak reference, so it stops
    /// when the coordinator is dropped.
    pub fn start_queue_driver(self: &Arc<Self>) {
        if self
            .queue_driver_started
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        let coordinator = Arc::downgrade(self);
        let queue_wakeup = Arc::clone(&self.queue_wakeup);
        queue_wakeup.notify_one();
        tokio::spawn(async move {
            loop {
                queue_wakeup.notified().await;
                let Some(coordinator) = coordinator.upgrade() else {
                    return;
                };
                if let Err(advance_error) = coordinator.advance_queue().await {
                    tracing::warn!(error = %advance_error, "the next listed model could not start");
                }
            }
        });
    }

    /// Asks for a model: starts it when the Library is free and nothing is
    /// waiting, otherwise adds it to the end of the list.
    pub async fn enqueue(
        &self,
        huggingface_id: &str,
    ) -> Result<DownloadEnqueueOutcome, LibraryDownloadCoordinatorError> {
        let _command_guard = self.command_lock.lock().await;
        self.clear_finished_task().await;
        let catalog_entry = self.catalog_entry(huggingface_id)?;
        let current_job = self.current_job().await?;
        let running_huggingface_id = self.running_huggingface_id().await;
        if current_model_id(current_job.as_ref(), running_huggingface_id.as_deref())
            == Some(huggingface_id)
        {
            return Ok(DownloadEnqueueOutcome::AlreadyCurrent);
        }
        let mut download_queue = self.queue_store.load();
        if let Some(waiting_position) = download_queue.waiting_position(huggingface_id) {
            return Ok(DownloadEnqueueOutcome::Queued { waiting_position });
        }
        let slot_is_free =
            !self.has_active_task().await && current_job.as_ref().is_none_or(is_failed_job);
        if slot_is_free && !download_queue.has_waiting_models() {
            self.start_listed_model(catalog_entry, current_job.as_ref(), &mut download_queue)
                .await?;
            self.queue_store.save(&download_queue)?;
            return Ok(DownloadEnqueueOutcome::Started);
        }
        let current_remaining_bytes =
            self.current_remaining_bytes(current_job.as_ref(), running_huggingface_id.as_deref());
        self.admit_onto_disk(&download_queue, current_remaining_bytes, &catalog_entry)?;
        let waiting_position = download_queue.enqueue(huggingface_id);
        self.queue_store.save(&download_queue)?;
        if slot_is_free {
            self.queue_wakeup.notify_one();
        }
        Ok(DownloadEnqueueOutcome::Queued { waiting_position })
    }

    /// Takes a waiting or failed model off the list. Returns whether it was
    /// listed. The current download is stopped with `cancel`, not here.
    pub async fn remove_from_queue(
        &self,
        huggingface_id: &str,
    ) -> Result<bool, LibraryDownloadCoordinatorError> {
        let _command_guard = self.command_lock.lock().await;
        let mut download_queue = self.queue_store.load();
        let was_listed = download_queue.remove(huggingface_id);
        if was_listed {
            self.queue_store.save(&download_queue)?;
        }
        Ok(was_listed)
    }

    /// The list as it stands now.
    #[must_use]
    pub fn queue_snapshot(&self) -> DownloadQueue {
        self.queue_store.load()
    }

    /// The catalog's size of a model, for showing how much a list will fetch.
    #[must_use]
    pub fn catalog_size_bytes(&self, huggingface_id: &str) -> Option<u64> {
        self.catalog_entry(huggingface_id)
            .ok()
            .map(|catalog_entry| catalog_entry.approximate_size_bytes())
    }

    /// Starts the next waiting model when nothing holds the Library. A paused
    /// download holds it on purpose: pausing pauses the whole list.
    async fn advance_queue(&self) -> Result<(), LibraryDownloadCoordinatorError> {
        let _command_guard = self.command_lock.lock().await;
        self.clear_finished_task().await;
        if self.has_active_task().await {
            return Ok(());
        }
        let current_job = self.current_job().await?;
        if current_job
            .as_ref()
            .is_some_and(|download_job| !is_failed_job(download_job))
        {
            return Ok(());
        }
        let mut download_queue = self.queue_store.load();
        while let Some(next_huggingface_id) = download_queue.take_next() {
            match self.catalog_entry(&next_huggingface_id) {
                Ok(catalog_entry) => {
                    self.start_listed_model(
                        catalog_entry,
                        current_job.as_ref(),
                        &mut download_queue,
                    )
                    .await?;
                    break;
                }
                // The catalog changed under the list (an update removed the
                // model), so it can never download; drop it and go on.
                Err(_) => tracing::warn!(
                    huggingface_id = %next_huggingface_id,
                    "a listed model left the catalog and was dropped from the download list"
                ),
            }
        }
        self.queue_store.save(&download_queue)?;
        Ok(())
    }

    /// Begins `catalog_entry` now. A failed download of another model gives
    /// way to it, and is kept on the list as failed so the person sees it.
    async fn start_listed_model(
        &self,
        catalog_entry: DownloadCatalogEntry,
        current_job: Option<&DownloadJob>,
        download_queue: &mut DownloadQueue,
    ) -> Result<(), LibraryDownloadCoordinatorError> {
        if let Some(failed_job) = current_job.filter(|download_job| is_failed_job(download_job)) {
            download_queue.record_failure(
                failed_job.huggingface_id(),
                failed_job
                    .error_code()
                    .unwrap_or(DownloadJobPublicErrorCode::DownloadFailed),
            );
        }
        download_queue.remove(catalog_entry.huggingface_id());
        self.make_room_for(catalog_entry.huggingface_id()).await?;
        self.spawn_download_task(catalog_entry).await;
        Ok(())
    }

    /// Refuses a model when the current download, every waiting model and it
    /// would not fit on the disk together. A free-space reading that fails
    /// admits the model: each download still checks the disk itself.
    fn admit_onto_disk(
        &self,
        download_queue: &DownloadQueue,
        current_remaining_bytes: u64,
        catalog_entry: &DownloadCatalogEntry,
    ) -> Result<(), LibraryDownloadCoordinatorError> {
        let available_bytes = match self
            .capacity_query
            .available_space_bytes(self.job_store.models_directory())
        {
            Ok(available_bytes) => available_bytes,
            Err(capacity_error) => {
                tracing::warn!(%capacity_error, "free space could not be read; admitting the model to the list");
                return Ok(());
            }
        };
        let waiting_sizes_bytes: Vec<u64> = download_queue
            .waiting_huggingface_ids()
            .iter()
            .map(|waiting_id| self.catalog_size_bytes(waiting_id).unwrap_or(0))
            .collect();
        admit_into_queue(
            available_bytes,
            current_remaining_bytes,
            &waiting_sizes_bytes,
            catalog_entry.approximate_size_bytes(),
        )
        .map_err(LibraryDownloadCoordinatorError::QueueDoesNotFit)
    }

    /// Bytes the current download still has to fetch: its job's when the job
    /// is recorded, the catalog's size of the running model in the moment
    /// before that, and nothing for a failed job, which fetches no more.
    fn current_remaining_bytes(
        &self,
        current_job: Option<&DownloadJob>,
        running_huggingface_id: Option<&str>,
    ) -> u64 {
        match current_job {
            Some(download_job) if !is_failed_job(download_job) => {
                self.remaining_bytes_of(download_job)
            }
            Some(_) => 0,
            None => running_huggingface_id
                .and_then(|running_id| self.catalog_size_bytes(running_id))
                .unwrap_or(0),
        }
    }

    /// The model of the task running now, if one runs.
    async fn running_huggingface_id(&self) -> Option<String> {
        if !self.has_active_task().await {
            return None;
        }
        self.active_huggingface_id.lock().await.clone()
    }

    /// Bytes the current download still has to fetch: exact once its
    /// manifest is known, the catalog's estimate before that.
    fn remaining_bytes_of(&self, download_job: &DownloadJob) -> u64 {
        if download_job.has_exact_manifest() {
            return download_job.remaining_bytes();
        }
        self.catalog_size_bytes(download_job.huggingface_id())
            .unwrap_or(0)
    }

    pub(super) fn catalog_entry(
        &self,
        huggingface_id: &str,
    ) -> Result<DownloadCatalogEntry, LibraryDownloadCoordinatorError> {
        self.download_catalog
            .entries()
            .iter()
            .find(|catalog_entry| catalog_entry.huggingface_id() == huggingface_id)
            .cloned()
            .ok_or(LibraryDownloadCoordinatorError::CatalogEntryNotFound)
    }
}

/// The model holding the Library: the recorded job's, or the running task's
/// in the moment before its job is recorded.
fn current_model_id<'model>(
    current_job: Option<&'model DownloadJob>,
    running_huggingface_id: Option<&'model str>,
) -> Option<&'model str> {
    current_job
        .map(DownloadJob::huggingface_id)
        .or(running_huggingface_id)
}

fn is_failed_job(download_job: &DownloadJob) -> bool {
    download_job.state() == DownloadJobState::Failed
}
