//! The background writer that puts a prompt's `astro_ssd_prompt_cache`
//! blocks on disk while the engine goes on to the reply.
//!
//! Writing a block inside the prompt loop held the first token back: on a
//! 15,000-token prompt of Ornith 35B-A3B (2026-11-27, M5 Pro) five blocks
//! took ~450 ms after the last layer (198 ms serializing key-value state,
//! 187 ms serializing recurrent snapshots, 38 ms of directory syncs). The
//! engine thread now only copies a block's tensors to host memory and queues
//! them; one writer thread publishes the queue in order, so a parent is
//! always written before its children and `require_published_parent` still
//! holds. A block whose write fails is logged and dropped; its children then
//! fail that parent check and are dropped too, while the reply, which already
//! holds the state in memory, continues.
//!
//! The store's readers drain the queue first (`pending_publications`), so a
//! later request always finds what an earlier one queued. Dropping the queue
//! writes everything still queued, then stops the writer.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

use model_primitives::performance_attribution::PerformanceAttribution;

use super::block_key::AstroSsdPromptCacheBlockKey;
use super::disk_store::AstroSsdPromptCacheDiskStore;
use super::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use super::host_state_tensor::{AstroSsdPromptCacheHostTensors, host_tensors_payload_byte_count};
use super::pending_publications::AstroSsdPromptCachePendingPublications;

/// Host memory queued blocks may hold before the engine waits for the writer.
/// A 4,096-token block of Ornith 35B-A3B is about 150 MB, so three to four
/// blocks wait while the first is written; past that the engine writes at the
/// disk's pace, as it did before the queue.
pub const MAXIMUM_QUEUED_PUBLICATION_PAYLOAD_BYTES: usize = 512 * 1_048_576;

const WRITER_THREAD_NAME: &str = "astro-ssd-prompt-cache-writer";

/// One block to publish, with its state already copied to the host.
pub struct AstroSsdPromptCachePublicationJob {
    pub block_key: AstroSsdPromptCacheBlockKey,
    pub parent_block_key: Option<AstroSsdPromptCacheBlockKey>,
    pub sequence_state_tensors: AstroSsdPromptCacheHostTensors,
    pub boundary_state_tensors: AstroSsdPromptCacheHostTensors,
}

impl AstroSsdPromptCachePublicationJob {
    #[must_use]
    pub fn payload_byte_count(&self) -> usize {
        host_tensors_payload_byte_count(&self.sequence_state_tensors)
            + host_tensors_payload_byte_count(&self.boundary_state_tensors)
    }
}

/// A queued block and its place in the pending count, given back when it is
/// dropped: after its write, after a failed write, or while unwinding from a
/// panic on the writer. A reader waiting for the queue to drain therefore
/// never waits on a block no one will finish.
struct QueuedPublication {
    publication_job: AstroSsdPromptCachePublicationJob,
    pending_publications: Arc<AstroSsdPromptCachePendingPublications>,
    payload_byte_count: usize,
}

impl Drop for QueuedPublication {
    fn drop(&mut self) {
        self.pending_publications.complete(self.payload_byte_count);
    }
}

/// The queue and the writer thread that empties it.
pub struct AstroSsdPromptCachePublicationQueue {
    job_sender: Option<Sender<QueuedPublication>>,
    writer_thread: Option<JoinHandle<()>>,
    pending_publications: Arc<AstroSsdPromptCachePendingPublications>,
    maximum_queued_payload_bytes: usize,
}

impl AstroSsdPromptCachePublicationQueue {
    /// Starts the writer for `astro_ssd_prompt_cache`.
    ///
    /// # Errors
    /// Fails when the operating system refuses a new thread.
    pub fn start(
        astro_ssd_prompt_cache: Arc<AstroSsdPromptCacheDiskStore>,
        maximum_queued_payload_bytes: usize,
    ) -> std::io::Result<Self> {
        let (job_sender, job_receiver) = mpsc::channel();
        let pending_publications = astro_ssd_prompt_cache.pending_publications();
        let writer_thread = thread::Builder::new()
            .name(WRITER_THREAD_NAME.to_owned())
            .spawn(move || run_publication_writer(&astro_ssd_prompt_cache, &job_receiver))?;
        Ok(Self {
            job_sender: Some(job_sender),
            writer_thread: Some(writer_thread),
            pending_publications,
            maximum_queued_payload_bytes,
        })
    }

    /// Queues `publication_job`, first waiting for room when queued blocks
    /// already hold the bound.
    ///
    /// # Errors
    /// Fails when the writer has stopped, which only a panic on the writer
    /// thread can cause.
    pub fn enqueue(
        &self,
        publication_job: AstroSsdPromptCachePublicationJob,
    ) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
        let payload_byte_count = publication_job.payload_byte_count();
        self.pending_publications
            .admit(payload_byte_count, self.maximum_queued_payload_bytes);
        let queued_publication = QueuedPublication {
            publication_job,
            pending_publications: Arc::clone(&self.pending_publications),
            payload_byte_count,
        };
        // A refused send hands the block back inside the error, and dropping
        // it there gives its count back.
        let Some(job_sender) = self.job_sender.as_ref() else {
            return Err(AstroSsdPromptCacheDiskStoreError::PublicationWriterStopped);
        };
        job_sender
            .send(queued_publication)
            .map_err(|_| AstroSsdPromptCacheDiskStoreError::PublicationWriterStopped)
    }

    /// Returns once every queued block is written.
    pub fn wait_until_drained(&self) {
        self.pending_publications.wait_until_drained();
    }
}

impl Drop for AstroSsdPromptCachePublicationQueue {
    fn drop(&mut self) {
        // Closing the channel lets the writer finish what is queued and
        // return; joining keeps a model unload from racing its own writes.
        drop(self.job_sender.take());
        if let Some(writer_thread) = self.writer_thread.take()
            && writer_thread.join().is_err()
        {
            tracing::error!("the astro_ssd_prompt_cache writer thread panicked");
        }
    }
}

fn run_publication_writer(
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    job_receiver: &Receiver<QueuedPublication>,
) {
    while let Ok(queued_publication) = job_receiver.recv() {
        publish_queued_block(astro_ssd_prompt_cache, &queued_publication.publication_job);
    }
}

fn publish_queued_block(
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    publication_job: &AstroSsdPromptCachePublicationJob,
) {
    let mut performance_attribution = PerformanceAttribution::disabled();
    if let Err(publication_error) = astro_ssd_prompt_cache
        .publish_host_block_with_performance_attribution(
            &publication_job.block_key,
            publication_job.parent_block_key.as_ref(),
            &publication_job.sequence_state_tensors,
            &publication_job.boundary_state_tensors,
            &mut performance_attribution,
        )
    {
        tracing::warn!(
            block_index = publication_job.block_key.block_index(),
            token_count = publication_job.block_key.token_count(),
            %publication_error,
            "astro_ssd_prompt_cache block not written; the reply continues from memory"
        );
    }
}

#[cfg(test)]
#[path = "publication_queue_tests.rs"]
mod publication_queue_tests;
