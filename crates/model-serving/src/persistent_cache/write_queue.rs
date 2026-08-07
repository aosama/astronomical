use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::PerformanceAttribution;

use super::block_key::PersistentPromptCacheBlockKey;
use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::PersistentPromptCacheSerializedFileWriter;
use super::disk_store_write::{
    ESTIMATED_SAFETENSORS_FILE_OVERHEAD_BYTES, PersistentPromptCacheSerializationOutcome,
    PersistentPromptCacheSerializedBlock, estimated_serialized_safetensors_file_byte_count,
};

const MAXIMUM_PENDING_SERIALIZED_BYTES: u64 = 256_000_000;
const PENDING_WRITE_RENDEZVOUS_CAPACITY: usize = 0;
const WRITE_SLICE_BYTES: usize = 64 * 1024;
const WRITER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentPromptCacheWriteQueueOutcome {
    Queued,
    AlreadyQueued,
    DroppedBecauseQueueIsFull,
    SkipBecauseCacheIsFull,
}

/// Pure rate policy used by the filesystem writer and hermetic tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentPromptCacheWriteRateLimiter {
    maximum_bytes_per_second: Option<u64>,
}

#[must_use]
pub fn persistent_prompt_cache_write_queue_can_accept(
    pending_serialized_bytes: u64,
    incoming_serialized_bytes: u64,
) -> bool {
    pending_serialized_bytes
        .checked_add(incoming_serialized_bytes)
        .is_some_and(|projected_pending_serialized_bytes| {
            projected_pending_serialized_bytes <= MAXIMUM_PENDING_SERIALIZED_BYTES
        })
}

impl PersistentPromptCacheWriteRateLimiter {
    #[must_use]
    pub fn new(ssd_write_rate_megabytes_per_second: Option<u64>) -> Self {
        Self {
            maximum_bytes_per_second: ssd_write_rate_megabytes_per_second
                .filter(|write_rate| *write_rate > 0)
                .map(|write_rate| write_rate.saturating_mul(1_000_000).max(1)),
        }
    }

    #[must_use]
    pub const fn maximum_bytes_per_second(&self) -> Option<u64> {
        self.maximum_bytes_per_second
    }

    #[must_use]
    pub fn minimum_elapsed_for_bytes(&self, written_byte_count: u64) -> Duration {
        let Some(maximum_bytes_per_second) = self.maximum_bytes_per_second else {
            return Duration::ZERO;
        };
        Duration::from_secs_f64(written_byte_count as f64 / maximum_bytes_per_second as f64)
    }
}

struct ConfiguredPersistentPromptCacheSerializedFileWriter {
    write_rate_limiter: PersistentPromptCacheWriteRateLimiter,
    shutdown_requested: Arc<AtomicBool>,
}

impl PersistentPromptCacheSerializedFileWriter
    for ConfiguredPersistentPromptCacheSerializedFileWriter
{
    fn write_serialized_file(
        &self,
        output_file: &mut std::fs::File,
        serialized_safetensors_bytes: &[u8],
    ) -> std::io::Result<()> {
        let write_started_at = Instant::now();
        let mut written_byte_count = 0_u64;
        for serialized_safetensors_slice in serialized_safetensors_bytes.chunks(WRITE_SLICE_BYTES) {
            if self.shutdown_requested.load(Ordering::Acquire) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "persistent prompt-cache writer shutdown interrupted the queued write",
                ));
            }
            let next_written_byte_count = written_byte_count.saturating_add(
                u64::try_from(serialized_safetensors_slice.len()).unwrap_or(u64::MAX),
            );
            let minimum_elapsed = self
                .write_rate_limiter
                .minimum_elapsed_for_bytes(next_written_byte_count);
            let actual_elapsed = write_started_at.elapsed();
            if minimum_elapsed > actual_elapsed {
                std::thread::sleep(minimum_elapsed - actual_elapsed);
            }
            output_file.write_all(serialized_safetensors_slice)?;
            written_byte_count = next_written_byte_count;
        }
        Ok(())
    }
}

/// Bounded write-behind owner for persistent prompt-cache files.
///
/// MLX serialization remains synchronous on the engine owner thread. Only
/// ordinary byte vectors cross to this filesystem thread, preserving the
/// repository's single-owner MLX architecture.
pub struct PersistentPromptCacheWriteQueue {
    disk_store: Arc<PersistentPromptCacheDiskStore>,
    pending_serialized_bytes: Arc<AtomicU64>,
    pending_block_hashes: Arc<Mutex<HashSet<[u8; 32]>>>,
    serialized_block_sender: Mutex<Option<SyncSender<PersistentPromptCacheSerializedBlock>>>,
    shutdown_requested: Arc<AtomicBool>,
    writer_completed_receiver: Receiver<()>,
    writer_thread: Option<JoinHandle<()>>,
}

impl PersistentPromptCacheWriteQueue {
    pub fn new(
        disk_store: Arc<PersistentPromptCacheDiskStore>,
        ssd_write_rate_megabytes_per_second: Option<u64>,
    ) -> Result<Self, PersistentPromptCacheDiskStoreError> {
        let (serialized_block_sender, serialized_block_receiver) =
            sync_channel(PENDING_WRITE_RENDEZVOUS_CAPACITY);
        let (writer_completed_sender, writer_completed_receiver) = sync_channel(1);
        let pending_serialized_bytes = Arc::new(AtomicU64::new(0));
        let pending_block_hashes = Arc::new(Mutex::new(HashSet::new()));
        let shutdown_requested = Arc::new(AtomicBool::new(false));
        let writer_disk_store = Arc::clone(&disk_store);
        let writer_pending_serialized_bytes = Arc::clone(&pending_serialized_bytes);
        let writer_pending_block_hashes = Arc::clone(&pending_block_hashes);
        let writer_shutdown_requested = Arc::clone(&shutdown_requested);
        let write_rate_limiter =
            PersistentPromptCacheWriteRateLimiter::new(ssd_write_rate_megabytes_per_second);
        let writer_thread = std::thread::Builder::new()
            .name("astronomical-prompt-cache-writer".to_owned())
            .spawn(move || {
                run_writer_loop(
                    &writer_disk_store,
                    serialized_block_receiver,
                    &writer_pending_serialized_bytes,
                    &writer_pending_block_hashes,
                    writer_shutdown_requested,
                    write_rate_limiter,
                );
                let _completion_receiver_was_dropped = writer_completed_sender.send(()).is_err();
            })
            .map_err(|source| PersistentPromptCacheDiskStoreError::StartWriterThread { source })?;
        Ok(Self {
            disk_store,
            pending_serialized_bytes,
            pending_block_hashes,
            serialized_block_sender: Mutex::new(Some(serialized_block_sender)),
            shutdown_requested,
            writer_completed_receiver,
            writer_thread: Some(writer_thread),
        })
    }

    pub fn serialize_and_enqueue(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
        kv_block_tensors: &HashMap<String, MlxArray>,
        recurrent_snapshot_tensors: &HashMap<String, MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PersistentPromptCacheWriteQueueOutcome, PersistentPromptCacheDiskStoreError> {
        let persistent_prompt_cache_block_hash = persistent_prompt_cache_block_key.block_hash();
        if self
            .pending_block_hashes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(&persistent_prompt_cache_block_hash)
        {
            return Ok(PersistentPromptCacheWriteQueueOutcome::AlreadyQueued);
        }
        let estimated_serialized_byte_count =
            estimated_serialized_safetensors_file_byte_count(kv_block_tensors).saturating_add(
                estimated_serialized_safetensors_file_byte_count(recurrent_snapshot_tensors),
            );
        if !persistent_prompt_cache_write_queue_can_accept(
            self.pending_serialized_bytes.load(Ordering::Acquire),
            estimated_serialized_byte_count,
        ) {
            return Ok(PersistentPromptCacheWriteQueueOutcome::DroppedBecauseQueueIsFull);
        }
        let serialization_outcome = self
            .disk_store
            .serialize_kv_block_and_recurrent_snapshot_with_performance_attribution(
                runtime,
                persistent_prompt_cache_block_key,
                parent_persistent_prompt_cache_block_key,
                kv_block_tensors,
                recurrent_snapshot_tensors,
                performance_attribution,
            )?;
        let PersistentPromptCacheSerializationOutcome::Serialized(serialized_block) =
            serialization_outcome
        else {
            return Ok(PersistentPromptCacheWriteQueueOutcome::SkipBecauseCacheIsFull);
        };
        let serialized_byte_count = serialized_block.serialized_byte_count();
        if !reserve_pending_serialized_bytes(&self.pending_serialized_bytes, serialized_byte_count)
        {
            return Ok(PersistentPromptCacheWriteQueueOutcome::DroppedBecauseQueueIsFull);
        }
        let sender_guard = self
            .serialized_block_sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(serialized_block_sender) = sender_guard.as_ref() else {
            subtract_pending_serialized_bytes(
                &self.pending_serialized_bytes,
                serialized_byte_count,
            );
            return Ok(PersistentPromptCacheWriteQueueOutcome::DroppedBecauseQueueIsFull);
        };
        self.pending_block_hashes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(persistent_prompt_cache_block_hash);
        let writer_handoff_started_at = Instant::now();
        match serialized_block_sender.send(*serialized_block) {
            Ok(()) => Ok(PersistentPromptCacheWriteQueueOutcome::Queued),
            Err(_) => {
                subtract_pending_serialized_bytes(
                    &self.pending_serialized_bytes,
                    serialized_byte_count,
                );
                self.pending_block_hashes
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .remove(&persistent_prompt_cache_block_hash);
                Ok(PersistentPromptCacheWriteQueueOutcome::DroppedBecauseQueueIsFull)
            }
        }
        .inspect(|write_queue_outcome| {
            tracing::info!(
                block_index = persistent_prompt_cache_block_key.block_index(),
                writer_handoff_elapsed_millis = writer_handoff_started_at.elapsed().as_millis(),
                outcome = ?write_queue_outcome,
                "persistent prompt-cache serialized block handoff completed"
            );
        })
    }

    #[must_use]
    pub fn can_accept_projected_captures(
        &self,
        projected_single_capture_tensor_payload_bytes: usize,
        projected_boundary_count: usize,
    ) -> bool {
        let projected_single_capture_tensor_payload_bytes =
            u64::try_from(projected_single_capture_tensor_payload_bytes).unwrap_or(u64::MAX);
        let projected_single_capture_serialized_bytes =
            projected_single_capture_tensor_payload_bytes
                .saturating_add(ESTIMATED_SAFETENSORS_FILE_OVERHEAD_BYTES.saturating_mul(2));
        let projected_all_capture_serialized_bytes = u64::try_from(projected_boundary_count)
            .unwrap_or(u64::MAX)
            .saturating_mul(projected_single_capture_serialized_bytes);
        !self.shutdown_requested.load(Ordering::Acquire)
            && self
                .serialized_block_sender
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_some()
            && persistent_prompt_cache_write_queue_can_accept(
                self.pending_serialized_bytes.load(Ordering::Acquire),
                projected_single_capture_serialized_bytes,
            )
            && self
                .disk_store
                .total_size_bytes()
                .checked_add(projected_all_capture_serialized_bytes)
                .is_some_and(|projected_total_size_bytes| {
                    projected_total_size_bytes
                        <= self.disk_store.global_prompt_cache_maximum_size_bytes
                })
    }

    #[must_use]
    pub fn pending_serialized_bytes(&self) -> u64 {
        self.pending_serialized_bytes.load(Ordering::Acquire)
    }

    #[doc(hidden)]
    pub fn disconnect_writer_for_tests(&mut self) {
        self.serialized_block_sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
    }

    #[doc(hidden)]
    pub fn request_writer_shutdown_for_tests(&self) {
        self.shutdown_requested.store(true, Ordering::Release);
    }
}

impl Drop for PersistentPromptCacheWriteQueue {
    fn drop(&mut self) {
        self.serialized_block_sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if self
            .writer_completed_receiver
            .recv_timeout(WRITER_SHUTDOWN_TIMEOUT)
            .is_err()
        {
            self.shutdown_requested.store(true, Ordering::Release);
            tracing::warn!(
                timeout_seconds = WRITER_SHUTDOWN_TIMEOUT.as_secs(),
                "persistent prompt-cache writer did not drain before shutdown; abandoning queued cache work"
            );
            return;
        }
        if let Some(writer_thread) = self.writer_thread.take()
            && writer_thread.join().is_err()
        {
            tracing::warn!("persistent prompt-cache writer thread panicked during shutdown");
        }
    }
}

fn run_writer_loop(
    disk_store: &PersistentPromptCacheDiskStore,
    serialized_block_receiver: Receiver<PersistentPromptCacheSerializedBlock>,
    pending_serialized_bytes: &AtomicU64,
    pending_block_hashes: &Mutex<HashSet<[u8; 32]>>,
    shutdown_requested: Arc<AtomicBool>,
    write_rate_limiter: PersistentPromptCacheWriteRateLimiter,
) {
    let serialized_file_writer = ConfiguredPersistentPromptCacheSerializedFileWriter {
        write_rate_limiter,
        shutdown_requested: Arc::clone(&shutdown_requested),
    };
    while let Ok(serialized_block) = serialized_block_receiver.recv() {
        if shutdown_requested.load(Ordering::Acquire) {
            break;
        }
        let serialized_byte_count = serialized_block.serialized_byte_count();
        let persistent_prompt_cache_block_hash = serialized_block
            .persistent_prompt_cache_block_key
            .block_hash();
        let block_index = serialized_block
            .persistent_prompt_cache_block_key
            .block_index();
        if let Some(parent_block_key) = serialized_block
            .parent_persistent_prompt_cache_block_key
            .as_ref()
            && !disk_store.has_kv_block(&parent_block_key.block_hash())
        {
            subtract_pending_serialized_bytes(pending_serialized_bytes, serialized_byte_count);
            pending_block_hashes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&persistent_prompt_cache_block_hash);
            tracing::warn!(
                block_index,
                "dropping queued persistent prompt-cache child because its parent was not published"
            );
            continue;
        }
        let write_started_at = Instant::now();
        let save_outcome =
            disk_store.save_serialized_block(serialized_block, &serialized_file_writer);
        let write_elapsed_millis = write_started_at.elapsed().as_millis();
        subtract_pending_serialized_bytes(pending_serialized_bytes, serialized_byte_count);
        pending_block_hashes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&persistent_prompt_cache_block_hash);
        match save_outcome {
            Ok(_) => tracing::info!(
                block_index,
                serialized_byte_count,
                write_elapsed_millis,
                "persistent prompt-cache block write completed"
            ),
            Err(save_error) => tracing::warn!(
                block_index,
                serialized_byte_count,
                write_elapsed_millis,
                error = %save_error,
                "persistent prompt-cache block write failed"
            ),
        }
    }
}

fn reserve_pending_serialized_bytes(
    pending_serialized_bytes: &AtomicU64,
    serialized_byte_count: u64,
) -> bool {
    let mut observed_pending_serialized_bytes = pending_serialized_bytes.load(Ordering::Acquire);
    loop {
        if !persistent_prompt_cache_write_queue_can_accept(
            observed_pending_serialized_bytes,
            serialized_byte_count,
        ) {
            return false;
        }
        let projected_pending_serialized_bytes =
            observed_pending_serialized_bytes.saturating_add(serialized_byte_count);
        match pending_serialized_bytes.compare_exchange_weak(
            observed_pending_serialized_bytes,
            projected_pending_serialized_bytes,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(updated_pending_serialized_bytes) => {
                observed_pending_serialized_bytes = updated_pending_serialized_bytes;
            }
        }
    }
}

fn subtract_pending_serialized_bytes(
    pending_serialized_bytes: &AtomicU64,
    serialized_byte_count: u64,
) {
    let _ = pending_serialized_bytes.fetch_update(
        Ordering::AcqRel,
        Ordering::Acquire,
        |pending_byte_count| Some(pending_byte_count.saturating_sub(serialized_byte_count)),
    );
}
