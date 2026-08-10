use std::collections::{HashMap, HashSet};
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
use super::disk_store_write::{
    PersistentPromptCacheSerializationOutcome, PersistentPromptCacheSerializedBlock,
};
use super::save_admission::PersistentPromptCacheBlockSaveAdmission;
use super::write_queue_writer::run_writer_loop;

const PENDING_WRITE_RENDEZVOUS_CAPACITY: usize = 0;
const WRITER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentPromptCacheWriteQueueOutcome {
    Queued,
    AlreadyQueued,
    DroppedBecauseQueueIsFull,
    SkipBecauseCacheIsFull,
}

pub(super) struct PersistentPromptCachePublicationRequest {
    pub(super) serialized_block: PersistentPromptCacheSerializedBlock,
    pub(super) publication_result_sender: SyncSender<
        Result<PersistentPromptCacheBlockSaveAdmission, PersistentPromptCacheDiskStoreError>,
    >,
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
    maximum_pending_serialized_bytes: u64,
) -> bool {
    pending_serialized_bytes
        .checked_add(incoming_serialized_bytes)
        .is_some_and(|projected_pending_serialized_bytes| {
            projected_pending_serialized_bytes <= maximum_pending_serialized_bytes
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

/// Bounded write-behind owner for persistent prompt-cache files.
///
/// MLX serialization remains synchronous on the engine owner thread. Only
/// ordinary byte vectors cross to this filesystem thread, preserving the
/// repository's single-owner MLX architecture.
pub struct PersistentPromptCacheWriteQueue {
    disk_store: Arc<PersistentPromptCacheDiskStore>,
    pending_serialized_bytes: Arc<AtomicU64>,
    pending_block_hashes: Arc<Mutex<HashSet<[u8; 32]>>>,
    serialized_block_sender: Mutex<Option<SyncSender<PersistentPromptCachePublicationRequest>>>,
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
        let persistent_prompt_cache_model_contract = self.disk_store.model_contract_ref();
        let sequence_state_is_published = !persistent_prompt_cache_model_contract
            .has_sequence_state()
            || self
                .disk_store
                .has_kv_block(&persistent_prompt_cache_block_hash);
        let boundary_state_is_published = !persistent_prompt_cache_model_contract
            .has_boundary_state()
            || self
                .disk_store
                .has_recurrent_snapshot(&persistent_prompt_cache_block_hash);
        if sequence_state_is_published && boundary_state_is_published {
            return Ok(PersistentPromptCacheWriteQueueOutcome::AlreadyQueued);
        }
        if self
            .pending_block_hashes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(&persistent_prompt_cache_block_hash)
        {
            return Ok(PersistentPromptCacheWriteQueueOutcome::AlreadyQueued);
        }
        let capture_tensor_payload_bytes = kv_block_tensors
            .values()
            .chain(recurrent_snapshot_tensors.values())
            .map(|persistent_state_tensor| {
                u64::try_from(persistent_state_tensor.byte_count()).unwrap_or(u64::MAX)
            })
            .fold(0_u64, u64::saturating_add);
        // Reject before requesting MLX serialization whenever the live runtime cannot reserve
        // the capture plus its contract workspace. SafeTensors overhead is intentionally not
        // guessed here: the exact serialized size is checked again after MLX produces it.
        let maximum_capture_serialized_bytes = self.available_capture_serialized_bytes(runtime)?;
        if capture_tensor_payload_bytes > maximum_capture_serialized_bytes {
            self.log_capture_rejection(
                runtime,
                persistent_prompt_cache_block_key,
                "tensor_payload_exceeds_live_capture_capacity_before_serialization",
                capture_tensor_payload_bytes,
                maximum_capture_serialized_bytes,
                None,
            );
            return Ok(PersistentPromptCacheWriteQueueOutcome::DroppedBecauseQueueIsFull);
        }
        let maximum_capture_serialized_byte_count =
            usize::try_from(maximum_capture_serialized_bytes).unwrap_or(usize::MAX);
        let serialization_outcome = self
            .disk_store
            .serialize_kv_block_and_recurrent_snapshot_with_performance_attribution(
                runtime,
                persistent_prompt_cache_block_key,
                parent_persistent_prompt_cache_block_key,
                kv_block_tensors,
                recurrent_snapshot_tensors,
                maximum_capture_serialized_byte_count,
                performance_attribution,
            )?;
        let PersistentPromptCacheSerializationOutcome::Serialized(serialized_block) =
            serialization_outcome
        else {
            self.log_capture_rejection(
                runtime,
                persistent_prompt_cache_block_key,
                "ssd_quota_rejected_serialized_capture",
                capture_tensor_payload_bytes,
                maximum_capture_serialized_bytes,
                None,
            );
            return Ok(PersistentPromptCacheWriteQueueOutcome::SkipBecauseCacheIsFull);
        };
        let serialized_byte_count = serialized_block.serialized_byte_count();
        // Serialization can add real header and alignment bytes that no static estimate can
        // safely predict. Re-sample admission before handing ordinary bytes to the writer.
        if !self.capture_memory_admission_is_available(runtime, serialized_byte_count)? {
            self.log_capture_rejection(
                runtime,
                persistent_prompt_cache_block_key,
                "serialized_bytes_exceed_live_capture_capacity_after_serialization",
                capture_tensor_payload_bytes,
                maximum_capture_serialized_bytes,
                Some(serialized_byte_count),
            );
            return Ok(PersistentPromptCacheWriteQueueOutcome::DroppedBecauseQueueIsFull);
        }
        let memory_snapshot = runtime.memory_snapshot().map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadMlxMemorySnapshot { source }
        })?;
        let maximum_pending_serialized_bytes = self.maximum_pending_serialized_bytes(
            runtime,
            memory_snapshot.active_memory_bytes(),
            memory_snapshot.allocator_cache_memory_bytes(),
        );
        if !reserve_pending_serialized_bytes(
            &self.pending_serialized_bytes,
            serialized_byte_count,
            maximum_pending_serialized_bytes,
        ) {
            self.log_capture_rejection(
                runtime,
                persistent_prompt_cache_block_key,
                "pending_serialized_bytes_exceed_live_capture_capacity",
                capture_tensor_payload_bytes,
                maximum_capture_serialized_bytes,
                Some(serialized_byte_count),
            );
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
        let (publication_result_sender, publication_result_receiver) = sync_channel(0);
        let publication_request = PersistentPromptCachePublicationRequest {
            serialized_block: *serialized_block,
            publication_result_sender,
        };
        match serialized_block_sender.send(publication_request) {
            // A child cache key is meaningful only after the parent state is durable. This
            // acknowledgement therefore supplies bounded backpressure instead of letting the
            // engine race ahead and later discover an unpublishable parent chain.
            Ok(()) => match publication_result_receiver.recv() {
                Ok(Ok(_)) => Ok(PersistentPromptCacheWriteQueueOutcome::Queued),
                Ok(Err(publication_error)) => Err(publication_error),
                Err(_) => {
                    Err(PersistentPromptCacheDiskStoreError::WriterPublicationAcknowledgementLost)
                }
            },
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
        let projected_single_capture_payload_bytes =
            u64::try_from(projected_single_capture_tensor_payload_bytes).unwrap_or(u64::MAX);
        let projected_all_capture_payload_bytes = u64::try_from(projected_boundary_count)
            .unwrap_or(u64::MAX)
            .saturating_mul(projected_single_capture_payload_bytes);
        let maximum_pending_serialized_bytes = self
            .disk_store
            .model_contract_ref()
            .effective_mlx_memory_ceiling_bytes()
            .saturating_sub(
                u64::try_from(
                    self.disk_store
                        .model_contract_ref()
                        .maximum_tensor_serialization_workspace_bytes(),
                )
                .unwrap_or(u64::MAX),
            );
        !self.shutdown_requested.load(Ordering::Acquire)
            && self
                .serialized_block_sender
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_some()
            && persistent_prompt_cache_write_queue_can_accept(
                self.pending_serialized_bytes.load(Ordering::Acquire),
                projected_all_capture_payload_bytes,
                maximum_pending_serialized_bytes,
            )
            && self
                .disk_store
                .total_size_bytes()
                .checked_add(projected_all_capture_payload_bytes)
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

    fn maximum_pending_serialized_bytes(
        &self,
        runtime: &MlxRuntime,
        active_memory_bytes: usize,
        allocator_cache_memory_bytes: usize,
    ) -> u64 {
        // Both MLX's active allocation limit and the artifact's effective ceiling constrain the
        // same process. The lower ceiling wins after subtracting live arrays, reclaimable
        // allocator storage, and the largest serializer workspace promised by the contract.
        let current_runtime_memory_bytes = u64::try_from(active_memory_bytes)
            .unwrap_or(u64::MAX)
            .saturating_add(u64::try_from(allocator_cache_memory_bytes).unwrap_or(u64::MAX));
        self.disk_store
            .model_contract_ref()
            .effective_mlx_memory_ceiling_bytes()
            .min(
                u64::try_from(runtime.memory_limits().active_memory_limit_bytes())
                    .unwrap_or(u64::MAX),
            )
            .saturating_sub(current_runtime_memory_bytes)
            .saturating_sub(
                u64::try_from(
                    self.disk_store
                        .model_contract_ref()
                        .maximum_tensor_serialization_workspace_bytes(),
                )
                .unwrap_or(u64::MAX),
            )
    }

    fn capture_memory_admission_is_available(
        &self,
        runtime: &MlxRuntime,
        next_capture_serialized_bytes: u64,
    ) -> Result<bool, PersistentPromptCacheDiskStoreError> {
        Ok(next_capture_serialized_bytes <= self.available_capture_serialized_bytes(runtime)?)
    }

    fn available_capture_serialized_bytes(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<u64, PersistentPromptCacheDiskStoreError> {
        let memory_snapshot = runtime.memory_snapshot().map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadMlxMemorySnapshot { source }
        })?;
        let maximum_pending_serialized_bytes = self.maximum_pending_serialized_bytes(
            runtime,
            memory_snapshot.active_memory_bytes(),
            memory_snapshot.allocator_cache_memory_bytes(),
        );
        Ok(maximum_pending_serialized_bytes
            .saturating_sub(self.pending_serialized_bytes.load(Ordering::Acquire)))
    }

    fn log_capture_rejection(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        rejection_stage: &'static str,
        capture_tensor_payload_bytes: u64,
        available_capture_serialized_bytes: u64,
        serialized_capture_bytes: Option<u64>,
    ) {
        let memory_snapshot = runtime.memory_snapshot().ok();
        let active_memory_bytes = memory_snapshot
            .as_ref()
            .map_or(usize::MAX, |snapshot| snapshot.active_memory_bytes());
        let allocator_cache_memory_bytes =
            memory_snapshot.as_ref().map_or(usize::MAX, |snapshot| {
                snapshot.allocator_cache_memory_bytes()
            });
        let peak_memory_bytes = memory_snapshot
            .as_ref()
            .map_or(usize::MAX, |snapshot| snapshot.peak_memory_bytes());
        let persistent_prompt_cache_model_contract = self.disk_store.model_contract_ref();
        tracing::error!(
            rejection_stage,
            block_index = persistent_prompt_cache_block_key.block_index(),
            active_memory_bytes,
            allocator_cache_memory_bytes,
            peak_memory_bytes,
            runtime_active_memory_limit_bytes = runtime.memory_limits().active_memory_limit_bytes(),
            contract_effective_mlx_memory_ceiling_bytes =
                persistent_prompt_cache_model_contract.effective_mlx_memory_ceiling_bytes(),
            maximum_tensor_serialization_workspace_bytes = persistent_prompt_cache_model_contract
                .maximum_tensor_serialization_workspace_bytes(),
            capture_tensor_payload_bytes,
            serialized_capture_bytes,
            available_capture_serialized_bytes,
            pending_serialized_bytes = self.pending_serialized_bytes.load(Ordering::Acquire),
            persistent_prompt_cache_disk_bytes = self.disk_store.total_size_bytes(),
            persistent_prompt_cache_quota_bytes =
                self.disk_store.global_prompt_cache_maximum_size_bytes,
            writer_shutdown_requested = self.shutdown_requested.load(Ordering::Acquire),
            "persistent prompt-cache capture admission rejected with complete live-memory evidence"
        );
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

fn reserve_pending_serialized_bytes(
    pending_serialized_bytes: &AtomicU64,
    serialized_byte_count: u64,
    maximum_pending_serialized_bytes: u64,
) -> bool {
    let mut observed_pending_serialized_bytes = pending_serialized_bytes.load(Ordering::Acquire);
    loop {
        if !persistent_prompt_cache_write_queue_can_accept(
            observed_pending_serialized_bytes,
            serialized_byte_count,
            maximum_pending_serialized_bytes,
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

pub(super) fn subtract_pending_serialized_bytes(
    pending_serialized_bytes: &AtomicU64,
    serialized_byte_count: u64,
) {
    let _ = pending_serialized_bytes.fetch_update(
        Ordering::AcqRel,
        Ordering::Acquire,
        |pending_byte_count| Some(pending_byte_count.saturating_sub(serialized_byte_count)),
    );
}
