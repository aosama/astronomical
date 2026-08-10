use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::PersistentPromptCacheSerializedFileWriter;
use super::write_queue::{
    PersistentPromptCachePublicationRequest, PersistentPromptCacheWriteRateLimiter,
    subtract_pending_serialized_bytes,
};

const WRITE_SLICE_BYTES: usize = 64 * 1024;

pub(super) struct ConfiguredPersistentPromptCacheSerializedFileWriter {
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

pub(super) fn run_writer_loop(
    disk_store: &PersistentPromptCacheDiskStore,
    serialized_block_receiver: Receiver<PersistentPromptCachePublicationRequest>,
    pending_serialized_bytes: &AtomicU64,
    pending_block_hashes: &Mutex<HashSet<[u8; 32]>>,
    shutdown_requested: Arc<AtomicBool>,
    write_rate_limiter: PersistentPromptCacheWriteRateLimiter,
) {
    let serialized_file_writer = ConfiguredPersistentPromptCacheSerializedFileWriter {
        write_rate_limiter,
        shutdown_requested: Arc::clone(&shutdown_requested),
    };
    while let Ok(publication_request) = serialized_block_receiver.recv() {
        let PersistentPromptCachePublicationRequest {
            serialized_block,
            publication_result_sender,
        } = publication_request;
        if shutdown_requested.load(Ordering::Acquire) {
            let _ = publication_result_sender.send(Err(
                PersistentPromptCacheDiskStoreError::WriterPublicationAcknowledgementLost,
            ));
            break;
        }
        let serialized_byte_count = serialized_block.serialized_byte_count();
        let persistent_prompt_cache_block_hash = serialized_block
            .persistent_prompt_cache_block_key
            .block_hash();
        let block_index = serialized_block
            .persistent_prompt_cache_block_key
            .block_index();
        // Sequence state is ancestral: publishing a child whose parent did not reach durable
        // storage would create a prefix that cannot be restored. Boundary-only layouts have no
        // sequence-state ancestry, so they do not require this check.
        if let Some(parent_block_key) = serialized_block
            .parent_persistent_prompt_cache_block_key
            .as_ref()
            && disk_store.model_contract_ref().has_sequence_state()
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
            let _ = publication_result_sender.send(Err(
                PersistentPromptCacheDiskStoreError::ParentStateNotPublished { block_index },
            ));
            continue;
        }
        let write_started_at = Instant::now();
        let save_outcome =
            disk_store.save_serialized_block(serialized_block, &serialized_file_writer);
        let write_elapsed_millis = write_started_at.elapsed().as_millis();
        // Release both reservations before acknowledging publication. The producer may be
        // blocked on this result, and it must observe capacity even when publication failed.
        subtract_pending_serialized_bytes(pending_serialized_bytes, serialized_byte_count);
        pending_block_hashes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&persistent_prompt_cache_block_hash);
        match save_outcome {
            Ok(save_admission) => {
                tracing::info!(
                    block_index,
                    serialized_byte_count,
                    write_elapsed_millis,
                    "persistent prompt-cache block write completed"
                );
                let _ = publication_result_sender.send(Ok(save_admission));
            }
            Err(save_error) => {
                tracing::warn!(block_index, serialized_byte_count, write_elapsed_millis, error = %save_error, "persistent prompt-cache block write failed");
                let _ = publication_result_sender.send(Err(save_error));
            }
        }
    }
}
