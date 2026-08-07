use std::collections::HashMap;
use std::path::PathBuf;

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::{PerformanceAttribution, PerformanceOperation};

use super::block_key::PersistentPromptCacheBlockKey;
use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    ImmediatePersistentPromptCacheSerializedFileWriter, PersistentPromptCacheFileKind,
    PersistentPromptCacheSerializedFileWriter, read_file_size_bytes,
    remove_cache_owned_file_or_confirm_absent, save_serialized_safetensors_file,
    serialize_safetensors_file,
};
use super::disk_store_index::TrackedPersistentPromptCacheFile;
use super::save_admission::{
    PersistentPromptCacheBlockSaveAdmission,
    persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint,
    persistent_prompt_cache_save_admission,
};

pub(super) const ESTIMATED_SAFETENSORS_FILE_OVERHEAD_BYTES: u64 = 16 * 1024;

pub(crate) struct PersistentPromptCacheSerializedBlock {
    pub(crate) persistent_prompt_cache_block_key: PersistentPromptCacheBlockKey,
    pub(crate) parent_persistent_prompt_cache_block_key: Option<PersistentPromptCacheBlockKey>,
    pub(crate) serialized_kv_block_bytes: Vec<u8>,
    pub(crate) serialized_recurrent_snapshot_bytes: Vec<u8>,
    pub(crate) save_admission: PersistentPromptCacheBlockSaveAdmission,
}

impl PersistentPromptCacheSerializedBlock {
    pub(crate) fn serialized_byte_count(&self) -> u64 {
        u64::try_from(
            self.serialized_kv_block_bytes
                .len()
                .saturating_add(self.serialized_recurrent_snapshot_bytes.len()),
        )
        .unwrap_or(u64::MAX)
    }
}

pub(crate) enum PersistentPromptCacheSerializationOutcome {
    SkipBecauseCacheIsFull,
    Serialized(Box<PersistentPromptCacheSerializedBlock>),
}

impl PersistentPromptCacheDiskStore {
    pub fn save_kv_block_and_recurrent_snapshot(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
        kv_block_tensors: &HashMap<String, MlxArray>,
        recurrent_snapshot_tensors: &HashMap<String, MlxArray>,
    ) -> Result<PersistentPromptCacheBlockSaveAdmission, PersistentPromptCacheDiskStoreError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.save_kv_block_and_recurrent_snapshot_with_performance_attribution(
            runtime,
            persistent_prompt_cache_block_key,
            parent_persistent_prompt_cache_block_key,
            kv_block_tensors,
            recurrent_snapshot_tensors,
            &mut disabled_performance_attribution,
        )
    }

    pub(crate) fn save_kv_block_and_recurrent_snapshot_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
        kv_block_tensors: &HashMap<String, MlxArray>,
        recurrent_snapshot_tensors: &HashMap<String, MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PersistentPromptCacheBlockSaveAdmission, PersistentPromptCacheDiskStoreError> {
        let serialization_outcome = self
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
            return Ok(PersistentPromptCacheBlockSaveAdmission::SkipBecauseCacheIsFull);
        };
        self.save_serialized_block(
            *serialized_block,
            &ImmediatePersistentPromptCacheSerializedFileWriter,
        )
    }

    pub(crate) fn serialize_kv_block_and_recurrent_snapshot_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
        kv_block_tensors: &HashMap<String, MlxArray>,
        recurrent_snapshot_tensors: &HashMap<String, MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PersistentPromptCacheSerializationOutcome, PersistentPromptCacheDiskStoreError>
    {
        let estimated_kv_block_bytes =
            estimated_serialized_safetensors_file_byte_count(kv_block_tensors);
        let estimated_recurrent_snapshot_bytes =
            estimated_serialized_safetensors_file_byte_count(recurrent_snapshot_tensors);
        let estimated_save_bytes =
            estimated_kv_block_bytes.saturating_add(estimated_recurrent_snapshot_bytes);
        if estimated_save_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(PersistentPromptCacheDiskStoreError::SizeBoundExceeded {
                maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                estimated_block_bytes: estimated_save_bytes,
            });
        }
        let parent_recurrent_snapshot_can_be_reclaimed = parent_persistent_prompt_cache_block_key
            .is_some_and(|parent_persistent_prompt_cache_block_key| {
                !persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(
                    parent_persistent_prompt_cache_block_key.block_index(),
                )
            });
        let kv_block_already_exists =
            self.has_kv_block(&persistent_prompt_cache_block_key.block_hash());
        let save_admission = {
            let tracked_files = self.lock_tracked_files();
            let reclaimable_parent_recurrent_snapshot_bytes =
                if parent_recurrent_snapshot_can_be_reclaimed {
                    parent_persistent_prompt_cache_block_key
                        .and_then(|parent_persistent_prompt_cache_block_key| {
                            tracked_files.recurrent_snapshot_file_size_bytes(
                                &parent_persistent_prompt_cache_block_key.block_hash(),
                            )
                        })
                        .unwrap_or(0)
                } else {
                    0
                };
            persistent_prompt_cache_save_admission(
                self.total_size_bytes(),
                estimated_kv_block_bytes,
                estimated_recurrent_snapshot_bytes,
                reclaimable_parent_recurrent_snapshot_bytes,
                self.global_prompt_cache_maximum_size_bytes,
                persistent_prompt_cache_block_key.block_index(),
                kv_block_already_exists,
            )
        };
        if save_admission == PersistentPromptCacheBlockSaveAdmission::SkipBecauseCacheIsFull {
            return Ok(PersistentPromptCacheSerializationOutcome::SkipBecauseCacheIsFull);
        }

        let serialized_kv_block_bytes = performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCacheKvBlockSerialization,
            |_performance_attribution| {
                serialize_safetensors_file(
                    runtime,
                    kv_block_tensors,
                    self.model_contract.model_id(),
                    self.model_contract.model_revision(),
                )
            },
        )?;
        let serialized_recurrent_snapshot_bytes = performance_attribution.measure_operation(
            PerformanceOperation::PersistentPromptCacheRecurrentSnapshotSerialization,
            |_performance_attribution| {
                serialize_safetensors_file(
                    runtime,
                    recurrent_snapshot_tensors,
                    self.model_contract.model_id(),
                    self.model_contract.model_revision(),
                )
            },
        )?;
        Ok(PersistentPromptCacheSerializationOutcome::Serialized(
            Box::new(PersistentPromptCacheSerializedBlock {
                persistent_prompt_cache_block_key: persistent_prompt_cache_block_key.clone(),
                parent_persistent_prompt_cache_block_key: parent_persistent_prompt_cache_block_key
                    .cloned(),
                serialized_kv_block_bytes,
                serialized_recurrent_snapshot_bytes,
                save_admission,
            }),
        ))
    }

    pub(crate) fn save_serialized_block(
        &self,
        serialized_block: PersistentPromptCacheSerializedBlock,
        serialized_file_writer: &dyn PersistentPromptCacheSerializedFileWriter,
    ) -> Result<PersistentPromptCacheBlockSaveAdmission, PersistentPromptCacheDiskStoreError> {
        let _write_operation_guard = self.lock_write_operations();
        self.prepare_active_model_storage_directories()?;
        let persistent_prompt_cache_block_key = &serialized_block.persistent_prompt_cache_block_key;
        let parent_persistent_prompt_cache_block_key = serialized_block
            .parent_persistent_prompt_cache_block_key
            .as_ref();
        let parent_recurrent_snapshot_can_be_reclaimed = parent_persistent_prompt_cache_block_key
            .is_some_and(|parent_block_key| {
                !persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(
                    parent_block_key.block_index(),
                )
            });
        let kv_block_file_path = save_serialized_safetensors_file(
            &self.kv_blocks_directory,
            persistent_prompt_cache_block_key.block_hash(),
            &serialized_block.serialized_kv_block_bytes,
            serialized_file_writer,
        )?;
        let recurrent_snapshot_file_path = match save_serialized_safetensors_file(
            &self.recurrent_snapshots_directory,
            persistent_prompt_cache_block_key.block_hash(),
            &serialized_block.serialized_recurrent_snapshot_bytes,
            serialized_file_writer,
        ) {
            Ok(file_path) => file_path,
            Err(recurrent_save_error) => {
                return Err(self.rollback_newly_saved_files_after_error(
                    &[kv_block_file_path],
                    recurrent_save_error,
                ));
            }
        };
        self.finish_serialized_block_save(
            persistent_prompt_cache_block_key,
            parent_persistent_prompt_cache_block_key,
            parent_recurrent_snapshot_can_be_reclaimed,
            kv_block_file_path,
            recurrent_snapshot_file_path,
            serialized_block.save_admission,
        )
    }

    fn finish_serialized_block_save(
        &self,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
        parent_recurrent_snapshot_can_be_reclaimed: bool,
        kv_block_file_path: PathBuf,
        recurrent_snapshot_file_path: PathBuf,
        save_admission: PersistentPromptCacheBlockSaveAdmission,
    ) -> Result<PersistentPromptCacheBlockSaveAdmission, PersistentPromptCacheDiskStoreError> {
        let kv_block_file_size_bytes = match read_file_size_bytes(&kv_block_file_path) {
            Ok(file_size_bytes) => file_size_bytes,
            Err(metadata_error) => {
                return Err(self.rollback_newly_saved_files_after_error(
                    &[kv_block_file_path, recurrent_snapshot_file_path],
                    metadata_error,
                ));
            }
        };
        let recurrent_snapshot_file_size_bytes =
            match read_file_size_bytes(&recurrent_snapshot_file_path) {
                Ok(file_size_bytes) => file_size_bytes,
                Err(metadata_error) => {
                    return Err(self.rollback_newly_saved_files_after_error(
                        &[kv_block_file_path, recurrent_snapshot_file_path],
                        metadata_error,
                    ));
                }
            };
        let actual_save_bytes =
            kv_block_file_size_bytes.saturating_add(recurrent_snapshot_file_size_bytes);
        if actual_save_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(self.rollback_newly_saved_files_after_error(
                &[kv_block_file_path, recurrent_snapshot_file_path],
                PersistentPromptCacheDiskStoreError::SizeBoundExceeded {
                    maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                    estimated_block_bytes: actual_save_bytes,
                },
            ));
        }
        let newly_saved_file_paths = [
            kv_block_file_path.clone(),
            recurrent_snapshot_file_path.clone(),
        ];
        let mut tracked_files = self.lock_tracked_files();
        tracked_files.insert_file(
            PersistentPromptCacheFileKind::SequenceStateBlock,
            persistent_prompt_cache_block_key.block_hash(),
            TrackedPersistentPromptCacheFile {
                file_path: kv_block_file_path,
                file_size_bytes: kv_block_file_size_bytes,
            },
        );
        tracked_files.insert_file(
            PersistentPromptCacheFileKind::BoundaryStateSnapshot,
            persistent_prompt_cache_block_key.block_hash(),
            TrackedPersistentPromptCacheFile {
                file_path: recurrent_snapshot_file_path,
                file_size_bytes: recurrent_snapshot_file_size_bytes,
            },
        );
        let reclaimable_parent_snapshot = if parent_recurrent_snapshot_can_be_reclaimed
            && let Some(parent_block_key) = parent_persistent_prompt_cache_block_key
            && parent_block_key.block_hash() != persistent_prompt_cache_block_key.block_hash()
            && let Some(parent_snapshot_file) = tracked_files.file(
                PersistentPromptCacheFileKind::BoundaryStateSnapshot,
                &parent_block_key.block_hash(),
            ) {
            Some((
                parent_block_key.block_hash(),
                parent_snapshot_file.file_path.clone(),
            ))
        } else {
            None
        };
        drop(tracked_files);
        if let Some((parent_snapshot_hash, parent_snapshot_file_path)) = reclaimable_parent_snapshot
        {
            if let Err(parent_removal_error) =
                remove_cache_owned_file_or_confirm_absent(&parent_snapshot_file_path)
            {
                return Err(self.rollback_newly_saved_files_after_error(
                    &newly_saved_file_paths,
                    parent_removal_error,
                ));
            }
            self.lock_tracked_files().remove_file(
                PersistentPromptCacheFileKind::BoundaryStateSnapshot,
                &parent_snapshot_hash,
            );
        }
        if let Err(global_quota_error) = self.enforce_global_prompt_cache_quota() {
            return Err(self.rollback_newly_saved_files_after_error(
                &newly_saved_file_paths,
                global_quota_error,
            ));
        }
        Ok(save_admission)
    }
}

pub(super) fn estimated_serialized_safetensors_file_byte_count(
    tensors: &HashMap<String, MlxArray>,
) -> u64 {
    let tensor_payload_byte_count = tensors
        .values()
        .map(|tensor| u64::try_from(tensor.byte_count()).unwrap_or(u64::MAX))
        .fold(0_u64, u64::saturating_add);
    tensor_payload_byte_count.saturating_add(ESTIMATED_SAFETENSORS_FILE_OVERHEAD_BYTES)
}
