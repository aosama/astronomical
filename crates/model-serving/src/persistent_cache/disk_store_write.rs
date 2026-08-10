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

pub(crate) struct PersistentPromptCacheSerializedBlock {
    pub(crate) persistent_prompt_cache_block_key: PersistentPromptCacheBlockKey,
    pub(crate) parent_persistent_prompt_cache_block_key: Option<PersistentPromptCacheBlockKey>,
    pub(crate) serialized_sequence_state_bytes: Option<Vec<u8>>,
    pub(crate) serialized_boundary_state_bytes: Option<Vec<u8>>,
    pub(crate) save_admission: PersistentPromptCacheBlockSaveAdmission,
}

impl PersistentPromptCacheSerializedBlock {
    pub(crate) fn serialized_byte_count(&self) -> u64 {
        u64::try_from(
            self.serialized_sequence_state_bytes
                .as_ref()
                .map_or(0, Vec::len)
                .saturating_add(
                    self.serialized_boundary_state_bytes
                        .as_ref()
                        .map_or(0, Vec::len),
                ),
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
                usize::try_from(
                    self.global_prompt_cache_maximum_size_bytes
                        .min(runtime.memory_limits().active_memory_limit_bytes() as u64),
                )
                .unwrap_or(usize::MAX),
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
        maximum_capture_serialized_byte_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PersistentPromptCacheSerializationOutcome, PersistentPromptCacheDiskStoreError>
    {
        validate_state_kind_tensor_presence(
            "sequence",
            self.model_contract.has_sequence_state(),
            kv_block_tensors.len(),
        )?;
        validate_state_kind_tensor_presence(
            "boundary",
            self.model_contract.has_boundary_state(),
            recurrent_snapshot_tensors.len(),
        )?;
        let sequence_state_payload_bytes =
            u64::try_from(self.model_contract.sequence_state_payload_bytes_per_block())
                .unwrap_or(u64::MAX);
        let boundary_state_payload_bytes =
            u64::try_from(self.model_contract.boundary_state_payload_bytes()).unwrap_or(u64::MAX);
        let capture_payload_bytes =
            sequence_state_payload_bytes.saturating_add(boundary_state_payload_bytes);
        if capture_payload_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(PersistentPromptCacheDiskStoreError::SizeBoundExceeded {
                maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                estimated_block_bytes: capture_payload_bytes,
            });
        }
        let kv_block_already_exists = !self.model_contract.has_sequence_state()
            || self.has_kv_block(&persistent_prompt_cache_block_key.block_hash());
        let recurrent_snapshot_already_exists = !self.model_contract.has_boundary_state()
            || self.has_recurrent_snapshot(&persistent_prompt_cache_block_key.block_hash());
        let save_admission = {
            let tracked_files = self.lock_tracked_files();
            let reclaimable_parent_recurrent_snapshot_bytes =
                parent_persistent_prompt_cache_block_key
                    .filter(|parent_persistent_prompt_cache_block_key| {
                        !persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint(
                            parent_persistent_prompt_cache_block_key.block_index(),
                        )
                    })
                    .and_then(|parent_persistent_prompt_cache_block_key| {
                        tracked_files.recurrent_snapshot_file_size_bytes(
                            &parent_persistent_prompt_cache_block_key.block_hash(),
                        )
                    })
                    .unwrap_or(0);
            persistent_prompt_cache_save_admission(
                self.total_size_bytes(),
                sequence_state_payload_bytes,
                boundary_state_payload_bytes,
                reclaimable_parent_recurrent_snapshot_bytes,
                self.global_prompt_cache_maximum_size_bytes,
                kv_block_already_exists,
                recurrent_snapshot_already_exists,
            )
        };
        if save_admission == PersistentPromptCacheBlockSaveAdmission::SkipBecauseCacheIsFull {
            return Ok(PersistentPromptCacheSerializationOutcome::SkipBecauseCacheIsFull);
        }

        let serialized_sequence_state_bytes = if self.model_contract.has_sequence_state() {
            let serialized_bytes = performance_attribution.measure_operation(
                PerformanceOperation::PersistentPromptCacheKvBlockSerialization,
                |_performance_attribution| {
                    serialize_safetensors_file(
                        runtime,
                        kv_block_tensors,
                        &self.model_contract,
                        maximum_capture_serialized_byte_count,
                    )
                },
            )?;
            Some(serialized_bytes)
        } else {
            None
        };
        let serialized_boundary_state_bytes = if self.model_contract.has_boundary_state() {
            let remaining_serialization_byte_count = maximum_capture_serialized_byte_count
                .checked_sub(serialized_sequence_state_bytes.as_ref().map_or(0, Vec::len))
                .ok_or(
                    PersistentPromptCacheDiskStoreError::SerializedCaptureByteLimitExceeded {
                        maximum_capture_serialized_byte_count,
                    },
                )?;
            let serialized_bytes = performance_attribution.measure_operation(
                PerformanceOperation::PersistentPromptCacheRecurrentSnapshotSerialization,
                |_performance_attribution| {
                    serialize_safetensors_file(
                        runtime,
                        recurrent_snapshot_tensors,
                        &self.model_contract,
                        remaining_serialization_byte_count,
                    )
                },
            )?;
            Some(serialized_bytes)
        } else {
            None
        };
        Ok(PersistentPromptCacheSerializationOutcome::Serialized(
            Box::new(PersistentPromptCacheSerializedBlock {
                persistent_prompt_cache_block_key: persistent_prompt_cache_block_key.clone(),
                parent_persistent_prompt_cache_block_key: parent_persistent_prompt_cache_block_key
                    .cloned(),
                serialized_sequence_state_bytes,
                serialized_boundary_state_bytes,
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
        let mut newly_saved_file_paths = Vec::new();
        let sequence_state_file_path = if let Some(serialized_sequence_state_bytes) =
            serialized_block.serialized_sequence_state_bytes.as_ref()
        {
            let file_path = save_serialized_safetensors_file(
                &self.kv_blocks_directory,
                persistent_prompt_cache_block_key.block_hash(),
                serialized_sequence_state_bytes,
                serialized_file_writer,
            )?;
            newly_saved_file_paths.push(file_path.clone());
            Some(file_path)
        } else {
            None
        };
        let boundary_state_file_path = if let Some(serialized_boundary_state_bytes) =
            serialized_block.serialized_boundary_state_bytes.as_ref()
        {
            match save_serialized_safetensors_file(
                &self.recurrent_snapshots_directory,
                persistent_prompt_cache_block_key.block_hash(),
                serialized_boundary_state_bytes,
                serialized_file_writer,
            ) {
                Ok(file_path) => {
                    newly_saved_file_paths.push(file_path.clone());
                    Some(file_path)
                }
                Err(boundary_state_save_error) => {
                    return Err(self.rollback_newly_saved_files_after_error(
                        &newly_saved_file_paths,
                        boundary_state_save_error,
                    ));
                }
            }
        } else {
            None
        };
        self.finish_serialized_block_save(
            persistent_prompt_cache_block_key,
            parent_persistent_prompt_cache_block_key,
            sequence_state_file_path,
            boundary_state_file_path,
            serialized_block.save_admission,
        )
    }

    fn finish_serialized_block_save(
        &self,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
        sequence_state_file_path: Option<PathBuf>,
        boundary_state_file_path: Option<PathBuf>,
        save_admission: PersistentPromptCacheBlockSaveAdmission,
    ) -> Result<PersistentPromptCacheBlockSaveAdmission, PersistentPromptCacheDiskStoreError> {
        let newly_saved_file_paths = sequence_state_file_path
            .iter()
            .chain(boundary_state_file_path.iter())
            .cloned()
            .collect::<Vec<_>>();
        let kv_block_file_size_bytes = read_optional_file_size_bytes(
            sequence_state_file_path.as_ref(),
            &newly_saved_file_paths,
            self,
        )?;
        let recurrent_snapshot_file_size_bytes = read_optional_file_size_bytes(
            boundary_state_file_path.as_ref(),
            &newly_saved_file_paths,
            self,
        )?;
        let actual_save_bytes =
            kv_block_file_size_bytes.saturating_add(recurrent_snapshot_file_size_bytes);
        if actual_save_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(self.rollback_newly_saved_files_after_error(
                &newly_saved_file_paths,
                PersistentPromptCacheDiskStoreError::SizeBoundExceeded {
                    maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                    estimated_block_bytes: actual_save_bytes,
                },
            ));
        }
        let mut tracked_files = self.lock_tracked_files();
        if let Some(sequence_state_file_path) = sequence_state_file_path {
            tracked_files.insert_file(
                PersistentPromptCacheFileKind::SequenceStateBlock,
                persistent_prompt_cache_block_key.block_hash(),
                TrackedPersistentPromptCacheFile {
                    file_path: sequence_state_file_path,
                    file_size_bytes: kv_block_file_size_bytes,
                },
            );
        }
        if let Some(boundary_state_file_path) = boundary_state_file_path {
            tracked_files.insert_file(
                PersistentPromptCacheFileKind::BoundaryStateSnapshot,
                persistent_prompt_cache_block_key.block_hash(),
                TrackedPersistentPromptCacheFile {
                    file_path: boundary_state_file_path,
                    file_size_bytes: recurrent_snapshot_file_size_bytes,
                },
            );
        }
        let reclaimable_parent_snapshot = if save_admission.should_reclaim_parent_boundary()
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

fn validate_state_kind_tensor_presence(
    state_kind: &'static str,
    expected_present: bool,
    actual_tensor_count: usize,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    if expected_present == (actual_tensor_count > 0) {
        return Ok(());
    }
    Err(
        PersistentPromptCacheDiskStoreError::StateKindTensorPresenceMismatch {
            state_kind,
            expected_present,
            actual_tensor_count,
        },
    )
}

fn read_optional_file_size_bytes(
    file_path: Option<&PathBuf>,
    newly_saved_file_paths: &[PathBuf],
    persistent_prompt_cache_disk_store: &PersistentPromptCacheDiskStore,
) -> Result<u64, PersistentPromptCacheDiskStoreError> {
    let Some(file_path) = file_path else {
        return Ok(0);
    };
    read_file_size_bytes(file_path).map_err(|metadata_error| {
        persistent_prompt_cache_disk_store
            .rollback_newly_saved_files_after_error(newly_saved_file_paths, metadata_error)
    })
}
