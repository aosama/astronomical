use std::collections::HashMap;
use std::sync::Arc;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, PositionalFileReadMetrics};

use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    ImmediatePersistentPromptCacheSerializedFileWriter, PersistentPromptCacheFileKind,
    open_without_following_symlinks, read_file_size_bytes, save_serialized_safetensors_file,
};
use super::disk_store_index::TrackedPersistentPromptCacheFile;
use super::speculative_prefill_target_state::{
    PersistentSpeculativePrefillTargetStateContract,
    PersistentSpeculativePrefillTargetStateFileHeader, RestoredSpeculativePrefillTargetState,
    SPECULATIVE_PREFILL_TARGET_SELECTED_POSITIONS_TENSOR_NAME,
    longest_reusable_speculative_prefill_target_prefix, target_state_metadata_entries,
};

impl PersistentPromptCacheDiskStore {
    pub fn save_speculative_prefill_target_state(
        &self,
        runtime: &MlxRuntime,
        target_state_contract: &PersistentSpeculativePrefillTargetStateContract,
        prompt_prefix_token_ids: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        selected_target_token_positions: &MlxArray,
        decoder_state_tensors: &[(&str, &MlxArray)],
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        if target_state_contract.target_model_id() != self.model_contract_ref().model_id()
            || target_state_contract.target_model_revision()
                != self.model_contract_ref().model_revision()
            || prompt_prefix_token_ids.is_empty()
            || selected_target_token_positions.dtype() != MlxDtype::UInt32
            || selected_target_token_positions.shape().len() != 1
            || decoder_state_tensors.is_empty()
        {
            return Err(invalid_target_state_input());
        }
        let target_state_identity_hash = target_state_contract
            .target_state_identity_hash(prompt_prefix_token_ids, ordered_image_sha256_digests);
        let estimated_target_state_file_bytes = decoder_state_tensors
            .iter()
            .fold(
                selected_target_token_positions.byte_count() as u64,
                |estimated_bytes, (_, target_state_tensor)| {
                    estimated_bytes.saturating_add(target_state_tensor.byte_count() as u64)
                },
            )
            .saturating_add(super::disk_store_write::ESTIMATED_SAFETENSORS_FILE_OVERHEAD_BYTES);
        if estimated_target_state_file_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(PersistentPromptCacheDiskStoreError::SizeBoundExceeded {
                maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                estimated_block_bytes: estimated_target_state_file_bytes,
            });
        }

        let _write_operation_guard = self.lock_write_operations();
        self.prepare_active_model_storage_directories()?;
        let metadata_entries = target_state_metadata_entries(
            target_state_contract,
            target_state_identity_hash,
            prompt_prefix_token_ids.len(),
        );
        let metadata_entry_references = metadata_entries
            .iter()
            .map(|(metadata_name, metadata_text)| (*metadata_name, metadata_text.as_str()))
            .collect::<Vec<_>>();
        let mut named_target_state_arrays = Vec::with_capacity(decoder_state_tensors.len() + 1);
        named_target_state_arrays.push((
            SPECULATIVE_PREFILL_TARGET_SELECTED_POSITIONS_TENSOR_NAME,
            selected_target_token_positions,
        ));
        named_target_state_arrays.extend_from_slice(decoder_state_tensors);
        let serialized_target_state_bytes = runtime
            .serialize_safetensors(&named_target_state_arrays, &metadata_entry_references)
            .map_err(|source| PersistentPromptCacheDiskStoreError::SaveSafetensors { source })?;
        let target_state_file_path = save_serialized_safetensors_file(
            &self.speculative_prefill_target_states_directory,
            target_state_identity_hash,
            &serialized_target_state_bytes,
            &ImmediatePersistentPromptCacheSerializedFileWriter,
        )?;
        let target_state_file_size_bytes = read_file_size_bytes(&target_state_file_path)?;
        self.lock_tracked_files().insert_file(
            PersistentPromptCacheFileKind::SpeculativePrefillTargetState,
            target_state_identity_hash,
            TrackedPersistentPromptCacheFile {
                file_path: target_state_file_path.clone(),
                file_size_bytes: target_state_file_size_bytes,
            },
        );
        if let Err(global_quota_error) = self.enforce_global_prompt_cache_quota() {
            return Err(self.rollback_newly_saved_files_after_error(
                &[target_state_file_path],
                global_quota_error,
            ));
        }
        Ok(())
    }

    pub fn load_longest_speculative_prefill_target_state(
        &self,
        runtime: &MlxRuntime,
        target_state_contract: &PersistentSpeculativePrefillTargetStateContract,
        prompt_token_ids: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
        positional_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Result<Option<RestoredSpeculativePrefillTargetState>, PersistentPromptCacheDiskStoreError>
    {
        if target_state_contract.target_model_id() != self.model_contract_ref().model_id()
            || target_state_contract.target_model_revision()
                != self.model_contract_ref().model_revision()
        {
            return Err(invalid_target_state_input());
        }
        let target_state_identity_hash = {
            let tracked_files = self.lock_tracked_files();
            let Some(restored_prompt_prefix_token_count) =
                longest_reusable_speculative_prefill_target_prefix(
                    target_state_contract,
                    prompt_token_ids,
                    ordered_image_sha256_digests,
                    |candidate_target_state_identity| {
                        tracked_files
                            .file(
                                PersistentPromptCacheFileKind::SpeculativePrefillTargetState,
                                &candidate_target_state_identity,
                            )
                            .is_some()
                    },
                )
            else {
                return Ok(None);
            };
            target_state_contract.target_state_identity_hash(
                &prompt_token_ids[..restored_prompt_prefix_token_count],
                ordered_image_sha256_digests,
            )
        };
        let target_state_file_path = {
            let tracked_files = self.lock_tracked_files();
            let Some(tracked_target_state_file) = tracked_files.file(
                PersistentPromptCacheFileKind::SpeculativePrefillTargetState,
                &target_state_identity_hash,
            ) else {
                return Ok(None);
            };
            tracked_target_state_file.file_path.clone()
        };
        let target_state_file =
            open_without_following_symlinks(&target_state_file_path).map_err(|source| {
                PersistentPromptCacheDiskStoreError::OpenBlockFile {
                    block_file_path: target_state_file_path.clone(),
                    source,
                }
            })?;
        let target_state_file_header =
            PersistentSpeculativePrefillTargetStateFileHeader::read_model_bound_from_file(
                &target_state_file,
                &target_state_file_path,
                self.model_contract_ref(),
            )
            .map_err(|description| {
                PersistentPromptCacheDiskStoreError::ValidateModelSpecificArtifact {
                    artifact_file_path: target_state_file_path.clone(),
                    source: Box::new(std::io::Error::other(description)),
                }
            })?;
        let loaded_safetensors = runtime
            .load_safetensors(target_state_file, positional_file_read_metrics)
            .map_err(|source| PersistentPromptCacheDiskStoreError::LoadSafetensors { source })?;
        let selected_target_token_positions = loaded_safetensors
            .tensor(SPECULATIVE_PREFILL_TARGET_SELECTED_POSITIONS_TENSOR_NAME)
            .map_err(|source| PersistentPromptCacheDiskStoreError::LoadSafetensors { source })?;
        if selected_target_token_positions.dtype() != MlxDtype::UInt32
            || selected_target_token_positions.shape().as_slice() == [0]
            || selected_target_token_positions.shape().len() != 1
        {
            return Err(invalid_target_state_input());
        }
        let mut decoder_state_tensors = HashMap::with_capacity(
            target_state_file_header
                .tensor_names()
                .len()
                .saturating_sub(1),
        );
        for tensor_name in target_state_file_header.tensor_names() {
            if tensor_name == SPECULATIVE_PREFILL_TARGET_SELECTED_POSITIONS_TENSOR_NAME {
                continue;
            }
            let target_state_tensor = loaded_safetensors.tensor(tensor_name).map_err(|source| {
                PersistentPromptCacheDiskStoreError::LoadSafetensors { source }
            })?;
            decoder_state_tensors.insert(tensor_name.clone(), target_state_tensor);
        }
        Ok(Some(RestoredSpeculativePrefillTargetState::new(
            target_state_file_header.prompt_prefix_token_count(),
            selected_target_token_positions,
            decoder_state_tensors,
        )))
    }
}

fn invalid_target_state_input() -> PersistentPromptCacheDiskStoreError {
    PersistentPromptCacheDiskStoreError::SaveSafetensors {
        source: astronomical_runtime_integration::MlxRuntimeError::RuntimeOperation {
            operation: "validate speculative-prefill target state",
            description: "sparse target state does not match its target model contract".to_owned(),
        },
    }
}
