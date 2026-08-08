use std::sync::Arc;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, PositionalFileReadMetrics};

use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    ImmediatePersistentPromptCacheSerializedFileWriter, PersistentPromptCacheFileKind,
    open_without_following_symlinks, read_file_size_bytes,
    remove_cache_owned_file_or_confirm_absent, save_serialized_safetensors_file,
};
use super::disk_store_index::TrackedPersistentPromptCacheFile;
use super::speculative_prefill_selection::{
    PersistentSpeculativePrefillSelectionContract, PersistentSpeculativePrefillSelectionFileHeader,
    selection_file_metadata_entries, selection_prompt_metadata_entries,
};

const SPECULATIVE_PREFILL_SELECTION_TENSOR_NAME: &str = "selected_token_positions";

impl PersistentPromptCacheDiskStore {
    /// Saves one exact SpecPrefill selection under the drafter's model namespace.
    pub fn save_speculative_prefill_selection(
        &self,
        runtime: &MlxRuntime,
        selection_contract: &PersistentSpeculativePrefillSelectionContract,
        prompt_token_ids: &[u32],
        selected_token_positions: &MlxArray,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        validate_selection_input(
            selection_contract,
            prompt_token_ids,
            selected_token_positions,
            self.model_contract_ref().model_id(),
            self.model_contract_ref().model_revision(),
        )?;
        let selection_identity_hash = selection_contract.selection_identity_hash(prompt_token_ids);
        let estimated_selection_file_bytes = u64::try_from(selected_token_positions.byte_count())
            .unwrap_or(u64::MAX)
            .saturating_add(super::disk_store_write::ESTIMATED_SAFETENSORS_FILE_OVERHEAD_BYTES);
        if estimated_selection_file_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(PersistentPromptCacheDiskStoreError::SizeBoundExceeded {
                maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                estimated_block_bytes: estimated_selection_file_bytes,
            });
        }

        let _write_operation_guard = self.lock_write_operations();
        self.prepare_active_model_storage_directories()?;
        let mut metadata_entries =
            selection_file_metadata_entries(selection_contract, selection_identity_hash)
                .into_iter()
                .collect::<Vec<_>>();
        metadata_entries.extend(selection_prompt_metadata_entries(selection_contract));
        let metadata_entry_references = metadata_entries
            .iter()
            .map(|(metadata_name, metadata_value)| (*metadata_name, metadata_value.as_str()))
            .collect::<Vec<_>>();
        let named_arrays = [(
            SPECULATIVE_PREFILL_SELECTION_TENSOR_NAME,
            selected_token_positions,
        )];
        let serialized_selection_bytes = runtime
            .serialize_safetensors(&named_arrays, &metadata_entry_references)
            .map_err(|source| PersistentPromptCacheDiskStoreError::SaveSafetensors { source })?;
        let selection_file_path = save_serialized_safetensors_file(
            &self.speculative_prefill_selections_directory,
            selection_identity_hash,
            &serialized_selection_bytes,
            &ImmediatePersistentPromptCacheSerializedFileWriter,
        )?;
        let selection_file_size_bytes = match read_file_size_bytes(&selection_file_path) {
            Ok(selection_file_size_bytes) => selection_file_size_bytes,
            Err(metadata_error) => {
                return Err(self.rollback_newly_saved_files_after_error(
                    &[selection_file_path],
                    metadata_error,
                ));
            }
        };
        let newly_saved_selection_file_path = selection_file_path.clone();
        self.lock_tracked_files().insert_file(
            PersistentPromptCacheFileKind::SpeculativePrefillSelection,
            selection_identity_hash,
            TrackedPersistentPromptCacheFile {
                file_path: selection_file_path,
                file_size_bytes: selection_file_size_bytes,
            },
        );
        if let Err(global_quota_error) = self.enforce_global_prompt_cache_quota() {
            return Err(self.rollback_newly_saved_files_after_error(
                &[newly_saved_selection_file_path],
                global_quota_error,
            ));
        }
        Ok(())
    }

    /// Loads one exact SpecPrefill selection without touching MLX until its file is validated.
    pub fn load_speculative_prefill_selection(
        &self,
        runtime: &MlxRuntime,
        selection_contract: &PersistentSpeculativePrefillSelectionContract,
        prompt_token_ids: &[u32],
        positional_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Result<Option<MlxArray>, PersistentPromptCacheDiskStoreError> {
        validate_selection_prompt_identity(selection_contract, prompt_token_ids)?;
        let selection_identity_hash = selection_contract.selection_identity_hash(prompt_token_ids);
        let selection_file_path = {
            let tracked_files = self.lock_tracked_files();
            let Some(tracked_selection_file) = tracked_files.file(
                PersistentPromptCacheFileKind::SpeculativePrefillSelection,
                &selection_identity_hash,
            ) else {
                return Ok(None);
            };
            tracked_selection_file.file_path.clone()
        };
        let selection_file = match open_without_following_symlinks(&selection_file_path) {
            Ok(selection_file) => selection_file,
            Err(open_error) => {
                if open_error.kind() == std::io::ErrorKind::NotFound {
                    self.untrack_file_and_subtract_global_accounting(
                        PersistentPromptCacheFileKind::SpeculativePrefillSelection,
                        selection_identity_hash,
                    );
                }
                return Err(PersistentPromptCacheDiskStoreError::OpenBlockFile {
                    block_file_path: selection_file_path,
                    source: open_error,
                });
            }
        };
        let selection_file_header =
            match PersistentSpeculativePrefillSelectionFileHeader::read_for_contract_from_file(
                &selection_file,
                &selection_file_path,
                self.model_contract_ref(),
                selection_contract,
                selection_identity_hash,
            ) {
                Ok(selection_file_header) => selection_file_header,
                Err(validation_error) => {
                    remove_cache_owned_file_or_confirm_absent(&selection_file_path)?;
                    self.untrack_file_and_subtract_global_accounting(
                        PersistentPromptCacheFileKind::SpeculativePrefillSelection,
                        selection_identity_hash,
                    );
                    return Err(
                        PersistentPromptCacheDiskStoreError::ValidateModelSpecificArtifact {
                            artifact_file_path: selection_file_path,
                            source: Box::new(validation_error),
                        },
                    );
                }
            };
        let loaded_safetensors = runtime
            .load_safetensors(selection_file, positional_file_read_metrics)
            .map_err(|source| PersistentPromptCacheDiskStoreError::LoadSafetensors { source })?;
        let selected_token_positions = loaded_safetensors
            .tensor(SPECULATIVE_PREFILL_SELECTION_TENSOR_NAME)
            .map_err(|source| PersistentPromptCacheDiskStoreError::LoadSafetensors { source })?;
        if selected_token_positions.dtype() != MlxDtype::UInt32
            || selected_token_positions.shape()
                != [
                    i32::try_from(selection_file_header.selected_token_position_count()).map_err(
                        |_| PersistentPromptCacheDiskStoreError::ValidateModelSpecificArtifact {
                            artifact_file_path: selection_file_path.clone(),
                            source: Box::new(std::io::Error::other(
                                "persisted SpecPrefill selection length exceeds the MLX range",
                            )),
                        },
                    )?,
                ]
        {
            return Err(
                PersistentPromptCacheDiskStoreError::ValidateModelSpecificArtifact {
                    artifact_file_path: selection_file_path,
                    source: Box::new(std::io::Error::other(
                        "loaded SpecPrefill selection tensor does not match its validated header",
                    )),
                },
            );
        }
        Ok(Some(selected_token_positions))
    }
}

fn validate_selection_input(
    selection_contract: &PersistentSpeculativePrefillSelectionContract,
    prompt_token_ids: &[u32],
    selected_token_positions: &MlxArray,
    expected_draft_model_id: &str,
    expected_draft_model_revision: &str,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    validate_selection_prompt_identity(selection_contract, prompt_token_ids)?;
    if selection_contract.draft_model_id() != expected_draft_model_id
        || selection_contract.draft_model_revision() != expected_draft_model_revision
        || selected_token_positions.dtype() != MlxDtype::UInt32
        || selected_token_positions.shape().len() != 1
        || selected_token_positions.shape()[0] <= 0
    {
        return Err(PersistentPromptCacheDiskStoreError::SaveSafetensors {
            source: astronomical_runtime_integration::MlxRuntimeError::RuntimeOperation {
                operation: "validate speculative-prefill selection",
                description:
                    "selection metadata or tensor shape does not match the drafter cache contract"
                        .to_owned(),
            },
        });
    }
    Ok(())
}

fn validate_selection_prompt_identity(
    selection_contract: &PersistentSpeculativePrefillSelectionContract,
    prompt_token_ids: &[u32],
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    if usize::try_from(selection_contract.prompt_token_count()).ok() != Some(prompt_token_ids.len())
    {
        return Err(PersistentPromptCacheDiskStoreError::SaveSafetensors {
            source: astronomical_runtime_integration::MlxRuntimeError::RuntimeOperation {
                operation: "validate speculative-prefill selection prompt",
                description: "selection prompt token count does not match the requested prompt"
                    .to_owned(),
            },
        });
    }
    Ok(())
}
