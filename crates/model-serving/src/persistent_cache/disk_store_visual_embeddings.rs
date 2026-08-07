use std::fs::{self, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::disk_store::PersistentPromptCacheDiskStore;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, hex_encode, open_without_following_symlinks,
    read_file_size_bytes, remove_cache_owned_file_or_confirm_absent,
};
use super::disk_store_index::TrackedPersistentPromptCacheFile;
use super::disk_store_scan::scan_current_format_directory;
use super::{
    PersistentVisualEmbeddingFileHeader, PersistentVisualEmbeddingKey,
    PersistentVisualEmbeddingModelContract,
};

impl PersistentPromptCacheDiskStore {
    pub fn save_visual_embedding(
        &self,
        runtime: &MlxRuntime,
        visual_embedding_key: &PersistentVisualEmbeddingKey,
        visual_embeddings: &MlxArray,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        let _write_operation_guard = self.lock_write_operations();
        self.prepare_active_model_storage_directories()?;
        let visual_embedding_hash = visual_embedding_key.visual_embedding_hash();
        let estimated_visual_embedding_bytes = u64::try_from(visual_embeddings.byte_count())
            .unwrap_or(u64::MAX)
            .saturating_add(16 * 1024);
        if estimated_visual_embedding_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(PersistentPromptCacheDiskStoreError::SizeBoundExceeded {
                maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                estimated_block_bytes: estimated_visual_embedding_bytes,
            });
        }
        let visual_embedding_file_path = save_visual_embedding_safetensors_file(
            runtime,
            &self.visual_embeddings_directory,
            visual_embedding_key,
            visual_embeddings,
            self.model_contract.model_id(),
            self.model_contract.model_revision(),
        )?;
        let visual_embedding_file_size_bytes =
            match read_file_size_bytes(&visual_embedding_file_path) {
                Ok(size_bytes) => size_bytes,
                Err(metadata_error) => {
                    return Err(self.rollback_newly_saved_files_after_error(
                        &[visual_embedding_file_path],
                        metadata_error,
                    ));
                }
            };
        if visual_embedding_file_size_bytes > self.global_prompt_cache_maximum_size_bytes {
            return Err(self.rollback_newly_saved_files_after_error(
                &[visual_embedding_file_path],
                PersistentPromptCacheDiskStoreError::SizeBoundExceeded {
                    maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
                    estimated_block_bytes: visual_embedding_file_size_bytes,
                },
            ));
        }
        let newly_saved_visual_embedding_file_path = visual_embedding_file_path.clone();
        let mut tracked_files = self.lock_tracked_files();
        tracked_files.insert_file(
            PersistentPromptCacheFileKind::VisualEmbedding,
            visual_embedding_hash,
            TrackedPersistentPromptCacheFile {
                file_path: visual_embedding_file_path,
                file_size_bytes: visual_embedding_file_size_bytes,
            },
        );
        drop(tracked_files);
        if let Err(global_quota_error) = self.enforce_global_prompt_cache_quota() {
            return Err(self.rollback_newly_saved_files_after_error(
                &[newly_saved_visual_embedding_file_path],
                global_quota_error,
            ));
        }
        Ok(())
    }

    pub fn load_visual_embedding(
        &self,
        runtime: &MlxRuntime,
        visual_embedding_key: &PersistentVisualEmbeddingKey,
        persistent_visual_embedding_model_contract: &PersistentVisualEmbeddingModelContract,
    ) -> Result<Option<MlxArray>, PersistentPromptCacheDiskStoreError> {
        let visual_embedding_hash = visual_embedding_key.visual_embedding_hash();
        let visual_embedding_file_path = {
            let tracked_files = self.lock_tracked_files();
            let Some(tracked_file) = tracked_files.file(
                PersistentPromptCacheFileKind::VisualEmbedding,
                &visual_embedding_hash,
            ) else {
                return Ok(None);
            };
            tracked_file.file_path.clone()
        };
        let visual_embedding_file =
            match open_without_following_symlinks(&visual_embedding_file_path) {
                Ok(file) => file,
                Err(open_error) => {
                    if open_error.kind() == std::io::ErrorKind::NotFound {
                        self.untrack_file_and_subtract_global_accounting(
                            PersistentPromptCacheFileKind::VisualEmbedding,
                            visual_embedding_hash,
                        );
                    }
                    return Err(PersistentPromptCacheDiskStoreError::OpenBlockFile {
                        block_file_path: visual_embedding_file_path,
                        source: open_error,
                    });
                }
            };
        if let Err(validation_error) = PersistentVisualEmbeddingFileHeader::read_from_file(
            &visual_embedding_file,
            &visual_embedding_file_path,
            persistent_visual_embedding_model_contract,
        ) {
            // Corrupt visual embedding: delete first (NotFound counts as
            // already absent), then untrack only after successful deletion.
            remove_cache_owned_file_or_confirm_absent(&visual_embedding_file_path)?;
            self.untrack_file_and_subtract_global_accounting(
                PersistentPromptCacheFileKind::VisualEmbedding,
                visual_embedding_hash,
            );
            return Err(
                PersistentPromptCacheDiskStoreError::ValidateModelSpecificArtifact {
                    artifact_file_path: visual_embedding_file_path,
                    source: Box::new(validation_error),
                },
            );
        }
        let loaded_safetensors = runtime
            .load_safetensors(visual_embedding_file, None)
            .map_err(|source| PersistentPromptCacheDiskStoreError::LoadSafetensors { source })?;
        let visual_embedding_tensor = loaded_safetensors
            .tensor("visual_embeddings")
            .map_err(|source| PersistentPromptCacheDiskStoreError::LoadSafetensors { source })?;
        Ok(Some(visual_embedding_tensor))
    }

    pub(crate) fn scan_visual_embeddings(
        &self,
        persistent_visual_embedding_model_contract: &PersistentVisualEmbeddingModelContract,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        {
            let mut tracked_files = self.lock_tracked_files();
            scan_current_format_directory(
                &self.visual_embeddings_directory,
                PersistentPromptCacheFileKind::VisualEmbedding,
                &mut tracked_files,
                |visual_embedding_file, visual_embedding_file_path| {
                    PersistentVisualEmbeddingFileHeader::read_from_file(
                        visual_embedding_file,
                        visual_embedding_file_path,
                        persistent_visual_embedding_model_contract,
                    )
                    .is_ok()
                },
            )?;
        }
        self.enforce_global_prompt_cache_quota()
    }
}

fn save_visual_embedding_safetensors_file(
    runtime: &MlxRuntime,
    visual_embeddings_directory: &Path,
    visual_embedding_key: &PersistentVisualEmbeddingKey,
    visual_embeddings: &MlxArray,
    model_id: &str,
    model_revision: &str,
) -> Result<PathBuf, PersistentPromptCacheDiskStoreError> {
    let visual_embedding_hash = visual_embedding_key.visual_embedding_hash();
    let encoded_image_sha256 = visual_embedding_key.encoded_image_sha256();
    let visual_embeddings_shape = visual_embeddings.shape();
    let visual_token_count = usize::try_from(visual_embeddings_shape[0]).map_err(|_| {
        PersistentPromptCacheDiskStoreError::SaveSafetensors {
            source: astronomical_runtime_integration::MlxRuntimeError::RuntimeOperation {
                operation: "save visual embedding",
                description: "visual embedding row count overflowed usize".to_owned(),
            },
        }
    })?;
    let file_name = format!("{}.safetensors", hex_encode(visual_embedding_hash));
    let file_path = visual_embeddings_directory.join(&file_name);
    let temp_file_path = visual_embeddings_directory.join(format!("{file_name}.tmp"));
    remove_cache_owned_file_or_confirm_absent(&temp_file_path)?;
    let named_array = ("visual_embeddings", visual_embeddings);
    let encoded_image_sha256_text = hex_encode(encoded_image_sha256);
    let visual_token_count_text = visual_token_count.to_string();
    let metadata_entries: [(&str, &str); 5] = [
        (
            "format_version",
            super::PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION,
        ),
        ("model_id", model_id),
        ("model_revision", model_revision),
        ("encoded_image_sha256", encoded_image_sha256_text.as_str()),
        ("visual_token_count", visual_token_count_text.as_str()),
    ];
    let temporary_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&temp_file_path)
        .map_err(|source| PersistentPromptCacheDiskStoreError::OpenTempFile {
            temp_file_path: temp_file_path.clone(),
            source,
        })?;
    if let Err(save_error) =
        runtime.save_safetensors(temporary_file, &[named_array], &metadata_entries)
    {
        remove_cache_owned_file_or_confirm_absent(&temp_file_path)?;
        return Err(PersistentPromptCacheDiskStoreError::SaveSafetensors { source: save_error });
    }
    fs::rename(&temp_file_path, &file_path).map_err(|rename_error| {
        let temporary_file_cleanup_result =
            remove_cache_owned_file_or_confirm_absent(&temp_file_path);
        if let Err(temporary_file_cleanup_error) = temporary_file_cleanup_result {
            return temporary_file_cleanup_error;
        }
        PersistentPromptCacheDiskStoreError::RenameTempFile {
            temp_file_path,
            block_file_path: file_path.clone(),
            source: rename_error,
        }
    })?;
    Ok(file_path)
}
