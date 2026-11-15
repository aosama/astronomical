use std::fs::{self, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use astronomical_runtime_integration::MlxRuntime;

use super::disk_store::AstroSsdPromptCacheDiskStore;
use super::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use super::disk_store_file::{
    AstroSsdPromptCacheFileKind, map_safetensors_writer_error, open_without_following_symlinks,
    read_file_size_bytes, remove_astro_ssd_prompt_cache_owned_file_or_confirm_absent,
};
use super::disk_store_index::TrackedAstroSsdPromptCacheFile;
use super::disk_store_scan;
use super::startup_cleanup_evidence::AstroSsdPromptCacheStartupCleanupEvidence;
use super::{
    PersistentVisualEmbeddingFileHeader, PersistentVisualEmbeddingKey,
    PersistentVisualEmbeddingModelContract,
};
use mlx_bridge::MlxArray;

impl AstroSsdPromptCacheDiskStore {
    pub fn save_visual_embedding(
        &self,
        runtime: &MlxRuntime,
        visual_embedding_key: &PersistentVisualEmbeddingKey,
        visual_embeddings: &MlxArray,
    ) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
        let _write_operation_guard = self.lock_write_operations();
        self.prepare_active_model_storage_directories()?;
        let visual_embedding_hash = visual_embedding_key.visual_embedding_hash();
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
        if visual_embedding_file_size_bytes > self.global_astro_ssd_prompt_cache_maximum_size_bytes
        {
            return Err(self.rollback_newly_saved_files_after_error(
                &[visual_embedding_file_path],
                AstroSsdPromptCacheDiskStoreError::SizeBoundExceeded {
                    maximum_size_bytes: self.global_astro_ssd_prompt_cache_maximum_size_bytes,
                    estimated_block_bytes: visual_embedding_file_size_bytes,
                },
            ));
        }
        let newly_saved_visual_embedding_file_path = visual_embedding_file_path.clone();
        let mut tracked_files = self.lock_tracked_files();
        tracked_files.insert_file(
            AstroSsdPromptCacheFileKind::VisualEmbedding,
            visual_embedding_hash,
            TrackedAstroSsdPromptCacheFile {
                file_path: visual_embedding_file_path,
                file_size_bytes: visual_embedding_file_size_bytes,
            },
        );
        drop(tracked_files);
        if let Err(global_quota_error) = self.enforce_global_astro_ssd_prompt_cache_quota() {
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
    ) -> Result<Option<MlxArray>, AstroSsdPromptCacheDiskStoreError> {
        let visual_embedding_hash = visual_embedding_key.visual_embedding_hash();
        let visual_embedding_file_path = {
            let tracked_files = self.lock_tracked_files();
            let Some(tracked_file) = tracked_files.file(
                AstroSsdPromptCacheFileKind::VisualEmbedding,
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
                            AstroSsdPromptCacheFileKind::VisualEmbedding,
                            visual_embedding_hash,
                        );
                    }
                    return Err(AstroSsdPromptCacheDiskStoreError::OpenBlockFile {
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
            remove_astro_ssd_prompt_cache_owned_file_or_confirm_absent(
                &visual_embedding_file_path,
            )?;
            self.untrack_file_and_subtract_global_accounting(
                AstroSsdPromptCacheFileKind::VisualEmbedding,
                visual_embedding_hash,
            );
            return Err(
                AstroSsdPromptCacheDiskStoreError::ValidateModelSpecificArtifact {
                    artifact_file_path: visual_embedding_file_path,
                    source: Box::new(validation_error),
                },
            );
        }
        let loaded_safetensors = runtime
            .load_safetensors(visual_embedding_file, None)
            .map_err(|source| AstroSsdPromptCacheDiskStoreError::LoadSafetensors { source })?;
        let visual_embedding_tensor = loaded_safetensors
            .tensor("visual_embeddings")
            .map_err(|source| AstroSsdPromptCacheDiskStoreError::LoadSafetensors { source })?;
        Ok(Some(visual_embedding_tensor))
    }

    pub(crate) fn scan_visual_embeddings(
        &self,
        persistent_visual_embedding_model_contract: &PersistentVisualEmbeddingModelContract,
    ) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
        let mut startup_cleanup_evidence = AstroSsdPromptCacheStartupCleanupEvidence::default();
        {
            let mut tracked_files = self.lock_tracked_files();
            disk_store_scan::scan_current_format_directory(
                &self.visual_embeddings_directory,
                AstroSsdPromptCacheFileKind::VisualEmbedding,
                &mut tracked_files,
                &mut startup_cleanup_evidence,
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
        self.record_startup_cleanup_evidence(startup_cleanup_evidence);
        self.enforce_startup_global_astro_ssd_prompt_cache_quota(&[])
    }
}

fn save_visual_embedding_safetensors_file(
    runtime: &MlxRuntime,
    visual_embeddings_directory: &Path,
    visual_embedding_key: &PersistentVisualEmbeddingKey,
    visual_embeddings: &MlxArray,
    model_id: &str,
    model_revision: &str,
) -> Result<PathBuf, AstroSsdPromptCacheDiskStoreError> {
    let visual_embedding_hash = visual_embedding_key.visual_embedding_hash();
    let encoded_image_sha256 = visual_embedding_key.encoded_image_sha256();
    let visual_embeddings_shape = visual_embeddings.shape();
    let visual_token_count = usize::try_from(visual_embeddings_shape[0]).map_err(|_| {
        AstroSsdPromptCacheDiskStoreError::SaveSafetensors {
            source: astronomical_runtime_integration::MlxRuntimeError::RuntimeOperation {
                operation: "save visual embedding",
                description: "visual embedding row count overflowed usize".to_owned(),
            },
        }
    })?;
    let file_name = format!(
        "{}.safetensors",
        astronomical_lowercase_hex::encode_lowercase_hex(&visual_embedding_hash)
    );
    let file_path = visual_embeddings_directory.join(&file_name);
    let temp_file_path = visual_embeddings_directory.join(format!("{file_name}.tmp"));
    remove_astro_ssd_prompt_cache_owned_file_or_confirm_absent(&temp_file_path)?;
    let named_array = ("visual_embeddings", visual_embeddings);
    let encoded_image_sha256_text =
        astronomical_lowercase_hex::encode_lowercase_hex(&encoded_image_sha256);
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
        .map_err(|source| AstroSsdPromptCacheDiskStoreError::OpenTempFile {
            temp_file_path: temp_file_path.clone(),
            source,
        })?;
    if let Err(save_error) =
        runtime.save_safetensors(temporary_file, &[named_array], &metadata_entries)
    {
        remove_astro_ssd_prompt_cache_owned_file_or_confirm_absent(&temp_file_path)?;
        return Err(map_safetensors_writer_error(save_error));
    }
    fs::rename(&temp_file_path, &file_path).map_err(|rename_error| {
        let temporary_file_cleanup_result =
            remove_astro_ssd_prompt_cache_owned_file_or_confirm_absent(&temp_file_path);
        if let Err(temporary_file_cleanup_error) = temporary_file_cleanup_result {
            return temporary_file_cleanup_error;
        }
        AstroSsdPromptCacheDiskStoreError::RenameTempFile {
            temp_file_path,
            block_file_path: file_path.clone(),
            source: rename_error,
        }
    })?;
    Ok(file_path)
}
