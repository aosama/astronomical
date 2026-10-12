//! Durable immutable-provider provenance written into verified publication trees.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

use super::{
    DownloadJobStoreError,
    download_job_store_filesystem::{metadata_without_symlink, synchronize_directory},
    download_staged_file::create_descendant_directories,
};

const CONFIG_REVISION_METADATA_PATH: &str = ".cache/huggingface/download/config.json.metadata";
const LIBRARY_PROVENANCE_FILE_NAME: &str = ".astronomical-library-provenance.json";
static NEXT_PROVENANCE_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(1);

pub(super) fn write_publication_provenance(
    model_directory: &Path,
    provider_model_id: &str,
    revision: &str,
) -> Result<(), DownloadJobStoreError> {
    write_atomic_provenance_file(
        model_directory,
        CONFIG_REVISION_METADATA_PATH,
        format!("{revision}\n").as_bytes(),
    )?;
    let provenance_bytes = serde_json::to_vec(&serde_json::json!({
        "schema_version": 1,
        "provider_model_id": provider_model_id,
        "revision": revision,
    }))
    .map_err(DownloadJobStoreError::SerializePublicationProvenance)?;
    write_atomic_provenance_file(
        model_directory,
        LIBRARY_PROVENANCE_FILE_NAME,
        &provenance_bytes,
    )
}

fn write_atomic_provenance_file(
    model_directory: &Path,
    relative_path: &str,
    provenance_bytes: &[u8],
) -> Result<(), DownloadJobStoreError> {
    let provenance_file_path = model_directory.join(relative_path);
    let provenance_directory = provenance_file_path.parent().ok_or_else(|| {
        DownloadJobStoreError::UnsafeFilesystemObject {
            path: provenance_file_path.clone(),
        }
    })?;
    create_descendant_directories(model_directory, provenance_directory)?;
    if let Some(existing_metadata) = metadata_without_symlink(&provenance_file_path)?
        && !existing_metadata.is_file()
    {
        return Err(DownloadJobStoreError::UnsafeFilesystemObject {
            path: provenance_file_path,
        });
    }

    let sequence_number = NEXT_PROVENANCE_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary_file_path = provenance_directory.join(format!(
        ".astronomical-provenance.tmp.{}.{}",
        std::process::id(),
        sequence_number
    ));
    let mut temporary_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&temporary_file_path)
        .map_err(|source| DownloadJobStoreError::WritePublicationProvenance {
            path: temporary_file_path.clone(),
            source,
        })?;
    if let Err(source) = temporary_file
        .write_all(provenance_bytes)
        .and_then(|()| temporary_file.sync_all())
    {
        let _cleanup_outcome = fs::remove_file(&temporary_file_path);
        return Err(DownloadJobStoreError::WritePublicationProvenance {
            path: temporary_file_path,
            source,
        });
    }
    if let Err(source) = fs::rename(&temporary_file_path, &provenance_file_path) {
        let _cleanup_outcome = fs::remove_file(&temporary_file_path);
        return Err(DownloadJobStoreError::WritePublicationProvenance {
            path: provenance_file_path,
            source,
        });
    }
    synchronize_directory(provenance_directory)
}
