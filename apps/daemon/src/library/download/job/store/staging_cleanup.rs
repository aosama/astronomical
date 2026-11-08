//! Startup tidy-up of the hidden staging tree.
//!
//! Versions before the publication tidy-up (#1929) left an empty `.incomplete/<organization>`
//! folder behind every published model. They are harmless, but they are clutter people find when
//! they look inside their models folder, so startup removes them. Only folders that are empty, or
//! hold nothing but Finder's `.DS_Store`, are removed: a paused or failed download keeps its
//! staged bytes, so it can still be resumed.

use std::path::Path;

use super::{
    DownloadJobStore, DownloadJobStoreError,
    filesystem::{
        metadata_without_symlink, read_directory_entries, remove_directory_if_only_finder_metadata,
        synchronize_directory,
    },
    lock,
};
use crate::library::download::job::download_job::STAGING_DIRECTORY_NAME;

impl DownloadJobStore {
    /// Removes organization folders under the staging folder that are empty or hold only
    /// Finder's `.DS_Store`, then the staging folder itself when nothing is left in it.
    ///
    /// It takes the cross-process transaction lock as every other store transaction does, so a
    /// second instance sharing this models folder cannot remove a folder another daemon's store
    /// transaction is about to fill. Payload transfer writes staged files without that lock;
    /// this pass never removes a folder that already holds one, so it cannot lose bytes.
    ///
    /// A folder that cannot be read or removed is logged and skipped, so one bad leftover does
    /// not keep every other one in place.
    pub fn remove_empty_staging_folders(&self) -> Result<(), DownloadJobStoreError> {
        let _operation_guard = self.lock_operations()?;
        let Some(_transaction_lock) =
            lock::acquire_existing_store_transaction_lock(&self.models_directory)?
        else {
            return Ok(());
        };
        let staging_root = self.models_directory.join(STAGING_DIRECTORY_NAME);
        let Some(staging_root_metadata) = metadata_without_symlink(&staging_root)? else {
            return Ok(());
        };
        if !staging_root_metadata.is_dir() {
            return Ok(());
        }
        if remove_empty_organization_folders(&staging_root)? {
            synchronize_directory(&staging_root)?;
        }
        if remove_directory_if_only_finder_metadata(&staging_root)? {
            synchronize_directory(&self.models_directory)?;
        }
        Ok(())
    }
}

/// Removes each empty organization folder, skipping symlinks and any folder it cannot handle.
/// Returns whether any folder was removed.
fn remove_empty_organization_folders(staging_root: &Path) -> Result<bool, DownloadJobStoreError> {
    let mut removed_any_folder = false;
    for staging_entry in read_directory_entries(staging_root)? {
        // The entry's own type does not follow links, so a symlink is skipped rather than
        // refused: it is not ours to remove, and neither is what it points to.
        let Ok(entry_type) = staging_entry.file_type() else {
            continue;
        };
        if !entry_type.is_dir() {
            continue;
        }
        match remove_directory_if_only_finder_metadata(&staging_entry.path()) {
            Ok(was_removed) => removed_any_folder |= was_removed,
            Err(cleanup_error) => {
                tracing::warn!(%cleanup_error, "skipped a staging folder that could not be tidied");
            }
        }
    }
    Ok(removed_any_folder)
}

#[cfg(test)]
#[path = "staging_cleanup_tests.rs"]
mod staging_cleanup_tests;
