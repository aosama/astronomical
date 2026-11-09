//! Removes a model that Download Models installed, by moving its folder to the
//! Trash.
//!
//! Removal is a coordinator command because it shares the download slot's
//! state: it takes the same command lock as start, pause and cancel, so it
//! cannot race a download of the same model, and it forgets the model's
//! validated publication so the catalog stops reporting it as on this Mac.
//!
//! Only the coordinator's own published folder for the catalog identity is
//! ever touched, and only when its provenance record names that model. A model
//! the person placed anywhere else is never removed from here.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use super::LibraryDownloadCoordinator;
use crate::library::DownloadJobState;
use crate::library::download::job::store::filesystem::{
    ensure_descendant_directory_or_absent, metadata_without_symlink, remove_directory_if_empty,
};
use crate::library::download::publication::provenance::publication_provenance_names_model;
use crate::library::model_folder_busy::{ModelFolderBusy, ModelFolderBusyCheck};
use crate::library::model_folder_trash::{ModelFolderTrash, ModelFolderTrashError};

/// Why a model could not be removed from disk.
#[derive(Debug, Error)]
pub enum InstalledModelRemovalError {
    #[error("the requested model is not in the release catalog")]
    CatalogEntryNotFound,
    #[error("the model has no folder in the Download Models folder on this Mac")]
    NotOnThisMac,
    #[error("the model's folder was not installed by Download Models")]
    NotInstalledByDownloadModels,
    #[error("the model's folder is reached through a link or lies outside the models folder")]
    UnsafeFolder,
    #[error("a download of this model is still in progress")]
    DownloadInProgress,
    #[error("the worker is using a model from this folder: {0:?}")]
    ModelFolderBusy(ModelFolderBusy),
    #[error("the download state could not be read: {0}")]
    DownloadState(String),
    #[error(transparent)]
    Trash(#[from] ModelFolderTrashError),
}

impl LibraryDownloadCoordinator {
    /// Whether this model's Download Models folder exists and carries its
    /// provenance record, so the catalog can offer Remove from disk.
    #[must_use]
    pub fn is_removable_installation(&self, huggingface_id: &str) -> bool {
        self.installed_model_folder(huggingface_id).is_ok()
    }

    fn installed_model_folder(
        &self,
        huggingface_id: &str,
    ) -> Result<PathBuf, InstalledModelRemovalError> {
        installed_model_folder_in(self.job_store.models_directory(), huggingface_id)
    }

    /// Moves the model's Download Models folder to the Trash, then forgets its
    /// publication and refreshes discovery. A failed refresh is logged rather
    /// than reported: the folder is already gone, and the next discovery
    /// (a reload or a restart) catches up.
    pub async fn remove_installed_model(
        &self,
        huggingface_id: &str,
        model_folder_busy_check: Arc<dyn ModelFolderBusyCheck>,
        model_folder_trash: Arc<dyn ModelFolderTrash>,
    ) -> Result<(), InstalledModelRemovalError> {
        let _command_guard = self.command_lock.lock().await;
        if !self
            .download_catalog
            .entries()
            .iter()
            .any(|catalog_entry| catalog_entry.huggingface_id() == huggingface_id)
        {
            return Err(InstalledModelRemovalError::CatalogEntryNotFound);
        }
        self.refuse_while_this_model_downloads(huggingface_id)
            .await?;
        let model_folder = self.installed_model_folder(huggingface_id)?;
        // Asked last, under the command lock and right before the move, so a
        // load cannot start between the check and the move's own decision.
        if let Some(folder_busy) =
            model_folder_busy_check.busy_reason(&model_folder, huggingface_id)
        {
            return Err(InstalledModelRemovalError::ModelFolderBusy(folder_busy));
        }
        let publisher_folder = model_folder.parent().map(Path::to_path_buf);
        let trash_outcome = tokio::task::spawn_blocking(move || {
            model_folder_trash.move_to_trash(&model_folder)
        })
        .await
        .map_err(|join_error| {
            // The Trash task panicked or was cancelled; whether the
            // folder moved is unknown, so the publication is kept and
            // the page reports that the move failed.
            tracing::warn!(huggingface_id, %join_error, "the move to the Trash did not finish");
            InstalledModelRemovalError::Trash(ModelFolderTrashError::Failed)
        })?;
        trash_outcome?;
        self.remove_empty_publisher_folder(publisher_folder.as_deref());
        self.validated_publications
            .lock()
            .await
            .remove(huggingface_id);
        self.refresh_discovery_after_removal(huggingface_id).await;
        Ok(())
    }

    /// A job of this model that has not failed still owns its folder, so the
    /// folder stays until the download is cancelled or finishes.
    async fn refuse_while_this_model_downloads(
        &self,
        huggingface_id: &str,
    ) -> Result<(), InstalledModelRemovalError> {
        let current_job = self.current_job().await.map_err(|job_error| {
            InstalledModelRemovalError::DownloadState(job_error.to_string())
        })?;
        match current_job {
            Some(download_job)
                if download_job.huggingface_id() == huggingface_id
                    && download_job.state() != DownloadJobState::Failed =>
            {
                Err(InstalledModelRemovalError::DownloadInProgress)
            }
            _ => Ok(()),
        }
    }

    /// A model's folder sits under its publisher's folder (`<models>/<org>/<name>`);
    /// once the last model of that publisher moves to the Trash the empty
    /// publisher folder is removed too, so Finder shows no leftover. A folder
    /// holding anything else, even Finder's own `.DS_Store`, stays.
    fn remove_empty_publisher_folder(&self, publisher_folder: Option<&Path>) {
        let Some(publisher_folder) = publisher_folder else {
            return;
        };
        if publisher_folder.parent() != Some(self.job_store.models_directory()) {
            return;
        }
        if let Err(removal_error) = remove_directory_if_empty(publisher_folder) {
            tracing::warn!(%removal_error, "the empty publisher folder could not be removed");
        }
    }

    async fn refresh_discovery_after_removal(&self, huggingface_id: &str) {
        let discovery_refresh = Arc::clone(&self.discovery_refresh);
        let refresh_outcome =
            tokio::task::spawn_blocking(move || discovery_refresh.refresh_after_removal()).await;
        match refresh_outcome {
            Ok(Ok(())) => {}
            Ok(Err(refresh_error)) => tracing::warn!(
                huggingface_id,
                error = %refresh_error,
                "model removed from disk, but discovery did not refresh"
            ),
            Err(join_error) => tracing::warn!(
                huggingface_id,
                %join_error,
                "model removed from disk, but the discovery refresh task failed"
            ),
        }
    }
}

/// The model's own folder under `models_directory`, but only when every
/// component from the models folder down is a real directory, the resolved
/// folder still sits inside the resolved models folder, and its provenance
/// record names the model. Checking only the last component would let a
/// symlinked publisher folder send a removal anywhere on the disk.
pub(super) fn installed_model_folder_in(
    models_directory: &Path,
    huggingface_id: &str,
) -> Result<PathBuf, InstalledModelRemovalError> {
    let model_folder = models_directory.join(huggingface_id);
    ensure_descendant_directory_or_absent(models_directory, &model_folder)
        .map_err(|_| InstalledModelRemovalError::UnsafeFolder)?;
    match metadata_without_symlink(&model_folder) {
        Ok(Some(folder_metadata)) if folder_metadata.is_dir() => {}
        Ok(None) => return Err(InstalledModelRemovalError::NotOnThisMac),
        _ => return Err(InstalledModelRemovalError::UnsafeFolder),
    }
    if !resolves_inside(models_directory, &model_folder) {
        return Err(InstalledModelRemovalError::UnsafeFolder);
    }
    if !publication_provenance_names_model(&model_folder, huggingface_id) {
        return Err(InstalledModelRemovalError::NotInstalledByDownloadModels);
    }
    Ok(model_folder)
}

/// Whether `model_folder`, with every link resolved, is still inside
/// `models_directory` with its links resolved. The component walk above
/// refuses links below the models folder; this also covers a models folder
/// that is itself reached through a link, by comparing resolved paths.
fn resolves_inside(models_directory: &Path, model_folder: &Path) -> bool {
    match (models_directory.canonicalize(), model_folder.canonicalize()) {
        (Ok(resolved_models_directory), Ok(resolved_model_folder)) => {
            resolved_model_folder.starts_with(&resolved_models_directory)
                && resolved_model_folder != resolved_models_directory
        }
        _ => false,
    }
}

#[cfg(test)]
#[path = "installed_model_removal_tests.rs"]
mod installed_model_removal_tests;
