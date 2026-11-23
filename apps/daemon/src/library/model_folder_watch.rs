//! Notices model folders a person adds, moves or removes by hand, so the new
//! model is listed without restarting the app.
//!
//! The daemon discovers models once at start and again after its own
//! downloads and removals; a folder copied into the library was invisible
//! until the next start, and a moved one failed its load with a misleading
//! "could not be classified". Every few seconds this reads the names of the
//! library's folders and the names, sizes and modification times of their
//! files, which costs `stat` calls and reads no file. It rescans only once a
//! change has held still for a whole interval: a model being copied in grows
//! file by file for minutes, and listing it half-copied would fail its load.
//! Folder modification times are deliberately not read: Download Models
//! saves its progress file in the library root every few seconds, which
//! would make every download look like a library change.
//! A file-system event stream was not used: it needs a native dependency
//! and still has to be backed by a poll for network volumes and symlinked
//! folders, which are common in model libraries.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{Duration, SystemTime},
};

use tokio::sync::watch;

use crate::ResolvedRuntimeConfig;
use crate::library::model_discovery_refresh::LibraryModelDiscoveryRefresh;

/// How often the library's folders are compared. A model copied in is listed
/// at most two intervals after its last file lands: one to see the change,
/// one to see it hold still.
pub const MODEL_FOLDER_WATCH_INTERVAL: Duration = Duration::from_secs(5);

/// Folder levels read below each library root: `organization/model`, plus
/// the Hugging Face layout's `models--org--name/snapshots/revision`, plus one
/// more so a `drafter` folder appearing in a model is noticed.
const WATCHED_FOLDER_DEPTH: usize = 4;

/// One watched entry: a folder by its path alone, a file also by its size
/// and modification time.
type WatchedEntry = (PathBuf, Option<(u64, Option<SystemTime>)>);

/// Every watched folder and file of the library.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ModelFolderFingerprint {
    watched_entries: Vec<WatchedEntry>,
}

/// Reads the fingerprint of every root. Symlinked folders are followed,
/// because Development's library is mostly symlinks into Stable's.
#[must_use]
pub fn model_folder_fingerprint(library_roots: &[PathBuf]) -> ModelFolderFingerprint {
    let mut watched_entries = Vec::new();
    for library_root in library_roots {
        record_folder(library_root, WATCHED_FOLDER_DEPTH, &mut watched_entries);
    }
    watched_entries.sort();
    ModelFolderFingerprint { watched_entries }
}

fn record_folder(folder: &Path, remaining_depth: usize, watched_entries: &mut Vec<WatchedEntry>) {
    if !folder.is_dir() {
        return;
    }
    watched_entries.push((folder.to_path_buf(), None));
    let Ok(folder_entries) = std::fs::read_dir(folder) else {
        return;
    };
    for folder_entry in folder_entries.flatten() {
        if folder_entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let entry_path = folder_entry.path();
        // `metadata` follows symlinks, so a symlinked model folder or file is
        // read as what it points at.
        let Ok(entry_metadata) = std::fs::metadata(&entry_path) else {
            continue;
        };
        if entry_metadata.is_dir() {
            if remaining_depth > 0 {
                record_folder(&entry_path, remaining_depth - 1, watched_entries);
            }
        } else {
            let file_state = (entry_metadata.len(), entry_metadata.modified().ok());
            watched_entries.push((entry_path, Some(file_state)));
        }
    }
}

/// What the watch does on one tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFolderWatchStep {
    /// Nothing changed since the last rescan.
    Idle,
    /// Something changed; wait a tick to see whether it has finished.
    WaitForSettling,
    /// The change held still for a whole tick; rediscover now.
    Rescan,
}

/// Decides one tick from the fingerprint last acted on, the one seen on the
/// previous tick, and the one seen now.
#[must_use]
pub fn model_folder_watch_step(
    acted_fingerprint: &ModelFolderFingerprint,
    previous_tick_fingerprint: &ModelFolderFingerprint,
    current_fingerprint: &ModelFolderFingerprint,
) -> ModelFolderWatchStep {
    if current_fingerprint == acted_fingerprint {
        ModelFolderWatchStep::Idle
    } else if current_fingerprint == previous_tick_fingerprint {
        ModelFolderWatchStep::Rescan
    } else {
        ModelFolderWatchStep::WaitForSettling
    }
}

/// The folders discovery reads: Download Models' own folder and every
/// folder the configuration lists, read afresh so a reload is honoured.
fn library_roots(
    automatic_models_directory: &Path,
    reloadable_config: &RwLock<ResolvedRuntimeConfig>,
) -> Vec<PathBuf> {
    let mut roots = vec![automatic_models_directory.to_path_buf()];
    if let Ok(live_config) = reloadable_config.read() {
        roots.extend(live_config.configured_model_directories.iter().cloned());
    }
    roots
}

/// What the watch needs, owned by its task.
pub struct ModelFolderWatch {
    pub automatic_models_directory: PathBuf,
    pub reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    pub discovery_refresh: Arc<LibraryModelDiscoveryRefresh>,
}

/// Runs the watch until shutdown is requested.
pub fn spawn_model_folder_watch(
    model_folder_watch: ModelFolderWatch,
    mut shutdown_receiver: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let model_folder_watch = Arc::new(model_folder_watch);
        let mut acted_fingerprint = current_fingerprint(&model_folder_watch).await;
        let mut previous_tick_fingerprint = acted_fingerprint.clone();
        let mut watch_interval = tokio::time::interval(MODEL_FOLDER_WATCH_INTERVAL);
        watch_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = shutdown_receiver.changed() => return,
                _ = watch_interval.tick() => {}
            }
            let fingerprint = current_fingerprint(&model_folder_watch).await;
            let watch_step = model_folder_watch_step(
                &acted_fingerprint,
                &previous_tick_fingerprint,
                &fingerprint,
            );
            // A failed rescan leaves the acted fingerprint behind, so the
            // next tick tries again rather than waiting for another change.
            if watch_step == ModelFolderWatchStep::Rescan
                && rescan_library(&model_folder_watch).await
            {
                acted_fingerprint = fingerprint.clone();
            }
            previous_tick_fingerprint = fingerprint;
        }
    })
}

async fn current_fingerprint(model_folder_watch: &Arc<ModelFolderWatch>) -> ModelFolderFingerprint {
    let watch_for_reading = Arc::clone(model_folder_watch);
    tokio::task::spawn_blocking(move || {
        model_folder_fingerprint(&library_roots(
            &watch_for_reading.automatic_models_directory,
            &watch_for_reading.reloadable_config,
        ))
    })
    .await
    .unwrap_or_default()
}

/// The refresh waits on the worker, so it runs off the async threads.
/// Whether the library was rediscovered.
async fn rescan_library(model_folder_watch: &Arc<ModelFolderWatch>) -> bool {
    let watch_for_rescan = Arc::clone(model_folder_watch);
    let rescan_outcome = tokio::task::spawn_blocking(move || {
        watch_for_rescan
            .discovery_refresh
            .rescan()
            .map_err(|error| error.to_string())
    })
    .await;
    match rescan_outcome {
        Ok(Ok(library_change)) => {
            if !library_change.is_empty() {
                tracing::info!(
                    added = ?library_change.added_model_ids,
                    removed = ?library_change.removed_model_ids,
                    changed = ?library_change.changed_model_ids,
                    "model folders changed on disk; the model list was refreshed"
                );
            }
            true
        }
        Ok(Err(rescan_error)) => {
            tracing::warn!(error = %rescan_error, "model folders changed on disk, but rediscovery failed");
            false
        }
        Err(join_error) => {
            tracing::warn!(error = %join_error, "model folder rescan stopped unexpectedly");
            false
        }
    }
}

#[cfg(test)]
#[path = "model_folder_watch_tests.rs"]
mod tests;
