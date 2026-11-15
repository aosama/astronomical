//! Keeps the download list in the models folder so it survives a restart.
//!
//! Only the download coordinator writes the list, and only while it holds its
//! command lock, so one writer at a time is guaranteed and a
//! write-then-rename is enough to never leave a half-written list behind.
//! The list is a few kilobytes, so the coordinator reads and writes it on its
//! own task rather than handing each access to a blocking thread.

use std::{
    fs,
    io::{self, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

use thiserror::Error;

use super::download_queue::DownloadQueue;

const DOWNLOAD_QUEUE_FILE_NAME: &str = ".download-queue.json";
const DOWNLOAD_QUEUE_TEMPORARY_FILE_NAME: &str = ".download-queue.json.tmp";
/// A list of every catalog model is a few kilobytes; anything far larger is
/// not a list this program wrote.
const MAXIMUM_DOWNLOAD_QUEUE_BYTES: u64 = 1_000_000;

/// Reads and writes the download list beside the models.
#[derive(Clone, Debug)]
pub struct DownloadQueueStore {
    models_directory: PathBuf,
}

#[derive(Debug, Error)]
pub enum DownloadQueueStoreError {
    #[error("could not write the download list at {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("could not serialize the download list: {0}")]
    Serialize(#[from] serde_json::Error),
}

impl DownloadQueueStore {
    #[must_use]
    pub const fn new(models_directory: PathBuf) -> Self {
        Self { models_directory }
    }

    #[must_use]
    pub fn queue_file_path(&self) -> PathBuf {
        self.models_directory.join(DOWNLOAD_QUEUE_FILE_NAME)
    }

    /// The saved list, or an empty one when none was saved. A list that
    /// cannot be read is logged and treated as empty: losing the list of
    /// models still to fetch is an inconvenience, while refusing every
    /// download over it would stop the person outright.
    #[must_use]
    pub fn load(&self) -> DownloadQueue {
        let queue_file_path = self.queue_file_path();
        match read_queue_text(&queue_file_path) {
            Ok(None) => DownloadQueue::default(),
            Ok(Some(queue_text)) => serde_json::from_str(&queue_text).unwrap_or_else(|parse_error| {
                tracing::warn!(%parse_error, path = %queue_file_path.display(), "the download list could not be read; starting an empty one");
                DownloadQueue::default()
            }),
            Err(read_error) => {
                tracing::warn!(%read_error, path = %queue_file_path.display(), "the download list could not be read; starting an empty one");
                DownloadQueue::default()
            }
        }
    }

    /// Saves the list whole, or removes the file when the list is empty so an
    /// idle models folder carries no leftover file.
    pub fn save(&self, download_queue: &DownloadQueue) -> Result<(), DownloadQueueStoreError> {
        let queue_file_path = self.queue_file_path();
        if download_queue == &DownloadQueue::default() {
            return match fs::remove_file(&queue_file_path) {
                Err(source) if source.kind() != io::ErrorKind::NotFound => {
                    Err(DownloadQueueStoreError::Write {
                        path: queue_file_path,
                        source,
                    })
                }
                _ => Ok(()),
            };
        }
        let queue_text = serde_json::to_vec_pretty(download_queue)?;
        let temporary_file_path = self
            .models_directory
            .join(DOWNLOAD_QUEUE_TEMPORARY_FILE_NAME);
        write_then_rename(&temporary_file_path, &queue_file_path, &queue_text).map_err(|source| {
            DownloadQueueStoreError::Write {
                path: queue_file_path.clone(),
                source,
            }
        })
    }
}

fn read_queue_text(queue_file_path: &Path) -> io::Result<Option<String>> {
    let queue_metadata = match fs::metadata(queue_file_path) {
        Ok(queue_metadata) => queue_metadata,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(source),
    };
    if queue_metadata.len() > MAXIMUM_DOWNLOAD_QUEUE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the download list is larger than any list this program writes",
        ));
    }
    fs::read_to_string(queue_file_path).map(Some)
}

/// Writes the list beside the models and renames it into place, then syncs
/// the folder so the rename survives a power loss. The temporary file is
/// opened without following a symbolic link, as the job store's files are,
/// so a link planted at its name cannot redirect the write elsewhere.
fn write_then_rename(
    temporary_file_path: &Path,
    queue_file_path: &Path,
    queue_text: &[u8],
) -> io::Result<()> {
    let models_directory = queue_file_path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "the download list has no folder",
        )
    })?;
    fs::create_dir_all(models_directory)?;
    let mut temporary_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(temporary_file_path)?;
    temporary_file.write_all(queue_text)?;
    temporary_file.sync_all()?;
    fs::rename(temporary_file_path, queue_file_path)?;
    fs::File::open(models_directory)?.sync_all()
}
