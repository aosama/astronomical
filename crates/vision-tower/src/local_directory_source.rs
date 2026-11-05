//! Reads an artifact from a directory on this machine, following symlinks so a
//! Hugging Face cache snapshot reads like any other folder. A snapshot's path also
//! names its repository and commit, so a local copy keeps its provenance.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::artifact_source::{ArtifactFile, ArtifactListing};
use crate::error::VisionTowerError;

// Deep enough for sidecars such as `optiq/optiq_vision.safetensors`, and shallow
// enough that pointing this at a broad folder does not walk a whole disk.
const MAXIMUM_DIRECTORY_DEPTH: usize = 3;
const CACHE_REPOSITORY_PREFIX: &str = "models--";
const CACHE_REPOSITORY_SEPARATOR: &str = "--";
const CACHE_SNAPSHOTS_FOLDER: &str = "snapshots";

#[derive(Debug)]
pub struct LocalDirectorySource {
    root_directory: PathBuf,
    listing: ArtifactListing,
}

impl LocalDirectorySource {
    pub fn open(root_directory: &Path) -> Result<Self, VisionTowerError> {
        let mut files = Vec::new();
        collect_files(root_directory, "", 0, &mut files)?;
        files.sort_by_key(relative_path_of);
        let (origin_repository, pinned_revision) = match cache_snapshot_origin(root_directory) {
            Some((origin_repository, revision)) => (Some(origin_repository), Some(revision)),
            None => (None, None),
        };
        Ok(Self {
            root_directory: root_directory.to_path_buf(),
            listing: ArtifactListing {
                display_name: root_directory.display().to_string(),
                origin_repository,
                pinned_revision,
                license: None,
                pipeline_tag: None,
                files,
            },
        })
    }

    #[must_use]
    pub const fn listing(&self) -> &ArtifactListing {
        &self.listing
    }

    #[must_use]
    pub fn root_directory(&self) -> &Path {
        &self.root_directory
    }

    pub fn read_whole_file(
        &self,
        relative_path: &str,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, VisionTowerError> {
        let file_path = self.root_directory.join(relative_path);
        let file_size = match fs::metadata(&file_path) {
            Ok(file_metadata) => file_metadata.len(),
            Err(read_error) => return Err(unreadable(&file_path, &read_error)),
        };
        if file_size > maximum_bytes {
            return Err(VisionTowerError::FileTooLarge {
                file: relative_path.to_owned(),
                maximum_bytes,
            });
        }
        match fs::read(&file_path) {
            Ok(file_contents) => Ok(file_contents),
            Err(read_error) => Err(unreadable(&file_path, &read_error)),
        }
    }

    pub fn read_byte_range(
        &self,
        relative_path: &str,
        start_byte: u64,
        byte_count: u64,
    ) -> Result<Vec<u8>, VisionTowerError> {
        let file_path = self.root_directory.join(relative_path);
        let mut range_bytes = vec![0_u8; byte_count as usize];
        let read_outcome = read_range_into(&file_path, start_byte, &mut range_bytes);
        match read_outcome {
            Ok(()) => Ok(range_bytes),
            Err(read_error) => Err(unreadable(&file_path, &read_error)),
        }
    }
}

/// `(organization/name, commit)` for a path ending
/// `models--organization--name/snapshots/commit`.
fn cache_snapshot_origin(root_directory: &Path) -> Option<(String, String)> {
    let revision = root_directory.file_name()?.to_str()?;
    let snapshots_folder = root_directory.parent()?;
    if snapshots_folder.file_name()?.to_str()? != CACHE_SNAPSHOTS_FOLDER {
        return None;
    }
    let cache_folder_name = snapshots_folder.parent()?.file_name()?.to_str()?;
    let encoded_repository = cache_folder_name.strip_prefix(CACHE_REPOSITORY_PREFIX)?;
    let (organization, model_name) = encoded_repository.split_once(CACHE_REPOSITORY_SEPARATOR)?;
    Some((format!("{organization}/{model_name}"), revision.to_owned()))
}

fn relative_path_of(artifact_file: &ArtifactFile) -> String {
    artifact_file.relative_path.clone()
}

fn read_range_into(
    file_path: &Path,
    start_byte: u64,
    range_bytes: &mut [u8],
) -> std::io::Result<()> {
    let mut opened_file = File::open(file_path)?;
    opened_file.seek(SeekFrom::Start(start_byte))?;
    opened_file.read_exact(range_bytes)
}

fn collect_files(
    directory: &Path,
    relative_prefix: &str,
    depth: usize,
    files: &mut Vec<ArtifactFile>,
) -> Result<(), VisionTowerError> {
    let directory_entries = match fs::read_dir(directory) {
        Ok(directory_entries) => directory_entries,
        Err(read_error) => return Err(unreadable(directory, &read_error)),
    };
    for directory_entry in directory_entries {
        let directory_entry = match directory_entry {
            Ok(directory_entry) => directory_entry,
            Err(read_error) => return Err(unreadable(directory, &read_error)),
        };
        let Some(entry_name) = directory_entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        // Download bookkeeping such as `.cache` is not part of the artifact.
        if entry_name.starts_with('.') {
            continue;
        }
        let entry_path = directory_entry.path();
        let entry_metadata = match fs::metadata(&entry_path) {
            Ok(entry_metadata) => entry_metadata,
            Err(read_error) => return Err(unreadable(&entry_path, &read_error)),
        };
        let relative_path = format!("{relative_prefix}{entry_name}");
        if entry_metadata.is_dir() {
            if depth + 1 < MAXIMUM_DIRECTORY_DEPTH {
                let nested_prefix = format!("{relative_path}/");
                collect_files(&entry_path, &nested_prefix, depth + 1, files)?;
            }
        } else {
            files.push(ArtifactFile {
                relative_path,
                byte_count: Some(entry_metadata.len()),
            });
        }
    }
    Ok(())
}

fn unreadable(path: &Path, read_error: &std::io::Error) -> VisionTowerError {
    VisionTowerError::FileUnreadable {
        path: path.to_path_buf(),
        cause: read_error.to_string(),
    }
}

#[cfg(test)]
#[path = "local_directory_source_tests.rs"]
mod tests;
