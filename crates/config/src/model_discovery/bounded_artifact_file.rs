//! Bounded artifact-sidecar reads shared by shallow model discovery contracts.

use std::fs;
use std::io::Read;
use std::path::Path;

use serde::de::DeserializeOwned;

pub(super) fn read_json<Document: DeserializeOwned>(
    file_path: &Path,
    maximum_bytes: u64,
) -> Result<Document, BoundedDocumentReadError> {
    let file_bytes = read_bounded_nonempty_file(file_path, maximum_bytes)?;
    serde_json::from_slice(&file_bytes).map_err(|_| BoundedDocumentReadError::MalformedJson)
}

pub(super) fn read_bounded_nonempty_file(
    file_path: &Path,
    maximum_bytes: u64,
) -> Result<Vec<u8>, BoundedFileReadError> {
    let file = fs::File::open(file_path).map_err(|_| BoundedFileReadError::Unavailable)?;
    let file_metadata = file
        .metadata()
        .map_err(|_| BoundedFileReadError::Unavailable)?;
    if !file_metadata.is_file() || file_metadata.len() == 0 || file_metadata.len() > maximum_bytes {
        return Err(BoundedFileReadError::InvalidSizeOrType);
    }
    let mut file_bytes = Vec::new();
    file.take(maximum_bytes + 1)
        .read_to_end(&mut file_bytes)
        .map_err(|_| BoundedFileReadError::Unavailable)?;
    (!file_bytes.is_empty() && file_bytes.len() as u64 <= maximum_bytes)
        .then_some(file_bytes)
        .ok_or(BoundedFileReadError::InvalidSizeOrType)
}

#[derive(Debug)]
pub(super) enum BoundedFileReadError {
    Unavailable,
    InvalidSizeOrType,
}

#[derive(Debug)]
pub(super) enum BoundedDocumentReadError {
    File,
    MalformedJson,
}

impl From<BoundedFileReadError> for BoundedDocumentReadError {
    fn from(_: BoundedFileReadError) -> Self {
        Self::File
    }
}
