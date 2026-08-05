//! Bounded parsing of the JSON prefix shared by persistent safetensors files.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::safetensors::{
    BoundedSafetensorsHeaderError, SafetensorsTensorView as PersistentSafetensorsTensorView,
    read_bounded_safetensors_json_header,
};

pub(crate) const MAXIMUM_PERSISTENT_SAFETENSORS_HEADER_LENGTH_BYTES: u64 = 1024 * 1024;

/// Parsed header data retained after bounded JSON parsing.
#[derive(Debug)]
pub(crate) struct PersistentSafetensorsHeader {
    pub(crate) tensor_views: HashMap<String, PersistentSafetensorsTensorView>,
    pub(crate) metadata: HashMap<String, String>,
    pub(crate) data_section_start_bytes: u64,
    pub(crate) file_size_bytes: u64,
}

/// One bounded failure while reading a persistent safetensors header.
#[derive(Debug, Error)]
pub(crate) enum PersistentSafetensorsHeaderError {
    #[error("failed to read persistent safetensors file metadata at {file_path:?}")]
    ReadFileMetadata {
        file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read persistent safetensors header bytes at {file_path:?}")]
    ReadHeaderBytes {
        file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "persistent safetensors header at {file_path:?} is {header_length_bytes} bytes, maximum {maximum_header_length_bytes}"
    )]
    HeaderLengthTooLarge {
        file_path: PathBuf,
        header_length_bytes: u64,
        maximum_header_length_bytes: u64,
    },
    #[error(
        "persistent safetensors file at {file_path:?} is truncated: expected {expected_minimum_bytes}, got {actual_file_size_bytes}"
    )]
    TruncatedFile {
        file_path: PathBuf,
        expected_minimum_bytes: u64,
        actual_file_size_bytes: u64,
    },
    #[error("persistent safetensors header at {file_path:?} is not valid JSON")]
    InvalidHeaderJson {
        file_path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

pub(crate) fn read_persistent_safetensors_header(
    file: &File,
    file_path: &Path,
) -> Result<PersistentSafetensorsHeader, PersistentSafetensorsHeaderError> {
    let file_size_bytes = file
        .metadata()
        .map_err(
            |source| PersistentSafetensorsHeaderError::ReadFileMetadata {
                file_path: file_path.to_path_buf(),
                source,
            },
        )?
        .len();
    let bounded_safetensors_json_header = read_bounded_safetensors_json_header(
        file,
        file_size_bytes,
        MAXIMUM_PERSISTENT_SAFETENSORS_HEADER_LENGTH_BYTES,
    )
    .map_err(|bounded_safetensors_header_error| {
        persistent_safetensors_header_error(bounded_safetensors_header_error, file_path)
    })?;
    let metadata = match bounded_safetensors_json_header.metadata_json_value {
        Some(metadata_json_value) => {
            serde_json::from_value(metadata_json_value).map_err(|source| {
                PersistentSafetensorsHeaderError::InvalidHeaderJson {
                    file_path: file_path.to_path_buf(),
                    source,
                }
            })?
        }
        None => HashMap::new(),
    };
    let mut tensor_views =
        HashMap::with_capacity(bounded_safetensors_json_header.tensor_json_values.len());
    for (header_entry_name, header_entry_json_value) in
        bounded_safetensors_json_header.tensor_json_values
    {
        let tensor_view = serde_json::from_value(header_entry_json_value).map_err(|source| {
            PersistentSafetensorsHeaderError::InvalidHeaderJson {
                file_path: file_path.to_path_buf(),
                source,
            }
        })?;
        tensor_views.insert(header_entry_name, tensor_view);
    }
    Ok(PersistentSafetensorsHeader {
        tensor_views,
        metadata,
        data_section_start_bytes: bounded_safetensors_json_header.data_section_start_bytes,
        file_size_bytes: bounded_safetensors_json_header.file_size_bytes,
    })
}

fn persistent_safetensors_header_error(
    bounded_safetensors_header_error: BoundedSafetensorsHeaderError,
    file_path: &Path,
) -> PersistentSafetensorsHeaderError {
    match bounded_safetensors_header_error {
        BoundedSafetensorsHeaderError::ReadLengthPrefix(source)
        | BoundedSafetensorsHeaderError::ReadHeader(source) => {
            PersistentSafetensorsHeaderError::ReadHeaderBytes {
                file_path: file_path.to_path_buf(),
                source,
            }
        }
        BoundedSafetensorsHeaderError::HeaderLengthTooLarge {
            header_length_bytes,
            maximum_header_length_bytes,
        } => PersistentSafetensorsHeaderError::HeaderLengthTooLarge {
            file_path: file_path.to_path_buf(),
            header_length_bytes,
            maximum_header_length_bytes,
        },
        BoundedSafetensorsHeaderError::HeaderBeyondFile {
            header_end_offset_bytes,
            file_size_bytes,
        } => PersistentSafetensorsHeaderError::TruncatedFile {
            file_path: file_path.to_path_buf(),
            expected_minimum_bytes: header_end_offset_bytes,
            actual_file_size_bytes: file_size_bytes,
        },
        BoundedSafetensorsHeaderError::InvalidHeaderJson(source) => {
            PersistentSafetensorsHeaderError::InvalidHeaderJson {
                file_path: file_path.to_path_buf(),
                source,
            }
        }
    }
}
