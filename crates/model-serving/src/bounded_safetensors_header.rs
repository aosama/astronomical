//! Shared bounded safetensors framing and raw JSON-header parsing.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;

use serde::Deserialize;
use serde_json::{Map, Value};

/// Size of the safetensors little-endian header-length prefix.
pub(crate) const SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES: u64 = 8;

/// Raw safetensors header entries after a bounded file read and JSON parse.
#[derive(Debug)]
pub(crate) struct BoundedSafetensorsJsonHeader {
    pub(crate) tensor_json_values: Map<String, Value>,
    pub(crate) metadata_json_value: Option<Value>,
    pub(crate) data_section_start_bytes: u64,
    pub(crate) file_size_bytes: u64,
}

/// One raw tensor declaration shared by artifact and persistent-cache readers.
#[derive(Debug, Deserialize)]
pub(crate) struct SafetensorsTensorView {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data_offsets: [u64; 2],
}

impl SafetensorsTensorView {
    pub(crate) fn data_start_offset(&self) -> u64 {
        self.data_offsets[0]
    }

    pub(crate) fn data_end_offset(&self) -> u64 {
        self.data_offsets[1]
    }
}

/// Framing or raw-JSON failures shared by safetensors consumers.
#[derive(Debug)]
pub(crate) enum BoundedSafetensorsHeaderError {
    ReadLengthPrefix(io::Error),
    HeaderLengthTooLarge {
        header_length_bytes: u64,
        maximum_header_length_bytes: u64,
    },
    HeaderBeyondFile {
        header_end_offset_bytes: u64,
        file_size_bytes: u64,
    },
    ReadHeader(io::Error),
    InvalidHeaderJson(serde_json::Error),
}

/// Reads one bounded safetensors header without touching payload bytes.
pub(crate) fn read_bounded_safetensors_json_header(
    safetensors_file: &File,
    file_size_bytes: u64,
    maximum_header_length_bytes: u64,
) -> Result<BoundedSafetensorsJsonHeader, BoundedSafetensorsHeaderError> {
    let mut header_length_prefix_bytes = [0_u8; SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES as usize];
    read_exact_at(safetensors_file, &mut header_length_prefix_bytes, 0)
        .map_err(BoundedSafetensorsHeaderError::ReadLengthPrefix)?;
    let header_length_bytes = u64::from_le_bytes(header_length_prefix_bytes);
    if header_length_bytes > maximum_header_length_bytes {
        return Err(BoundedSafetensorsHeaderError::HeaderLengthTooLarge {
            header_length_bytes,
            maximum_header_length_bytes,
        });
    }
    let header_length_usize = usize::try_from(header_length_bytes).map_err(|_| {
        BoundedSafetensorsHeaderError::HeaderLengthTooLarge {
            header_length_bytes,
            maximum_header_length_bytes,
        }
    })?;
    let data_section_start_bytes = SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES
        .checked_add(header_length_bytes)
        .ok_or(BoundedSafetensorsHeaderError::HeaderLengthTooLarge {
            header_length_bytes,
            maximum_header_length_bytes,
        })?;
    if data_section_start_bytes > file_size_bytes {
        return Err(BoundedSafetensorsHeaderError::HeaderBeyondFile {
            header_end_offset_bytes: data_section_start_bytes,
            file_size_bytes,
        });
    }
    let mut header_json_bytes = vec![0_u8; header_length_usize];
    read_exact_at(
        safetensors_file,
        &mut header_json_bytes,
        SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES,
    )
    .map_err(BoundedSafetensorsHeaderError::ReadHeader)?;
    let mut raw_header_entries: Map<String, Value> = serde_json::from_slice(&header_json_bytes)
        .map_err(BoundedSafetensorsHeaderError::InvalidHeaderJson)?;
    let metadata_json_value = raw_header_entries.remove("__metadata__");
    Ok(BoundedSafetensorsJsonHeader {
        tensor_json_values: raw_header_entries,
        metadata_json_value,
        data_section_start_bytes,
        file_size_bytes,
    })
}

fn read_exact_at(
    safetensors_file: &File,
    target_bytes: &mut [u8],
    start_offset_bytes: u64,
) -> io::Result<()> {
    let mut completed_byte_count = 0_usize;
    while completed_byte_count < target_bytes.len() {
        let completed_offset_bytes =
            u64::try_from(completed_byte_count).map_err(io::Error::other)?;
        let read_offset_bytes = start_offset_bytes
            .checked_add(completed_offset_bytes)
            .ok_or_else(|| io::Error::other("safetensors read offset overflowed"))?;
        let bytes_read = safetensors_file
            .read_at(&mut target_bytes[completed_byte_count..], read_offset_bytes)?;
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "safetensors file ended before the requested bytes",
            ));
        }
        completed_byte_count = completed_byte_count
            .checked_add(bytes_read)
            .ok_or_else(|| io::Error::other("safetensors completed byte count overflowed"))?;
    }
    Ok(())
}
