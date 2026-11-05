//! Parses a safetensors header. Headers come from strangers' repositories, so
//! every offset and size is checked rather than trusted.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::error::VisionTowerError;

/// Bytes before the header holding its little-endian length.
pub const HEADER_LENGTH_PREFIX_BYTES: u64 = 8;
// The safetensors format's own limit; a model's real header is a few megabytes.
pub const MAXIMUM_HEADER_BYTES: u64 = 100 * 1024 * 1024;
const METADATA_KEY: &str = "__metadata__";

/// One tensor's place in its file. `data_start` is from the start of the file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorEntry {
    pub data_type: String,
    pub shape: Vec<u64>,
    pub data_start: u64,
    pub byte_count: u64,
}

impl TensorEntry {
    /// The product of the shape, saturating because shapes are outside data.
    #[must_use]
    pub fn element_count(&self) -> u64 {
        let mut element_count = 1_u64;
        for dimension in &self.shape {
            element_count = element_count.saturating_mul(*dimension);
        }
        element_count
    }
}

#[derive(Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// The header length from a file's first eight bytes, refused when absurd.
pub fn parse_header_length(file: &str, prefix_bytes: &[u8]) -> Result<u64, VisionTowerError> {
    let Ok(length_bytes) = <[u8; 8]>::try_from(prefix_bytes) else {
        return Err(malformed(
            file,
            "the file is shorter than its length prefix",
        ));
    };
    let header_length = u64::from_le_bytes(length_bytes);
    if header_length > MAXIMUM_HEADER_BYTES {
        return Err(malformed(
            file,
            &format!("its header claims {header_length} bytes"),
        ));
    }
    Ok(header_length)
}

pub fn parse_header(
    file: &str,
    header_bytes: &[u8],
    file_byte_count: Option<u64>,
) -> Result<BTreeMap<String, TensorEntry>, VisionTowerError> {
    let raw_entries: BTreeMap<String, serde_json::Value> =
        match serde_json::from_slice(header_bytes) {
            Ok(raw_entries) => raw_entries,
            Err(parse_error) => return Err(malformed(file, &parse_error.to_string())),
        };
    let data_section_start = HEADER_LENGTH_PREFIX_BYTES + header_bytes.len() as u64;
    let mut tensor_entries = BTreeMap::new();
    for (tensor_name, raw_entry) in raw_entries {
        if tensor_name == METADATA_KEY {
            continue;
        }
        let raw_tensor: RawTensorEntry = match serde_json::from_value(raw_entry) {
            Ok(raw_tensor) => raw_tensor,
            Err(parse_error) => {
                return Err(malformed(file, &format!("{tensor_name}: {parse_error}")));
            }
        };
        let tensor_entry = checked_entry(file, &tensor_name, raw_tensor, data_section_start)?;
        check_inside_file(file, &tensor_name, &tensor_entry, file_byte_count)?;
        tensor_entries.insert(tensor_name, tensor_entry);
    }
    Ok(tensor_entries)
}

fn checked_entry(
    file: &str,
    tensor_name: &str,
    raw_tensor: RawTensorEntry,
    data_section_start: u64,
) -> Result<TensorEntry, VisionTowerError> {
    let [start_offset, end_offset] = raw_tensor.data_offsets;
    let Some(byte_count) = end_offset.checked_sub(start_offset) else {
        return Err(malformed(
            file,
            &format!("{tensor_name} ends before it starts"),
        ));
    };
    // Checked against the end, because a listing may not state the file's size, so
    // this can be the only thing keeping a later `start + count` from overflowing.
    if data_section_start.checked_add(end_offset).is_none() {
        return Err(malformed(
            file,
            &format!("{tensor_name} ends past any file"),
        ));
    }
    let Some(data_start) = data_section_start.checked_add(start_offset) else {
        return Err(malformed(
            file,
            &format!("{tensor_name} starts past any file"),
        ));
    };
    let tensor_entry = TensorEntry {
        data_type: raw_tensor.dtype,
        shape: raw_tensor.shape,
        data_start,
        byte_count,
    };
    if let Some(element_bytes) = element_byte_width(&tensor_entry.data_type)
        && tensor_entry.element_count().checked_mul(element_bytes) != Some(byte_count)
    {
        return Err(malformed(
            file,
            &format!("{tensor_name}'s shape does not match its {byte_count} bytes"),
        ));
    }
    Ok(tensor_entry)
}

fn check_inside_file(
    file: &str,
    tensor_name: &str,
    tensor_entry: &TensorEntry,
    file_byte_count: Option<u64>,
) -> Result<(), VisionTowerError> {
    let Some(file_byte_count) = file_byte_count else {
        return Ok(());
    };
    match tensor_entry.data_start.checked_add(tensor_entry.byte_count) {
        Some(data_end) if data_end <= file_byte_count => Ok(()),
        _ => Err(malformed(
            file,
            &format!("{tensor_name} runs past the end of the file"),
        )),
    }
}

/// `None` for a type this crate does not know, which is not an error: a new type
/// should not stop someone reading the rest of the file.
#[must_use]
pub fn element_byte_width(data_type: &str) -> Option<u64> {
    match data_type {
        "BOOL" | "U8" | "I8" => Some(1),
        "BF16" | "F16" | "U16" | "I16" => Some(2),
        "F32" | "U32" | "I32" => Some(4),
        "F64" | "U64" | "I64" => Some(8),
        _ => None,
    }
}

fn malformed(file: &str, description: &str) -> VisionTowerError {
    VisionTowerError::SafetensorsMalformed {
        file: file.to_owned(),
        description: description.to_owned(),
    }
}

#[cfg(test)]
#[path = "safetensors_header_tests.rs"]
mod tests;
