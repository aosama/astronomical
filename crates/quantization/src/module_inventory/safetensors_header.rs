//! Reads a safetensors shard's JSON header without touching tensor data.
//!
//! A shard is an eight-byte little-endian header length, the JSON header,
//! then the raw tensor bytes. Only the first two are read, so inventorying a
//! 300B checkpoint costs a few kilobytes of I/O per shard. The header is
//! parsed with `serde_json` directly rather than through the `safetensors`
//! crate's reader, which wants the whole file mapped to validate offsets.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;

use crate::error::QuantizationError;

/// Header size above which a shard is refused unread. Real headers are tens
/// of kilobytes even for 1,000-expert shards; 100 MB bounds a corrupt length
/// prefix before it becomes an allocation.
const MAXIMUM_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// One tensor's header entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeaderTensor {
    pub tensor_name: String,
    /// The safetensors dtype spelling, for example `BF16`, `F8_E4M3`, `U32`.
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// The fields of a header entry the inventory needs; offsets are ignored.
#[derive(Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
}

/// Reads every tensor entry of one shard, skipping `__metadata__`, in the
/// header's key order (sorted, as `BTreeMap` keeps it).
pub fn read_header_tensors(shard_path: &Path) -> Result<Vec<HeaderTensor>, QuantizationError> {
    let header_bytes = read_header_bytes(shard_path)?;
    let header_values: BTreeMap<String, serde_json::Value> =
        match serde_json::from_slice(&header_bytes) {
            Ok(header_values) => header_values,
            Err(parse_error) => {
                return Err(header_unreadable(shard_path, &parse_error.to_string()));
            }
        };
    let mut header_tensors = Vec::with_capacity(header_values.len());
    for (tensor_name, entry_value) in header_values {
        if tensor_name == "__metadata__" {
            continue;
        }
        let entry: HeaderEntry = match serde_json::from_value(entry_value) {
            Ok(entry) => entry,
            Err(parse_error) => {
                return Err(header_unreadable(shard_path, &parse_error.to_string()));
            }
        };
        header_tensors.push(HeaderTensor {
            tensor_name,
            dtype: entry.dtype,
            shape: entry.shape,
        });
    }
    Ok(header_tensors)
}

/// The raw JSON header bytes after the length prefix.
fn read_header_bytes(shard_path: &Path) -> Result<Vec<u8>, QuantizationError> {
    let mut shard_file = match File::open(shard_path) {
        Ok(shard_file) => shard_file,
        Err(open_error) => return Err(header_unreadable(shard_path, &open_error.to_string())),
    };
    let mut length_prefix = [0_u8; 8];
    if let Err(read_error) = shard_file.read_exact(&mut length_prefix) {
        return Err(header_unreadable(shard_path, &read_error.to_string()));
    }
    let header_length = u64::from_le_bytes(length_prefix);
    if header_length > MAXIMUM_HEADER_BYTES {
        return Err(header_unreadable(
            shard_path,
            &format!("header length {header_length} exceeds {MAXIMUM_HEADER_BYTES} bytes"),
        ));
    }
    let mut header_bytes = vec![0_u8; header_length as usize];
    if let Err(read_error) = shard_file.read_exact(&mut header_bytes) {
        return Err(header_unreadable(shard_path, &read_error.to_string()));
    }
    Ok(header_bytes)
}

fn header_unreadable(shard_path: &Path, description: &str) -> QuantizationError {
    QuantizationError::SafetensorsHeaderUnreadable {
        shard_path: shard_path.to_path_buf(),
        description: description.to_owned(),
    }
}
