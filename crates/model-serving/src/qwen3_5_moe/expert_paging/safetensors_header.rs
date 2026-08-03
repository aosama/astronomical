//! Bounded safetensors header parsing for expert-page manifest construction.
//!
//! Reads only the safetensors framing (8-byte length prefix + JSON header)
//! without mapping the payload into memory. This module is pure Rust with no
//! MLX dependency and can be unit-tested against any safetensors file.

use std::fs::File;
use std::path::Path;

use thiserror::Error;

use crate::bounded_safetensors_header::{
    BoundedSafetensorsHeaderError, SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES,
    read_bounded_safetensors_json_header,
};

/// Maximum safetensors header size accepted for validation.
/// This bound limits header-only reads before any expert payload is mapped.
const MAXIMUM_HEADER_BYTES: usize = 64 * 1024 * 1024;

/// Byte widths for safetensors dtype strings.
/// These deliberately match MLX 0.32's dtype_from_safetensor_str().
/// Accepting newer spellings here would defer a deterministic manifest
/// error until the later native MLX loader boundary.
static DTYPE_BYTE_WIDTHS: &[(&str, usize)] = &[
    ("BOOL", 1),
    ("I8", 1),
    ("U8", 1),
    ("F8_E4M3", 1),
    ("F8_E5M2", 1),
    ("I16", 2),
    ("U16", 2),
    ("F16", 2),
    ("BF16", 2),
    ("I32", 4),
    ("U32", 4),
    ("F32", 4),
    ("I64", 8),
    ("U64", 8),
];

/// Typed failures during safetensors header parsing and validation.
#[derive(Debug, Error)]
pub enum SafetensorsHeaderError {
    #[error("safetensors file does not exist: {file_path:?}")]
    FileNotFound { file_path: std::path::PathBuf },
    #[error("safetensors path is not a regular file: {file_path:?}")]
    NotARegularFile { file_path: std::path::PathBuf },
    #[error(
        "safetensors file is too small to contain a header length prefix: {file_size_bytes} bytes"
    )]
    FileTooSmall { file_size_bytes: u64 },
    #[error(
        "safetensors header length {header_length_bytes} exceeds the safety limit of {maximum_header_bytes} bytes"
    )]
    HeaderTooLarge {
        header_length_bytes: usize,
        maximum_header_bytes: usize,
    },
    #[error(
        "safetensors header extends beyond the file: header ends at {header_end_offset} but file is {file_size_bytes} bytes"
    )]
    HeaderBeyondFile {
        header_end_offset: u64,
        file_size_bytes: u64,
    },
    #[error(
        "safetensors header is truncated: expected {expected_bytes} bytes, read {actual_bytes}"
    )]
    HeaderTruncated {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    #[error("safetensors header must contain valid JSON")]
    HeaderNotJson(#[source] serde_json::Error),
    #[error("safetensors header must be a JSON object, got {actual_type}")]
    HeaderNotObject { actual_type: String },
    #[error("tensor {tensor_name:?} has an unsupported dtype: {dtype:?}")]
    UnsupportedDtype { tensor_name: String, dtype: String },
    #[error("tensor {tensor_name:?} has an invalid shape")]
    InvalidShape { tensor_name: String },
    #[error("tensor {tensor_name:?} has invalid data_offsets")]
    InvalidDataOffsets { tensor_name: String },
    #[error(
        "tensor {tensor_name:?} data_offsets are outside the file payload: [{start}, {end}) vs payload size {payload_byte_count}"
    )]
    DataOffsetsOutsidePayload {
        tensor_name: String,
        start: u64,
        end: u64,
        payload_byte_count: u64,
    },
    #[error(
        "tensor {tensor_name:?} byte count mismatch: header declares {declared_bytes} bytes but shape×dtype expects {expected_bytes} bytes"
    )]
    ByteCountMismatch {
        tensor_name: String,
        declared_bytes: u64,
        expected_bytes: u64,
    },
    #[error("I/O error reading safetensors file: {0}")]
    Io(#[from] std::io::Error),
}

/// One validated tensor entry extracted from a safetensors header.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorHeaderEntry {
    pub tensor_name: String,
    pub dtype: SafetensorsDtype,
    pub shape: Vec<usize>,
    /// Byte offset from the start of the file where this tensor's payload begins.
    pub data_start_offset: u64,
    /// Byte offset from the start of the file where this tensor's payload ends.
    pub data_end_offset: u64,
}

/// Supported safetensors dtypes with known byte widths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafetensorsDtype {
    Bool,
    Int8,
    Uint8,
    Float8E4M3,
    Float8E5M2,
    Int16,
    Uint16,
    Float16,
    BFloat16,
    Int32,
    Uint32,
    Float32,
    Int64,
    Uint64,
}

impl std::fmt::Display for SafetensorsDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl SafetensorsDtype {
    /// Returns the byte width of one element for this dtype.
    #[must_use]
    pub const fn byte_width(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::Uint8 | Self::Float8E4M3 | Self::Float8E5M2 => 1,
            Self::Int16 | Self::Uint16 | Self::Float16 | Self::BFloat16 => 2,
            Self::Int32 | Self::Uint32 | Self::Float32 => 4,
            Self::Int64 | Self::Uint64 => 8,
        }
    }

    /// Returns the safetensors dtype string representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::Int8 => "I8",
            Self::Uint8 => "U8",
            Self::Float8E4M3 => "F8_E4M3",
            Self::Float8E5M2 => "F8_E5M2",
            Self::Int16 => "I16",
            Self::Uint16 => "U16",
            Self::Float16 => "F16",
            Self::BFloat16 => "BF16",
            Self::Int32 => "I32",
            Self::Uint32 => "U32",
            Self::Float32 => "F32",
            Self::Int64 => "I64",
            Self::Uint64 => "U64",
        }
    }
}

/// The validated safetensors header and its file context.
#[derive(Clone, Debug)]
pub struct SafetensorsHeader {
    pub tensor_entries: Vec<TensorHeaderEntry>,
    /// Byte offset from the start of the file where the payload begins
    /// (immediately after the 8-byte length prefix + JSON header).
    pub payload_start_offset: u64,
    pub total_file_size_bytes: u64,
}

impl SafetensorsHeader {
    /// Returns the byte width for a safetensors dtype string, or `None` if unsupported.
    pub fn byte_width_for_dtype(dtype_str: &str) -> Option<usize> {
        DTYPE_BYTE_WIDTHS
            .iter()
            .find(|(name, _)| *name == dtype_str)
            .map(|(_, width)| *width)
    }

    /// Returns the tensor entry for the given tensor name, or `None` if not found.
    #[must_use]
    pub fn tensor_entry_for_name(&self, tensor_name: &str) -> Option<&TensorHeaderEntry> {
        self.tensor_entries
            .iter()
            .find(|entry| entry.tensor_name == tensor_name)
    }
}

/// Parses and validates a safetensors header from a file path.
///
/// Reads only the framing bytes (8-byte length prefix + JSON header) without
/// mapping the payload into memory. Validates dtype, shape, and data offsets
/// against the file's actual size.
pub fn parse_safetensors_header(
    file_path: &Path,
) -> Result<SafetensorsHeader, SafetensorsHeaderError> {
    let file_path = std::path::PathBuf::from(file_path);
    let file_metadata = std::fs::metadata(&file_path).map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            SafetensorsHeaderError::FileNotFound {
                file_path: file_path.clone(),
            }
        } else {
            SafetensorsHeaderError::Io(source)
        }
    })?;
    if !file_metadata.is_file() {
        return Err(SafetensorsHeaderError::NotARegularFile { file_path });
    }
    let file_size_bytes = file_metadata.len();
    if file_size_bytes < SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES {
        return Err(SafetensorsHeaderError::FileTooSmall { file_size_bytes });
    }

    let safetensors_file = File::open(&file_path).map_err(SafetensorsHeaderError::Io)?;
    let bounded_safetensors_json_header = read_bounded_safetensors_json_header(
        &safetensors_file,
        file_size_bytes,
        MAXIMUM_HEADER_BYTES as u64,
    )
    .map_err(expert_safetensors_header_error)?;
    let payload_start_offset = bounded_safetensors_json_header.data_section_start_bytes;
    let header_mapping = bounded_safetensors_json_header.tensor_json_values;

    let payload_byte_count = file_size_bytes - payload_start_offset;
    let mut tensor_entries = Vec::with_capacity(header_mapping.len());
    for (tensor_name, tensor_value) in &header_mapping {
        let tensor_header = match tensor_value.as_object() {
            Some(obj) => obj,
            None => {
                return Err(SafetensorsHeaderError::HeaderNotObject {
                    actual_type: tensor_value.to_string(),
                });
            }
        };
        let dtype_str = tensor_header
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SafetensorsHeaderError::UnsupportedDtype {
                tensor_name: tensor_name.clone(),
                dtype: "missing".to_owned(),
            })?;
        let dtype =
            parse_dtype(dtype_str).ok_or_else(|| SafetensorsHeaderError::UnsupportedDtype {
                tensor_name: tensor_name.clone(),
                dtype: dtype_str.to_owned(),
            })?;
        let shape = validate_shape(tensor_name, tensor_header.get("shape"))?;
        let (data_start_offset, data_end_offset) = validate_data_offsets(
            tensor_name,
            tensor_header.get("data_offsets"),
            payload_byte_count,
        )?;
        let declared_byte_count = data_end_offset - data_start_offset;
        let element_count = shape.iter().try_fold(1_usize, |product, &dimension| {
            product.checked_mul(dimension)
        });
        let expected_byte_count = element_count
            .map(|count| count * dtype.byte_width())
            .ok_or_else(|| SafetensorsHeaderError::ByteCountMismatch {
                tensor_name: tensor_name.clone(),
                declared_bytes: declared_byte_count,
                expected_bytes: 0,
            })?;
        if declared_byte_count != expected_byte_count as u64 {
            return Err(SafetensorsHeaderError::ByteCountMismatch {
                tensor_name: tensor_name.clone(),
                declared_bytes: declared_byte_count,
                expected_bytes: expected_byte_count as u64,
            });
        }
        // Convert payload-relative offsets to file-relative offsets.
        let file_relative_start = payload_start_offset + data_start_offset;
        let file_relative_end = payload_start_offset + data_end_offset;
        tensor_entries.push(TensorHeaderEntry {
            tensor_name: tensor_name.clone(),
            dtype,
            shape,
            data_start_offset: file_relative_start,
            data_end_offset: file_relative_end,
        });
    }
    Ok(SafetensorsHeader {
        tensor_entries,
        payload_start_offset,
        total_file_size_bytes: file_size_bytes,
    })
}

fn expert_safetensors_header_error(
    bounded_safetensors_header_error: BoundedSafetensorsHeaderError,
) -> SafetensorsHeaderError {
    match bounded_safetensors_header_error {
        BoundedSafetensorsHeaderError::ReadLengthPrefix(source)
        | BoundedSafetensorsHeaderError::ReadHeader(source) => SafetensorsHeaderError::Io(source),
        BoundedSafetensorsHeaderError::HeaderLengthTooLarge {
            header_length_bytes,
            maximum_header_length_bytes,
        } => SafetensorsHeaderError::HeaderTooLarge {
            header_length_bytes: safetensors_header_length_as_usize(header_length_bytes),
            maximum_header_bytes: safetensors_header_length_as_usize(maximum_header_length_bytes),
        },
        BoundedSafetensorsHeaderError::HeaderBeyondFile {
            header_end_offset_bytes,
            file_size_bytes,
        } => SafetensorsHeaderError::HeaderBeyondFile {
            header_end_offset: header_end_offset_bytes,
            file_size_bytes,
        },
        BoundedSafetensorsHeaderError::InvalidHeaderJson(source) => {
            SafetensorsHeaderError::HeaderNotJson(source)
        }
    }
}

fn safetensors_header_length_as_usize(header_length_bytes: u64) -> usize {
    match usize::try_from(header_length_bytes) {
        Ok(header_length_usize) => header_length_usize,
        Err(_) => usize::MAX,
    }
}

fn parse_dtype(dtype_str: &str) -> Option<SafetensorsDtype> {
    match dtype_str {
        "BOOL" => Some(SafetensorsDtype::Bool),
        "I8" => Some(SafetensorsDtype::Int8),
        "U8" => Some(SafetensorsDtype::Uint8),
        "F8_E4M3" => Some(SafetensorsDtype::Float8E4M3),
        "F8_E5M2" => Some(SafetensorsDtype::Float8E5M2),
        "I16" => Some(SafetensorsDtype::Int16),
        "U16" => Some(SafetensorsDtype::Uint16),
        "F16" => Some(SafetensorsDtype::Float16),
        "BF16" => Some(SafetensorsDtype::BFloat16),
        "I32" => Some(SafetensorsDtype::Int32),
        "U32" => Some(SafetensorsDtype::Uint32),
        "F32" => Some(SafetensorsDtype::Float32),
        "I64" => Some(SafetensorsDtype::Int64),
        "U64" => Some(SafetensorsDtype::Uint64),
        _ => None,
    }
}

fn validate_shape(
    tensor_name: &str,
    shape_value: Option<&serde_json::Value>,
) -> Result<Vec<usize>, SafetensorsHeaderError> {
    let shape_array = match shape_value.and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => {
            return Err(SafetensorsHeaderError::InvalidShape {
                tensor_name: tensor_name.to_owned(),
            });
        }
    };
    let mut shape = Vec::with_capacity(shape_array.len());
    for dimension in shape_array {
        let dimension_value =
            dimension
                .as_i64()
                .ok_or_else(|| SafetensorsHeaderError::InvalidShape {
                    tensor_name: tensor_name.to_owned(),
                })?;
        if dimension_value < 0 {
            return Err(SafetensorsHeaderError::InvalidShape {
                tensor_name: tensor_name.to_owned(),
            });
        }
        shape.push(dimension_value as usize);
    }
    Ok(shape)
}

fn validate_data_offsets(
    tensor_name: &str,
    offsets_value: Option<&serde_json::Value>,
    payload_byte_count: u64,
) -> Result<(u64, u64), SafetensorsHeaderError> {
    let offsets_array = match offsets_value.and_then(|v| v.as_array()) {
        Some(arr) if arr.len() == 2 => arr,
        _ => {
            return Err(SafetensorsHeaderError::InvalidDataOffsets {
                tensor_name: tensor_name.to_owned(),
            });
        }
    };
    let start =
        offsets_array[0]
            .as_i64()
            .ok_or_else(|| SafetensorsHeaderError::InvalidDataOffsets {
                tensor_name: tensor_name.to_owned(),
            })?;
    let end =
        offsets_array[1]
            .as_i64()
            .ok_or_else(|| SafetensorsHeaderError::InvalidDataOffsets {
                tensor_name: tensor_name.to_owned(),
            })?;
    if start < 0 || end < start {
        return Err(SafetensorsHeaderError::InvalidDataOffsets {
            tensor_name: tensor_name.to_owned(),
        });
    }
    let start_u64 = start as u64;
    let end_u64 = end as u64;
    if end_u64 > payload_byte_count {
        return Err(SafetensorsHeaderError::DataOffsetsOutsidePayload {
            tensor_name: tensor_name.to_owned(),
            start: start_u64,
            end: end_u64,
            payload_byte_count,
        });
    }
    Ok((start_u64, end_u64))
}
