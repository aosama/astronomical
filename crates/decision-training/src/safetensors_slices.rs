//! Reads a few named tensors out of a safetensors file without reading the
//! whole file, widened to 32-bit floats.
//!
//! A Laya weights file is about 500 MB and Clef's head file about 240 MB,
//! while training needs a few megabytes of them. The safetensors layout is
//! an 8-byte little-endian header length, a JSON index of every tensor's
//! dtype, shape and byte range, then the data; so the index is read, and
//! only the wanted ranges after it.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde_json::Value;
use thiserror::Error;

/// No real index is anywhere near this; a larger length means a broken file.
const MAXIMUM_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// One tensor read out, widened to 32-bit floats.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedFloatTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// Why the tensors could not be read.
#[derive(Debug, Error)]
pub enum SafetensorsSliceError {
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("{path} is not a readable safetensors file: {reason}")]
    Malformed { path: PathBuf, reason: String },
    #[error("{path} has no tensor {tensor_name}")]
    MissingTensor { path: PathBuf, tensor_name: String },
    #[error("{path}: {tensor_name} is stored as {dtype}, which training cannot read")]
    UnreadableDtype {
        path: PathBuf,
        tensor_name: String,
        dtype: String,
    },
}

/// Reads `tensor_names` from the safetensors file at `file_path`.
pub fn read_named_float_tensors(
    file_path: &Path,
    tensor_names: &[&str],
) -> Result<Vec<NamedFloatTensor>, SafetensorsSliceError> {
    let mut weights_file = open_file(file_path)?;
    let index = read_index(&mut weights_file, file_path)?;
    let data_start = 8 + index.header_length;
    let mut named_tensors = Vec::with_capacity(tensor_names.len());
    for tensor_name in tensor_names {
        let Some(tensor_entry) = index.entries.get(*tensor_name) else {
            return Err(SafetensorsSliceError::MissingTensor {
                path: file_path.to_path_buf(),
                tensor_name: (*tensor_name).to_owned(),
            });
        };
        named_tensors.push(read_tensor(
            &mut weights_file,
            file_path,
            tensor_name,
            tensor_entry,
            data_start,
        )?);
    }
    Ok(named_tensors)
}

struct SafetensorsIndex {
    header_length: u64,
    entries: serde_json::Map<String, Value>,
}

fn open_file(file_path: &Path) -> Result<File, SafetensorsSliceError> {
    match File::open(file_path) {
        Ok(weights_file) => Ok(weights_file),
        Err(source) => Err(SafetensorsSliceError::Io {
            path: file_path.to_path_buf(),
            source,
        }),
    }
}

fn read_index(
    weights_file: &mut File,
    file_path: &Path,
) -> Result<SafetensorsIndex, SafetensorsSliceError> {
    let mut length_bytes = [0_u8; 8];
    read_exact(weights_file, &mut length_bytes, file_path)?;
    let header_length = u64::from_le_bytes(length_bytes);
    if header_length > MAXIMUM_HEADER_BYTES {
        return Err(malformed(file_path, "its index is implausibly large"));
    }
    let mut header_bytes = vec![0_u8; header_length as usize];
    read_exact(weights_file, &mut header_bytes, file_path)?;
    match serde_json::from_slice::<Value>(&header_bytes) {
        Ok(Value::Object(entries)) => Ok(SafetensorsIndex {
            header_length,
            entries,
        }),
        Ok(_) => Err(malformed(file_path, "its index is not a JSON object")),
        Err(parse_error) => Err(malformed(file_path, &parse_error.to_string())),
    }
}

fn read_tensor(
    weights_file: &mut File,
    file_path: &Path,
    tensor_name: &str,
    tensor_entry: &Value,
    data_start: u64,
) -> Result<NamedFloatTensor, SafetensorsSliceError> {
    let Some((dtype, shape, byte_range)) = entry_fields(tensor_entry) else {
        return Err(malformed(
            file_path,
            &format!("{tensor_name}'s entry is malformed"),
        ));
    };
    let Some(element_bytes) = element_byte_width(dtype) else {
        return Err(SafetensorsSliceError::UnreadableDtype {
            path: file_path.to_path_buf(),
            tensor_name: tensor_name.to_owned(),
            dtype: dtype.to_owned(),
        });
    };
    let mut element_count = 1;
    for dimension in &shape {
        element_count *= dimension;
    }
    let (range_start, range_end) = byte_range;
    if range_end < range_start || range_end - range_start != element_count * element_bytes {
        return Err(malformed(
            file_path,
            &format!("{tensor_name}'s byte range does not match its shape"),
        ));
    }
    let mut stored_bytes = vec![0_u8; range_end - range_start];
    if let Err(source) = weights_file.seek(SeekFrom::Start(data_start + range_start as u64)) {
        return Err(SafetensorsSliceError::Io {
            path: file_path.to_path_buf(),
            source,
        });
    }
    read_exact(weights_file, &mut stored_bytes, file_path)?;
    Ok(NamedFloatTensor {
        name: tensor_name.to_owned(),
        shape,
        values: widened_values(dtype, &stored_bytes),
    })
}

/// The dtype, shape and byte range of one index entry.
fn entry_fields(tensor_entry: &Value) -> Option<(&str, Vec<usize>, (usize, usize))> {
    let dtype = tensor_entry.get("dtype")?.as_str()?;
    let shape = entry_shape(tensor_entry.get("shape")?)?;
    let byte_range = entry_byte_range(tensor_entry.get("data_offsets")?)?;
    Some((dtype, shape, byte_range))
}

fn entry_shape(shape_value: &Value) -> Option<Vec<usize>> {
    let mut shape = Vec::new();
    for dimension in shape_value.as_array()? {
        shape.push(usize::try_from(dimension.as_u64()?).ok()?);
    }
    Some(shape)
}

fn entry_byte_range(offsets_value: &Value) -> Option<(usize, usize)> {
    let offsets = offsets_value.as_array()?;
    let range_start = usize::try_from(offsets.first()?.as_u64()?).ok()?;
    let range_end = usize::try_from(offsets.get(1)?.as_u64()?).ok()?;
    Some((range_start, range_end))
}

const fn element_byte_width(dtype: &str) -> Option<usize> {
    match dtype.as_bytes() {
        b"F32" => Some(4),
        b"F16" | b"BF16" => Some(2),
        _ => None,
    }
}

/// Every stored value as a 32-bit float; both 16-bit formats widen exactly.
fn widened_values(dtype: &str, stored_bytes: &[u8]) -> Vec<f32> {
    let mut values = Vec::new();
    if dtype == "F32" {
        let (whole_floats, _) = stored_bytes.as_chunks::<4>();
        for float_bytes in whole_floats {
            values.push(f32::from_le_bytes(*float_bytes));
        }
        return values;
    }
    let (whole_halves, _) = stored_bytes.as_chunks::<2>();
    for half_bytes in whole_halves {
        let half_bits = u16::from_le_bytes(*half_bytes);
        values.push(if dtype == "BF16" {
            f32::from_bits(u32::from(half_bits) << 16)
        } else {
            float16_to_float32(half_bits)
        });
    }
    values
}

/// IEEE 754 half precision to single precision, subnormals and specials included.
#[must_use]
pub fn float16_to_float32(half_bits: u16) -> f32 {
    let sign = u32::from(half_bits >> 15) << 31;
    let exponent = u32::from((half_bits >> 10) & 0x1f);
    let mantissa = u32::from(half_bits & 0x3ff);
    let single_bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // A subnormal half is mantissa · 2^-24; renormalize it.
            let mut shifted_mantissa = mantissa;
            let mut single_exponent = 127 - 15 + 1;
            while shifted_mantissa & 0x400 == 0 {
                shifted_mantissa <<= 1;
                single_exponent -= 1;
            }
            sign | (single_exponent << 23) | ((shifted_mantissa & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(single_bits)
}

fn read_exact(
    weights_file: &mut File,
    read_bytes: &mut [u8],
    file_path: &Path,
) -> Result<(), SafetensorsSliceError> {
    match weights_file.read_exact(read_bytes) {
        Ok(()) => Ok(()),
        Err(source) => Err(SafetensorsSliceError::Io {
            path: file_path.to_path_buf(),
            source,
        }),
    }
}

fn malformed(file_path: &Path, reason: &str) -> SafetensorsSliceError {
    SafetensorsSliceError::Malformed {
        path: file_path.to_path_buf(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
#[path = "safetensors_slices_tests.rs"]
mod tests;
