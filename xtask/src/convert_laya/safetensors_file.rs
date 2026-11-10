//! Reading and writing the safetensors container, without loading a whole
//! checkpoint into memory.
//!
//! A safetensors file is an 8-byte little-endian header length, a JSON
//! header naming every tensor with its type, shape and byte range, and then
//! the tensor bytes. The converter reads the header once, pulls one source
//! tensor at a time, and appends each converted tensor to the output as it
//! is produced.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde_json::{Map, Value};

/// A safetensors header larger than this is not a Laya checkpoint's; the
/// bound stops a corrupt length from allocating gigabytes.
const MAXIMUM_HEADER_BYTES: u64 = 16 * 1024 * 1024;
/// Readers map tensor data at 8-byte alignment, so the header is padded.
const HEADER_ALIGNMENT_BYTES: usize = 8;

/// The element types a Laya checkpoint and its conversion use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TensorElementType {
    Float16,
    Float32,
    Unsigned32,
}

impl TensorElementType {
    pub(crate) const fn safetensors_name(self) -> &'static str {
        match self {
            Self::Float16 => "F16",
            Self::Float32 => "F32",
            Self::Unsigned32 => "U32",
        }
    }

    pub(crate) const fn bytes_per_element(self) -> u64 {
        match self {
            Self::Float16 => 2,
            Self::Float32 | Self::Unsigned32 => 4,
        }
    }

    fn from_safetensors_name(type_name: &str) -> Option<Self> {
        match type_name {
            "F16" => Some(Self::Float16),
            "F32" => Some(Self::Float32),
            "U32" => Some(Self::Unsigned32),
            _ => None,
        }
    }
}

/// One tensor's place in a safetensors file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorRecord {
    pub(crate) tensor_name: String,
    pub(crate) element_type: TensorElementType,
    pub(crate) shape: Vec<u64>,
    /// Offset of the first byte, counted from the start of the tensor data.
    pub(crate) data_start: u64,
    pub(crate) byte_count: u64,
}

impl TensorRecord {
    pub(crate) fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// An open source checkpoint: its tensor records and where its data begins.
pub(crate) struct SourceCheckpoint {
    checkpoint_file: File,
    tensor_data_offset: u64,
    /// Every tensor, in name order so a conversion is reproducible.
    pub(crate) tensor_records: Vec<TensorRecord>,
}

impl SourceCheckpoint {
    pub(crate) fn open(weights_path: &Path) -> Result<Self, String> {
        let mut checkpoint_file = File::open(weights_path)
            .map_err(|open_error| format!("could not open model.safetensors: {open_error}"))?;
        let mut header_length_bytes = [0_u8; 8];
        checkpoint_file
            .read_exact(&mut header_length_bytes)
            .map_err(|read_error| format!("model.safetensors has no header: {read_error}"))?;
        let header_byte_count = u64::from_le_bytes(header_length_bytes);
        if header_byte_count > MAXIMUM_HEADER_BYTES {
            return Err(format!(
                "model.safetensors declares a {header_byte_count}-byte header, above the \
                 {MAXIMUM_HEADER_BYTES}-byte limit"
            ));
        }
        let mut header_bytes = vec![0_u8; header_byte_count as usize];
        checkpoint_file
            .read_exact(&mut header_bytes)
            .map_err(|read_error| format!("model.safetensors header is cut short: {read_error}"))?;
        let header: Value = serde_json::from_slice(&header_bytes)
            .map_err(|json_error| format!("model.safetensors header is not JSON: {json_error}"))?;
        let mut tensor_records = tensor_records_from_header(&header)?;
        let file_length = checkpoint_file
            .metadata()
            .map_err(|metadata_error| {
                format!("could not read the size of model.safetensors: {metadata_error}")
            })?
            .len();
        let tensor_data_length = file_length.saturating_sub(8 + header_byte_count);
        refuse_tensors_past_the_end(&tensor_records, tensor_data_length)?;
        tensor_records.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
        Ok(Self {
            checkpoint_file,
            tensor_data_offset: 8 + header_byte_count,
            tensor_records,
        })
    }

    /// Reads one tensor's bytes.
    pub(crate) fn read_tensor_bytes(
        &mut self,
        tensor_record: &TensorRecord,
    ) -> Result<Vec<u8>, String> {
        let mut tensor_bytes = vec![0_u8; tensor_record.byte_count as usize];
        self.checkpoint_file
            .seek(SeekFrom::Start(
                self.tensor_data_offset + tensor_record.data_start,
            ))
            .and_then(|_| self.checkpoint_file.read_exact(&mut tensor_bytes))
            .map_err(|read_error| {
                format!(
                    "could not read tensor {}: {read_error}",
                    tensor_record.tensor_name
                )
            })?;
        Ok(tensor_bytes)
    }
}

/// Refuses a header that places a tensor past the end of the file. Checked
/// before any tensor is read, so a corrupt header cannot ask for an
/// allocation the file could never fill.
fn refuse_tensors_past_the_end(
    tensor_records: &[TensorRecord],
    tensor_data_length: u64,
) -> Result<(), String> {
    for tensor_record in tensor_records {
        let tensor_data_end = tensor_record
            .data_start
            .saturating_add(tensor_record.byte_count);
        if tensor_data_end > tensor_data_length {
            return Err(format!(
                "tensor {} ends at byte {tensor_data_end}, past the {tensor_data_length} bytes of \
                 tensor data in model.safetensors",
                tensor_record.tensor_name
            ));
        }
    }
    Ok(())
}

fn tensor_records_from_header(header: &Value) -> Result<Vec<TensorRecord>, String> {
    let Some(header_members) = header.as_object() else {
        return Err("model.safetensors header is not a JSON object".to_owned());
    };
    let mut tensor_records = Vec::with_capacity(header_members.len());
    for (tensor_name, tensor_description) in header_members {
        if tensor_name == "__metadata__" {
            continue;
        }
        tensor_records.push(tensor_record_from_description(
            tensor_name,
            tensor_description,
        )?);
    }
    Ok(tensor_records)
}

fn tensor_record_from_description(
    tensor_name: &str,
    tensor_description: &Value,
) -> Result<TensorRecord, String> {
    let malformed = || format!("model.safetensors describes tensor {tensor_name} incompletely");
    let type_name = tensor_description
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(malformed)?;
    let Some(element_type) = TensorElementType::from_safetensors_name(type_name) else {
        return Err(format!(
            "tensor {tensor_name} has element type {type_name}; a Laya checkpoint holds F16 or F32"
        ));
    };
    let shape_values = tensor_description
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(malformed)?;
    let mut shape = Vec::with_capacity(shape_values.len());
    for dimension in shape_values {
        shape.push(dimension.as_u64().ok_or_else(malformed)?);
    }
    let data_offsets = tensor_description
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(malformed)?;
    let data_start = data_offsets
        .first()
        .and_then(Value::as_u64)
        .ok_or_else(malformed)?;
    let data_end = data_offsets
        .get(1)
        .and_then(Value::as_u64)
        .ok_or_else(malformed)?;
    let tensor_record = TensorRecord {
        tensor_name: tensor_name.to_owned(),
        element_type,
        shape,
        data_start,
        byte_count: data_end.saturating_sub(data_start),
    };
    if tensor_record.byte_count != tensor_record.element_count() * element_type.bytes_per_element()
    {
        return Err(format!(
            "tensor {tensor_name} spans {} bytes, which its shape and type do not explain",
            tensor_record.byte_count
        ));
    }
    Ok(tensor_record)
}

/// One tensor the conversion will write, known before any byte is produced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PlannedTensor {
    pub(crate) tensor_name: String,
    pub(crate) element_type: TensorElementType,
    pub(crate) shape: Vec<u64>,
}

impl PlannedTensor {
    pub(crate) fn byte_count(&self) -> u64 {
        self.shape.iter().product::<u64>() * self.element_type.bytes_per_element()
    }
}

/// The complete header of an output file whose tensors are written in the
/// order given: the length prefix, the JSON, and its alignment padding.
pub(crate) fn output_header_bytes(planned_tensors: &[PlannedTensor]) -> Vec<u8> {
    let mut header_members = Map::new();
    let mut format_metadata = Map::new();
    // MLX reads this to know the weights are already in its own layout.
    format_metadata.insert("format".to_owned(), Value::from("mlx"));
    header_members.insert("__metadata__".to_owned(), Value::Object(format_metadata));
    let mut next_data_start = 0_u64;
    for planned_tensor in planned_tensors {
        let data_end = next_data_start + planned_tensor.byte_count();
        let mut tensor_description = Map::new();
        tensor_description.insert(
            "dtype".to_owned(),
            Value::from(planned_tensor.element_type.safetensors_name()),
        );
        tensor_description.insert(
            "shape".to_owned(),
            Value::from(planned_tensor.shape.clone()),
        );
        tensor_description.insert(
            "data_offsets".to_owned(),
            Value::from(vec![next_data_start, data_end]),
        );
        header_members.insert(
            planned_tensor.tensor_name.clone(),
            Value::Object(tensor_description),
        );
        next_data_start = data_end;
    }
    let mut header_json_bytes = Value::Object(header_members).to_string().into_bytes();
    while !header_json_bytes
        .len()
        .is_multiple_of(HEADER_ALIGNMENT_BYTES)
    {
        header_json_bytes.push(b' ');
    }
    let mut header_bytes = (header_json_bytes.len() as u64).to_le_bytes().to_vec();
    header_bytes.extend(header_json_bytes);
    header_bytes
}

/// Appends tensors to an output file and refuses one whose size is not the
/// size its header promised, so a wrong plan cannot produce a readable file
/// with shifted weights.
pub(crate) struct OutputCheckpoint<OutputSink: Write> {
    output_sink: OutputSink,
    planned_tensors: Vec<PlannedTensor>,
    written_tensor_count: usize,
}

impl<OutputSink: Write> OutputCheckpoint<OutputSink> {
    pub(crate) fn begin(
        mut output_sink: OutputSink,
        planned_tensors: Vec<PlannedTensor>,
    ) -> Result<Self, String> {
        output_sink
            .write_all(&output_header_bytes(&planned_tensors))
            .map_err(|write_error| format!("could not write the output header: {write_error}"))?;
        Ok(Self {
            output_sink,
            planned_tensors,
            written_tensor_count: 0,
        })
    }

    pub(crate) fn append_tensor(
        &mut self,
        tensor_name: &str,
        tensor_bytes: &[u8],
    ) -> Result<(), String> {
        let Some(planned_tensor) = self.planned_tensors.get(self.written_tensor_count) else {
            return Err(format!("tensor {tensor_name} was not planned"));
        };
        if planned_tensor.tensor_name != tensor_name
            || planned_tensor.byte_count() != tensor_bytes.len() as u64
        {
            return Err(format!(
                "tensor {tensor_name} ({} bytes) does not match the planned {} ({} bytes)",
                tensor_bytes.len(),
                planned_tensor.tensor_name,
                planned_tensor.byte_count()
            ));
        }
        self.output_sink
            .write_all(tensor_bytes)
            .map_err(|write_error| {
                format!("could not write tensor {tensor_name}: {write_error}")
            })?;
        self.written_tensor_count += 1;
        Ok(())
    }

    /// Flushes the file, failing when a planned tensor was never written.
    pub(crate) fn finish(mut self) -> Result<(), String> {
        if self.written_tensor_count != self.planned_tensors.len() {
            return Err(format!(
                "only {} of {} planned tensors were written",
                self.written_tensor_count,
                self.planned_tensors.len()
            ));
        }
        self.output_sink
            .flush()
            .map_err(|flush_error| format!("could not finish the output file: {flush_error}"))
    }
}

#[cfg(test)]
#[path = "safetensors_file_tests.rs"]
mod tests;
