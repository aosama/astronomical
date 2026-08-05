use std::{ffi::CString, fs::File, sync::Arc};

use crate::{
    MlxRuntimeError,
    mlx_array::MlxArray,
    mlx_bounded_safetensors_reader::{
        BoundedMultiRangeReaderState, OwnedBoundedMultiRangeReader,
        OwnedBoundedMultiRangeReaderHolder,
    },
    mlx_descriptor_file_reader::OwnedFileReader,
    mlx_runtime::check_status,
    mlx_stream::MlxStream,
    positional_file_read_metrics::PositionalFileReadMetrics,
    raw,
};

/// One virtual interval mapping payload offsets to source file reads.
#[derive(Clone, Debug)]
pub struct BoundedReadInterval {
    /// Byte offset in the virtual payload where this interval starts.
    pub virtual_payload_offset: u64,
    /// Byte offset in the source file where reading begins.
    pub source_file_offset: u64,
    /// Number of bytes to read from the source file for this interval.
    pub source_byte_count: usize,
}

/// Loads safetensors tensors from bounded multi-range reads on a source file.
///
/// Constructs a synthetic safetensors header from the provided header bytes and
/// maps virtual payload offsets through the interval list to exact byte ranges
/// in the source file. This is the core I/O primitive for expert paging: it
/// reads only the selected expert rows from the safetensors shard, never the
/// full shard file.
///
/// The interval list must be contiguous in virtual payload space (each
/// interval's virtual_payload_offset equals the sum of all preceding
/// source_byte_counts) and must not contain overlapping source file ranges.
pub(crate) fn load_safetensors_from_bounded_ranges(
    source_file: File,
    synthetic_header_bytes: Vec<u8>,
    intervals: Vec<BoundedReadInterval>,
    total_payload_bytes: u64,
    stream: &MlxStream,
    expert_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
) -> Result<SafetensorsLoadResult, MlxRuntimeError> {
    let reader_state = BoundedMultiRangeReaderState::new(
        source_file,
        synthetic_header_bytes,
        intervals,
        total_payload_bytes,
        expert_file_read_metrics,
    )?;
    let reader = OwnedBoundedMultiRangeReader::new(reader_state)?;
    let mut tensor_map = OwnedTensorMap::new()?;
    let mut metadata_map = OwnedMetadataMap::new()?;
    // SAFETY: All output handles are live and uniquely borrowed. The reader
    // owns a valid callback table and remains alive in the returned owner.
    let status = unsafe {
        raw::mlx_load_safetensors_reader(
            &mut tensor_map.0,
            &mut metadata_map.0,
            reader.0,
            stream.raw(),
        )
    };
    check_status(
        status,
        "load safetensors through a bounded multi-range reader",
    )?;
    Ok(SafetensorsLoadResult {
        tensor_map,
        _metadata_map: metadata_map,
        _reader: OwnedBoundedMultiRangeReaderHolder(reader.0),
    })
}

/// Result of loading safetensors through a bounded multi-range reader.
pub struct SafetensorsLoadResult {
    tensor_map: OwnedTensorMap,
    _metadata_map: OwnedMetadataMap,
    _reader: OwnedBoundedMultiRangeReaderHolder,
}

impl SafetensorsLoadResult {
    /// Returns an owned reference-counted handle for one named tensor.
    pub fn tensor(&self, tensor_name: &str) -> Result<MlxArray, MlxRuntimeError> {
        let tensor_name =
            CString::new(tensor_name).map_err(|_| MlxRuntimeError::RuntimeOperation {
                operation: "look up a safetensors tensor",
                description: "tensor name contains an interior null byte".to_owned(),
            })?;
        let mut tensor = MlxArray::empty();
        // SAFETY: The map and output array are live, and the C string remains
        // valid for the duration of this non-retaining lookup call.
        let status = unsafe {
            raw::mlx_map_string_to_array_get(
                tensor.raw_mut(),
                self.tensor_map.0,
                tensor_name.as_ptr(),
            )
        };
        check_status(status, "look up a safetensors tensor")?;
        if tensor.is_empty() {
            return Err(empty_handle_error("look up a safetensors tensor"));
        }
        Ok(tensor)
    }

    /// Returns all tensor names in the loaded safetensors map.
    pub fn tensor_names(&self) -> Vec<String> {
        // MLX doesn't expose a direct iteration API for string-to-array maps.
        // The caller should know the tensor names from the manifest.
        Vec::new()
    }
}

/// Tensor map loaded lazily from one retained read-only weights descriptor.
#[derive(Debug)]
pub struct MlxSafetensors {
    tensor_map: OwnedTensorMap,
    _metadata_map: OwnedMetadataMap,
    _reader: OwnedFileReader,
}

impl MlxSafetensors {
    pub(crate) fn load(
        weights_file: File,
        positional_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Result<Self, MlxRuntimeError> {
        let reader = OwnedFileReader::new(weights_file, positional_file_read_metrics)?;
        let stream = MlxStream::default_cpu()?;
        let mut tensor_map = OwnedTensorMap::new()?;
        let mut metadata_map = OwnedMetadataMap::new()?;
        // SAFETY: All output handles are live and uniquely borrowed. The reader
        // owns a valid callback table and remains alive in the returned owner.
        let status = unsafe {
            raw::mlx_load_safetensors_reader(
                &mut tensor_map.0,
                &mut metadata_map.0,
                reader.0,
                stream.raw(),
            )
        };
        check_status(
            status,
            "load safetensors through a retained file descriptor",
        )?;
        Ok(Self {
            tensor_map,
            _metadata_map: metadata_map,
            _reader: reader,
        })
    }

    /// Returns an owned reference-counted handle for one named tensor.
    pub fn tensor(&self, tensor_name: &str) -> Result<MlxArray, MlxRuntimeError> {
        let tensor_name =
            CString::new(tensor_name).map_err(|_| MlxRuntimeError::RuntimeOperation {
                operation: "look up a safetensors tensor",
                description: "tensor name contains an interior null byte".to_owned(),
            })?;
        let mut tensor = MlxArray::empty();
        // SAFETY: The map and output array are live, and the C string remains
        // valid for the duration of this non-retaining lookup call.
        let status = unsafe {
            raw::mlx_map_string_to_array_get(
                tensor.raw_mut(),
                self.tensor_map.0,
                tensor_name.as_ptr(),
            )
        };
        check_status(status, "look up a safetensors tensor")?;
        if tensor.is_empty() {
            return Err(empty_handle_error("look up a safetensors tensor"));
        }
        Ok(tensor)
    }
}

#[derive(Debug)]
struct OwnedTensorMap(raw::mlx_map_string_to_array);

impl OwnedTensorMap {
    fn new() -> Result<Self, MlxRuntimeError> {
        // SAFETY: The returned handle is placed under RAII ownership.
        let raw_map = unsafe { raw::mlx_map_string_to_array_new() };
        if raw_map.ctx.is_null() {
            return Err(empty_handle_error("allocate an MLX tensor map"));
        }
        Ok(Self(raw_map))
    }
}

impl Drop for OwnedTensorMap {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live map exactly once.
        unsafe {
            raw::mlx_map_string_to_array_free(self.0);
        }
    }
}

#[derive(Debug)]
struct OwnedMetadataMap(raw::mlx_map_string_to_string);

impl OwnedMetadataMap {
    fn new() -> Result<Self, MlxRuntimeError> {
        // SAFETY: The returned handle is placed under RAII ownership.
        let raw_map = unsafe { raw::mlx_map_string_to_string_new() };
        if raw_map.ctx.is_null() {
            return Err(empty_handle_error("allocate an MLX metadata map"));
        }
        Ok(Self(raw_map))
    }
}

impl Drop for OwnedMetadataMap {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live map exactly once.
        unsafe {
            raw::mlx_map_string_to_string_free(self.0);
        }
    }
}

fn empty_handle_error(operation: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: "MLX returned an empty handle".to_owned(),
    }
}
