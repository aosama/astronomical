//! Loading entry points and lazy result owners for the custom readers.
//!
//! # The two loading paths
//!
//! [`MlxSafetensors::load`] reads one whole file through the retained
//! descriptor reader; [`load_safetensors_from_bounded_ranges`] reads a
//! virtual payload assembled by the bounded multi-range reader. Both bottom
//! out in the same MLX-C call (`mlx_load_safetensors_reader`), which
//! populates a tensor map and a metadata map by driving the caller's vtable.
//!
//! # Laziness is the design constraint
//!
//! MLX materializes safetensors arrays lazily: a tensor may first be read
//! from the vtable when it is evaluated, not when it is loaded. That is why
//! [`MlxSafetensors`] and [`SafetensorsLoadResult`] keep their reader owners
//! alive as fields, and why `tensor()` returns an owned, reference-counted
//! [`MlxArray`] handle rather than a borrow — the handle keeps the underlying
//! reader alive through MLX's own reference counting even after the result
//! object itself is dropped.

use std::{ffi::CString, fs::File, sync::Arc};

use crate::{
    MlxRuntimeError,
    mlx_runtime::check_status,
    safetensors::bounded_range_reader::{
        BoundedMultiRangeReaderState, OwnedBoundedMultiRangeReader,
        OwnedBoundedMultiRangeReaderHolder,
    },
    safetensors::descriptor_file_reader::OwnedFileReader,
    safetensors::name_maps::{OwnedMetadataMap, OwnedTensorMap},
    safetensors::read_metrics::PositionalFileReadMetrics,
};
use mlx_c_rust::{MlxArray, MlxStream, raw};

/// One virtual interval mapping payload offsets to source file reads.
///
/// The bounded reader's interval contract, validated at construction:
///
/// - intervals must be **contiguous in virtual payload space** — each
///   interval's `virtual_payload_offset` equals the sum of all preceding
///   `source_byte_counts`, so the virtual payload has no holes;
/// - source file ranges must **not overlap** — the same source bytes must not
///   be mapped twice;
/// - the intervals must cover exactly `total_payload_bytes`.
///
/// Callers build this table from an expert-page manifest that has already
/// validated tensor geometry; the reader re-validates because it is the last
/// line of defense before bytes reach MLX.
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
///
/// The reader and metadata map are held (not used) so their lifetimes cover
/// the laziness window of the returned arrays — see the module documentation.
pub struct SafetensorsLoadResult {
    tensor_map: OwnedTensorMap,
    _metadata_map: OwnedMetadataMap,
    _reader: OwnedBoundedMultiRangeReaderHolder,
}

impl SafetensorsLoadResult {
    /// Returns an owned reference-counted handle for one named tensor.
    ///
    /// # Errors
    ///
    /// Fails when the name contains an interior NUL byte, when MLX-C rejects
    /// the lookup, or when the map has no entry under that name (reported as
    /// an empty handle). Callers know the expected names from the manifest
    /// that built the intervals, so a miss indicates manifest/builder drift
    /// and must fail rather than return a placeholder.
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
            return Err(super::name_maps::empty_handle_error(
                "look up a safetensors tensor",
            ));
        }
        Ok(tensor)
    }

    /// Returns all tensor names in the loaded safetensors map.
    ///
    /// MLX doesn't expose a direct iteration API for string-to-array maps.
    /// The caller should know the tensor names from the manifest.
    pub fn tensor_names(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Tensor map loaded lazily from one retained read-only weights descriptor.
///
/// Produced by [`MlxSafetensors::load`]; see the module documentation for why
/// the reader is held for the object's lifetime.
#[derive(Debug)]
pub struct MlxSafetensors {
    tensor_map: OwnedTensorMap,
    _metadata_map: OwnedMetadataMap,
    _reader: OwnedFileReader,
}

impl MlxSafetensors {
    /// Loads a whole safetensors file through the retained descriptor reader.
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
    ///
    /// # Errors
    ///
    /// Same contract as [`SafetensorsLoadResult::tensor`]: interior-NUL
    /// names, native lookup failures, and missing entries all fail loudly.
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
            return Err(super::name_maps::empty_handle_error(
                "look up a safetensors tensor",
            ));
        }
        Ok(tensor)
    }
}
