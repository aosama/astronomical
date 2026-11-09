//! Which bytes to read for one dense decoder layer, as bounded-range loads.
//!
//! A streamed dense layer is about thirty tensors scattered through a shard
//! (issue #1985). Each shard that holds part of a layer becomes one bounded
//! load: a synthetic SafeTensors header naming the layer's tensors by their
//! canonical names over a contiguous virtual payload, and the file intervals
//! that fill it. Tensors that sit next to each other in the file merge into one
//! read. This module is that arithmetic only; it reads nothing.

use std::collections::BTreeMap;

use astronomical_runtime_integration::BoundedReadInterval;

/// One tensor of a dense layer: its canonical name, stored form and absolute
/// byte range in its shard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DenseLayerTensorRange {
    pub(crate) canonical_name: String,
    pub(crate) stored_dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) file_start_offset: u64,
    pub(crate) file_end_offset: u64,
}

impl DenseLayerTensorRange {
    fn byte_count(&self) -> u64 {
        self.file_end_offset - self.file_start_offset
    }

    fn file_start_offset_key(&self) -> u64 {
        self.file_start_offset
    }
}

/// One bounded load of a layer's tensors from one shard.
#[derive(Clone, Debug)]
pub(crate) struct DenseLayerShardRead {
    pub(crate) shard_position: usize,
    /// The canonical names the synthetic header declares; the loaded map
    /// cannot list its own entries, so readers look these up by name.
    pub(crate) tensor_names: Vec<String>,
    pub(crate) synthetic_header_bytes: Vec<u8>,
    pub(crate) read_intervals: Vec<BoundedReadInterval>,
    pub(crate) payload_bytes: u64,
}

/// Plans one bounded load per shard for a layer's tensors, grouped by the
/// shard position each tensor lives in.
pub(crate) fn dense_layer_shard_reads(
    tensor_ranges_by_shard: BTreeMap<usize, Vec<DenseLayerTensorRange>>,
) -> Vec<DenseLayerShardRead> {
    let mut shard_reads = Vec::with_capacity(tensor_ranges_by_shard.len());
    for (shard_position, mut tensor_ranges) in tensor_ranges_by_shard {
        // File order keeps the intervals ascending, which the bounded reader
        // needs, and lets neighbouring tensors share one read.
        tensor_ranges.sort_by_key(DenseLayerTensorRange::file_start_offset_key);
        let mut header_entries = serde_json::Map::with_capacity(tensor_ranges.len());
        let mut tensor_names = Vec::with_capacity(tensor_ranges.len());
        let mut read_intervals: Vec<BoundedReadInterval> = Vec::new();
        let mut virtual_payload_offset = 0_u64;
        for tensor_range in &tensor_ranges {
            header_entries.insert(
                tensor_range.canonical_name.clone(),
                serde_json::json!({
                    "dtype": tensor_range.stored_dtype,
                    "shape": tensor_range.shape,
                    "data_offsets": [
                        virtual_payload_offset,
                        virtual_payload_offset + tensor_range.byte_count(),
                    ],
                }),
            );
            append_tensor_interval(&mut read_intervals, tensor_range, virtual_payload_offset);
            tensor_names.push(tensor_range.canonical_name.clone());
            virtual_payload_offset += tensor_range.byte_count();
        }
        shard_reads.push(DenseLayerShardRead {
            shard_position,
            tensor_names,
            synthetic_header_bytes: framed_header_bytes(&header_entries),
            read_intervals,
            payload_bytes: virtual_payload_offset,
        });
    }
    shard_reads
}

/// Extends the last interval when this tensor starts where it ends in both
/// the file and the virtual payload; otherwise starts a new interval.
fn append_tensor_interval(
    read_intervals: &mut Vec<BoundedReadInterval>,
    tensor_range: &DenseLayerTensorRange,
    virtual_payload_offset: u64,
) {
    let tensor_byte_count = tensor_range.byte_count() as usize;
    if let Some(last_interval) = read_intervals.last_mut() {
        let last_file_end =
            last_interval.source_file_offset + last_interval.source_byte_count as u64;
        if last_file_end == tensor_range.file_start_offset {
            last_interval.source_byte_count += tensor_byte_count;
            return;
        }
    }
    read_intervals.push(BoundedReadInterval {
        virtual_payload_offset,
        source_file_offset: tensor_range.file_start_offset,
        source_byte_count: tensor_byte_count,
    });
}

/// A SafeTensors header: the little-endian length followed by the JSON.
fn framed_header_bytes(header_entries: &serde_json::Map<String, serde_json::Value>) -> Vec<u8> {
    let encoded_header = serde_json::Value::Object(header_entries.clone())
        .to_string()
        .into_bytes();
    let mut header_bytes = Vec::with_capacity(8 + encoded_header.len());
    header_bytes.extend_from_slice(&(encoded_header.len() as u64).to_le_bytes());
    header_bytes.extend_from_slice(&encoded_header);
    header_bytes
}
