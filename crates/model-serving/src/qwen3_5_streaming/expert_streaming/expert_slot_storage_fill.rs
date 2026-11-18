//! Fills table slots with experts read straight from storage.
//!
//! When a reply token routes to experts a layer's table does not hold, the
//! bytes used to arrive in three steps: MLX loaded them into arrays of their
//! own, the token multiplied from those arrays, and after the forward the
//! rows were copied into the table. Measured on a layer's fetch of five
//! experts that was 0.99 ms to read and 0.45 ms to copy. Here the bytes are
//! read once, by the native record reader, into the table's own storage,
//! which the graphics processor then multiplies from: 0.81 ms, no arrays
//! built, no copy (`cargo measure-fetching-layer-stages`).

use std::collections::HashMap;
use std::fs::File;
use std::os::fd::AsFd;
use std::path::PathBuf;

use astronomical_expert_record_reader::{ExpertRecordReader, RecordRead, open_for_direct_reads};
use astronomical_runtime_integration::{MlxRuntimeError, evaluated_array_storage_mut};
use mlx_bridge::MlxArray;

use super::record_read_plan::{StorageSlice, plan_record_reads};
use crate::expert_paging::QuantizedExpertLayerPlan;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;

/// Helper threads of the record reader. A reply token's fetch on one layer
/// is about five experts, each read as three overlapping parts; with the
/// calling thread that is one thread for each. Measured on the 4-bit 35B-A3B
/// under a 10 GB limit, a reply token took 45.6 ms this way against 46.4 ms
/// with two parts and ten threads.
const RECORD_READER_HELPER_THREAD_COUNT: u32 = 14;

/// How often the reader touches storage between fetches, to keep it awake.
/// A drive left alone for a millisecond answers its next request more
/// slowly, and a reply token's fetches come about a millisecond apart.
/// Measured on a four-expert fetch after an idle millisecond
/// (`cargo measure-expert-fetch-size`): 0.81 ms left alone, 0.68 ms with one
/// 16 KiB page read every 100 microseconds, 0.76 ms every 250 and 0.80 ms
/// every 500, so the drive goes quiet after roughly 100 to 200 microseconds.
/// The cost, measured by the same alias over two seconds of fetches a
/// millisecond apart: about 3,900 pages a second (61 MB a second, under one
/// percent of what the reply's own fetches read) from one thread that mostly
/// waits on the drive, with no measurable rise in processor time. Power was
/// not measured.
const STORAGE_AWAKE_READ_PAUSE: std::time::Duration = std::time::Duration::from_micros(100);

/// How long after the last fetch the reader goes on keeping storage awake: a
/// few tokens' worth, so it runs through a reply and stops when the reply
/// does.
const STORAGE_AWAKE_LINGER: std::time::Duration = std::time::Duration::from_millis(20);

/// Reads experts from a model's files into the storage of its tables.
pub(crate) struct ExpertSlotStorageFiller {
    record_reader: ExpertRecordReader,
    /// Every file an expert tensor lives in, opened once for direct reads.
    source_files: Vec<File>,
    source_file_index_by_path: HashMap<PathBuf, usize>,
}

impl std::fmt::Debug for ExpertSlotStorageFiller {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExpertSlotStorageFiller")
            .field("source_file_count", &self.source_files.len())
            .finish()
    }
}

fn fill_error(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "fill expert table slots from storage",
        description,
    }
}

impl ExpertSlotStorageFiller {
    /// Opens every file the layer plans read experts from.
    pub(crate) fn open(layer_plans: &[QuantizedExpertLayerPlan]) -> Result<Self, MlxRuntimeError> {
        let mut source_files = Vec::new();
        let mut source_file_index_by_path = HashMap::new();
        for layer_plan in layer_plans {
            for tensor_source in &layer_plan.tensor_sources {
                if source_file_index_by_path.contains_key(&tensor_source.source_file) {
                    continue;
                }
                let source_file =
                    open_for_direct_reads(&tensor_source.source_file).map_err(|open_error| {
                        fill_error(format!("an expert file could not be opened: {open_error}"))
                    })?;
                source_file_index_by_path
                    .insert(tensor_source.source_file.clone(), source_files.len());
                source_files.push(source_file);
            }
        }
        let record_reader =
            ExpertRecordReader::new(RECORD_READER_HELPER_THREAD_COUNT).map_err(|start_error| {
                fill_error(format!("the record reader did not start: {start_error}"))
            })?;
        record_reader.keep_storage_awake(STORAGE_AWAKE_READ_PAUSE, STORAGE_AWAKE_LINGER);
        Ok(Self {
            record_reader,
            source_files,
            source_file_index_by_path,
        })
    }

    /// Whether this filler can write `table_weights` from storage: every
    /// projection must be stored as MLX keeps it in the table, which holds
    /// for affine-quantized experts. Other encodings keep the copying path.
    pub(crate) fn can_fill(table_weights: &Qwen3_5PagedExpertWeights) -> bool {
        [
            &table_weights.gate_projection,
            &table_weights.up_projection,
            &table_weights.down_projection,
        ]
        .into_iter()
        .all(|projection| matches!(projection, Qwen3_5AffineWeights::Quantized { .. }))
    }

    /// Reads each `(expert id, slot)` of the layer from storage into that
    /// slot of every table array, and returns how many reads that took. The
    /// caller must hold no unevaluated graph that reads the slots being
    /// written.
    pub(crate) fn fill_slots(
        &self,
        layer_plan: &QuantizedExpertLayerPlan,
        table_weights: &mut Qwen3_5PagedExpertWeights,
        expert_slots: &[(usize, usize)],
    ) -> Result<usize, MlxRuntimeError> {
        if expert_slots.is_empty() {
            return Ok(0);
        }
        let mut table_arrays = table_arrays_by_tensor_name(table_weights)?;
        // One entry per (tensor, expert): where the slice is, and where it goes.
        let mut storage_slices =
            Vec::with_capacity(layer_plan.tensor_sources.len() * expert_slots.len());
        let mut slice_destinations: Vec<Option<&mut [u8]>> =
            Vec::with_capacity(storage_slices.capacity());
        for tensor_source in &layer_plan.tensor_sources {
            let tensor_name = (
                tensor_source.projection_name.as_str(),
                tensor_source.parameter_name.as_str(),
            );
            let table_array = table_arrays.remove(&tensor_name).ok_or_else(|| {
                fill_error(format!(
                    "the table has no array for {}.{}",
                    tensor_name.0, tensor_name.1
                ))
            })?;
            let file_index = *self
                .source_file_index_by_path
                .get(&tensor_source.source_file)
                .ok_or_else(|| fill_error("an expert tensor's file was never opened".to_owned()))?;
            let table_storage = evaluated_array_storage_mut(table_array)?;
            let mut slot_storages = slot_storages(table_storage, tensor_source.bytes_per_expert)?;
            for &(expert_id, slot) in expert_slots {
                let slot_storage = slot_storages
                    .get_mut(slot)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        fill_error(format!("slot {slot} is outside the table or planned twice"))
                    })?;
                storage_slices.push(StorageSlice {
                    file_index,
                    file_offset: tensor_source.tensor_payload_offset
                        + expert_id as u64 * tensor_source.expert_stride_bytes as u64,
                    byte_count: tensor_source.bytes_per_expert,
                });
                slice_destinations.push(Some(slot_storage));
            }
        }
        self.read_slices(&storage_slices, slice_destinations)
    }

    fn read_slices(
        &self,
        storage_slices: &[StorageSlice],
        mut slice_destinations: Vec<Option<&mut [u8]>>,
    ) -> Result<usize, MlxRuntimeError> {
        let planned_reads = plan_record_reads(storage_slices);
        let mut record_reads = Vec::with_capacity(planned_reads.len());
        for planned_read in planned_reads {
            let first_slice = storage_slices[planned_read[0]];
            let mut destinations = Vec::with_capacity(planned_read.len());
            for slice_index in planned_read {
                destinations.push(slice_destinations[slice_index].take().ok_or_else(|| {
                    fill_error("a planned read named one slice twice".to_owned())
                })?);
            }
            record_reads.push(RecordRead {
                file: self.source_files[first_slice.file_index].as_fd(),
                file_offset: first_slice.file_offset,
                destinations,
            });
        }
        self.record_reader
            .read(&mut record_reads)
            .map_err(|read_error| fill_error(read_error.to_string()))?;
        Ok(record_reads.len())
    }
}

/// The table's storage cut into its slots, each takeable once.
fn slot_storages(
    table_storage: &mut [u8],
    bytes_per_expert: usize,
) -> Result<Vec<Option<&mut [u8]>>, MlxRuntimeError> {
    if bytes_per_expert == 0 || !table_storage.len().is_multiple_of(bytes_per_expert) {
        return Err(fill_error(format!(
            "a table of {} bytes does not hold whole experts of {bytes_per_expert} bytes",
            table_storage.len()
        )));
    }
    let mut slot_storages = Vec::with_capacity(table_storage.len() / bytes_per_expert);
    for slot_storage in table_storage.chunks_exact_mut(bytes_per_expert) {
        slot_storages.push(Some(slot_storage));
    }
    Ok(slot_storages)
}

/// The nine table arrays keyed by projection and parameter name, as the
/// layer plan names its tensors.
fn table_arrays_by_tensor_name(
    table_weights: &mut Qwen3_5PagedExpertWeights,
) -> Result<HashMap<(&'static str, &'static str), &mut MlxArray>, MlxRuntimeError> {
    let mut table_arrays = HashMap::with_capacity(9);
    for (projection_name, projection) in [
        ("gate_proj", &mut table_weights.gate_projection),
        ("up_proj", &mut table_weights.up_projection),
        ("down_proj", &mut table_weights.down_projection),
    ] {
        let Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            ..
        } = projection
        else {
            return Err(fill_error(format!(
                "{projection_name} is not stored affine-quantized"
            )));
        };
        table_arrays.insert((projection_name, "weight"), packed_weight);
        table_arrays.insert((projection_name, "scales"), quantization_scales);
        table_arrays.insert((projection_name, "biases"), quantization_biases);
    }
    Ok(table_arrays)
}
