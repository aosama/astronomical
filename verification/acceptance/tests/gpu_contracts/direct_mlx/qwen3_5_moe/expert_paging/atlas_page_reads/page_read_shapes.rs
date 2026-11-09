//! Loads one routed-expert page into MLX arrays under each read shape and
//! times it from the first call until every array is ready for the expert
//! math.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use astronomical_model_serving::{
    QuantizedExpertPageManifest, SafetensorsDtype, load_quantized_expert_page,
};
use astronomical_runtime_integration::{BoundedReadInterval, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};

use super::atlas_conversion::{AtlasTestFile, tensor_name_suffix};
use super::uncached_file_io::{DIRECT_READ_ALIGNMENT_BYTES, open_uncached, scatter_read_record};

/// The one tensor an atlas page presents to MLX: `[experts, record stride]`
/// bytes, read as one MLX Load whose reader fetches one record per expert.
const ATLAS_PAGE_TENSOR_NAME: &str = "expert_records";

pub(super) struct TimedPage {
    pub(super) page_load_elapsed: Duration,
    /// Tensor suffix (`gate_proj.weight`) to its `[experts, ...]` array.
    pub(super) tensors_by_suffix: HashMap<String, MlxArray>,
}

pub(super) struct TimedRecords {
    pub(super) page_load_elapsed: Duration,
    pub(super) expert_records: MlxArray,
}

/// Today's production path: the bounded reader over the stock shards, one
/// MLX Load per tensor.
pub(super) fn load_stock_page(
    runtime: &MlxRuntime,
    page_manifest: &QuantizedExpertPageManifest,
) -> TimedPage {
    let started_at = Instant::now();
    let loaded_tensors = load_quantized_expert_page(runtime, page_manifest, None)
        .expect("the stock page should load");
    let array_references: Vec<&MlxArray> = loaded_tensors.values().collect();
    runtime
        .evaluate_arrays(&array_references)
        .expect("the stock page should materialize");
    let page_load_elapsed = started_at.elapsed();
    let tensors_by_suffix = loaded_tensors
        .into_iter()
        .map(|(tensor_name, tensor)| (tensor_name_suffix(&tensor_name), tensor))
        .collect();
    TimedPage {
        page_load_elapsed,
        tensors_by_suffix,
    }
}

/// Reads the page's records as one `[experts, stride]` byte array through the
/// same bounded reader production uses.
pub(super) fn load_atlas_records(
    runtime: &MlxRuntime,
    atlas: &AtlasTestFile,
    atlas_layer_position: usize,
    expert_ids: &[usize],
) -> TimedRecords {
    let record_stride = atlas.record_stride_bytes;
    let payload_byte_count = record_stride * expert_ids.len() as u64;
    let synthetic_header = atlas_page_header(expert_ids.len(), record_stride);
    let record_read_intervals = expert_ids
        .iter()
        .enumerate()
        .map(|(page_slot, &expert_id)| BoundedReadInterval {
            virtual_payload_offset: page_slot as u64 * record_stride,
            source_file_offset: atlas.record_offset(atlas_layer_position, expert_id),
            source_byte_count: record_stride as usize,
        })
        .collect();
    let started_at = Instant::now();
    let atlas_page_load = runtime
        .load_safetensors_from_bounded_ranges(
            open_uncached(&atlas.path),
            synthetic_header,
            record_read_intervals,
            payload_byte_count,
            None,
        )
        .expect("the atlas page should load");
    let expert_records = atlas_page_load
        .tensor(ATLAS_PAGE_TENSOR_NAME)
        .expect("the atlas page tensor exists");
    expert_records
        .evaluate()
        .expect("the atlas page should materialize");
    TimedRecords {
        page_load_elapsed: started_at.elapsed(),
        expert_records,
    }
}

/// Records, then the nine `[experts, ...]` tensors the expert math consumes:
/// each is a byte slice of every record, viewed as its dtype and made
/// contiguous by MLX on the graphics processor.
pub(super) fn load_atlas_page_split_on_gpu(
    runtime: &MlxRuntime,
    atlas: &AtlasTestFile,
    atlas_layer_position: usize,
    expert_ids: &[usize],
) -> TimedPage {
    let started_at = Instant::now();
    let expert_records =
        load_atlas_records(runtime, atlas, atlas_layer_position, expert_ids).expert_records;
    let page_expert_count = expert_ids.len() as i32;
    let mut tensors_by_suffix = HashMap::with_capacity(atlas.slots.len());
    for slot in &atlas.slots {
        let slot_start = slot.offset_in_record as i32;
        let slot_end = (slot.offset_in_record + slot.bytes_per_expert) as i32;
        let slot_bytes = runtime
            .slice(
                &expert_records,
                &[0, slot_start],
                &[page_expert_count, slot_end],
                &[1, 1],
            )
            .expect("the slot slice should build");
        let slot_values = runtime
            .view(&slot_bytes, mlx_dtype(slot.dtype))
            .expect("the slot view should build");
        let slot_tensor = runtime
            .reshape(
                &slot_values,
                &page_tensor_shape(page_expert_count, &slot.per_expert_shape),
            )
            .expect("the slot reshape should build");
        let contiguous_tensor = runtime
            .copy(&slot_tensor)
            .expect("the slot copy should build");
        tensors_by_suffix.insert(slot.tensor_name_suffix.clone(), contiguous_tensor);
    }
    let array_references: Vec<&MlxArray> = tensors_by_suffix.values().collect();
    runtime
        .evaluate_arrays(&array_references)
        .expect("the atlas tensors should materialize");
    TimedPage {
        page_load_elapsed: started_at.elapsed(),
        tensors_by_suffix,
    }
}

/// One `preadv` per expert scatters its record straight into the nine tensor
/// buffers, which MLX then wraps as Metal buffers.
///
/// This is the shape the atlas exists for: one contiguous, direct SSD read
/// per expert, no file-cache copy, no graphics-processor split. The buffers
/// are `Vec<u8>` because `MlxArray::from_managed_buffer` takes ownership of a
/// `Vec` and frees it with byte alignment, so a 16 KiB-aligned custom
/// allocation could not be handed over; large `Vec` allocations come back
/// page aligned from the macOS allocator, which is asserted rather than
/// assumed. MLX wraps page-aligned memory as a Metal buffer without copying;
/// if Metal ever refused the pointer MLX would copy, so this timing is an
/// upper bound on the zero-copy cost. The timing includes the buffers'
/// first-touch page faults and the per-expert thread starts.
pub(super) fn load_atlas_page_by_scatter_reads(
    atlas: &AtlasTestFile,
    atlas_layer_position: usize,
    expert_ids: &[usize],
) -> TimedPage {
    for slot in &atlas.slots {
        assert_eq!(
            slot.bytes_per_expert % DIRECT_READ_ALIGNMENT_BYTES,
            0,
            "{} keeps scatter targets page aligned",
            slot.tensor_name_suffix
        );
    }
    let started_at = Instant::now();
    let mut tensor_buffers: Vec<Vec<u8>> = atlas
        .slots
        .iter()
        .map(|slot| vec![0u8; slot.bytes_per_expert as usize * expert_ids.len()])
        .collect();
    for tensor_buffer in &tensor_buffers {
        assert_eq!(
            tensor_buffer.as_ptr() as u64 % DIRECT_READ_ALIGNMENT_BYTES,
            0,
            "large allocations are page aligned on macOS"
        );
    }
    let atlas_file = open_uncached(&atlas.path);
    let mut slot_destinations_by_expert =
        slot_destinations_by_page_expert(atlas, &mut tensor_buffers, expert_ids.len());
    std::thread::scope(|scope| {
        for (&expert_id, slot_destinations) in expert_ids
            .iter()
            .zip(slot_destinations_by_expert.iter_mut())
        {
            let record_offset = atlas.record_offset(atlas_layer_position, expert_id);
            let atlas_file = &atlas_file;
            scope.spawn(move || scatter_read_record(atlas_file, record_offset, slot_destinations));
        }
    });
    drop(slot_destinations_by_expert);
    let tensors_by_suffix = wrap_tensor_buffers(atlas, tensor_buffers, expert_ids.len());
    TimedPage {
        page_load_elapsed: started_at.elapsed(),
        tensors_by_suffix,
    }
}

/// For each page expert, its nine destination slices (one per tensor), in
/// record order. `chunks_exact_mut` proves the slices are disjoint.
fn slot_destinations_by_page_expert<'buffers>(
    atlas: &AtlasTestFile,
    tensor_buffers: &'buffers mut [Vec<u8>],
    page_expert_count: usize,
) -> Vec<Vec<&'buffers mut [u8]>> {
    let mut slot_destinations_by_expert: Vec<Vec<&mut [u8]>> = (0..page_expert_count)
        .map(|_| Vec::with_capacity(atlas.slots.len()))
        .collect();
    for (slot, tensor_buffer) in atlas.slots.iter().zip(tensor_buffers.iter_mut()) {
        for (page_slot, expert_destination) in tensor_buffer
            .chunks_exact_mut(slot.bytes_per_expert as usize)
            .enumerate()
        {
            slot_destinations_by_expert[page_slot].push(expert_destination);
        }
    }
    slot_destinations_by_expert
}

fn wrap_tensor_buffers(
    atlas: &AtlasTestFile,
    tensor_buffers: Vec<Vec<u8>>,
    page_expert_count: usize,
) -> HashMap<String, MlxArray> {
    let mut tensors_by_suffix = HashMap::with_capacity(atlas.slots.len());
    for (slot, tensor_buffer) in atlas.slots.iter().zip(tensor_buffers) {
        let tensor_shape = page_tensor_shape(page_expert_count as i32, &slot.per_expert_shape);
        // SAFETY: the buffer holds exactly experts × slice bytes of `dtype`
        // elements, is moved into the array, and nothing else writes it.
        let tensor = unsafe {
            MlxArray::from_managed_buffer(tensor_buffer, &tensor_shape, mlx_dtype(slot.dtype))
        }
        .expect("the tensor wraps its buffer");
        tensors_by_suffix.insert(slot.tensor_name_suffix.clone(), tensor);
    }
    tensors_by_suffix
}

fn page_tensor_shape(page_expert_count: i32, per_expert_shape: &[usize]) -> Vec<i32> {
    let mut tensor_shape = vec![page_expert_count];
    tensor_shape.extend(per_expert_shape.iter().map(|&dimension| dimension as i32));
    tensor_shape
}

fn atlas_page_header(page_expert_count: usize, record_stride: u64) -> Vec<u8> {
    let header_document = serde_json::json!({
        ATLAS_PAGE_TENSOR_NAME: {
            "dtype": "U8",
            "shape": [page_expert_count, record_stride],
            "data_offsets": [0, page_expert_count as u64 * record_stride],
        }
    });
    let encoded_header = serde_json::to_vec(&header_document).expect("the header serializes");
    let mut header_bytes = Vec::with_capacity(8 + encoded_header.len());
    header_bytes.extend_from_slice(&(encoded_header.len() as u64).to_le_bytes());
    header_bytes.extend_from_slice(&encoded_header);
    header_bytes
}

/// The stored dtypes of quantized expert tensors. The serving crate keeps its
/// own mapping private, so this probe states the few it meets.
fn mlx_dtype(safetensors_dtype: SafetensorsDtype) -> MlxDtype {
    match safetensors_dtype {
        SafetensorsDtype::Uint32 => MlxDtype::UInt32,
        SafetensorsDtype::BFloat16 => MlxDtype::BFloat16,
        SafetensorsDtype::Float16 => MlxDtype::Float16,
        SafetensorsDtype::Float32 => MlxDtype::Float32,
        SafetensorsDtype::Uint8 => MlxDtype::UInt8,
        other => panic!("quantized expert tensors do not use {other}"),
    }
}
