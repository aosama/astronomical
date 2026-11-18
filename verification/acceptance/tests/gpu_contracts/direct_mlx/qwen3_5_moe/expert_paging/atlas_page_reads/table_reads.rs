//! Reads and writes whose destination is a layer's table: the copy the
//! `astro_expert_cache` makes today, and reads that land straight in the
//! table's own storage, through threads started for them or through the
//! native record reader.

use std::time::{Duration, Instant};

use astronomical_expert_record_reader::{ExpertRecordReader, RecordRead};
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::atlas_conversion::AtlasTestFile;
use super::cold_page_plan::PlannedPage;
use super::macos_file_cache_bypassing_io::{open_bypassing_macos_file_cache, scatter_read_record};
use super::page_read_shapes::{TimedPage, mlx_dtype, page_tensor_shape};

/// One table per tensor, sized for a whole layer and already in memory.
pub(super) fn empty_tables(runtime: &MlxRuntime, atlas: &AtlasTestFile) -> Vec<MlxArray> {
    let mut tables = Vec::with_capacity(atlas.slots.len());
    for slot in &atlas.slots {
        let table_shape = page_tensor_shape(atlas.expert_count as i32, &slot.per_expert_shape);
        tables.push(
            runtime
                .zeros(&table_shape, mlx_dtype(slot.dtype))
                .expect("the table should build"),
        );
    }
    let table_references: Vec<&MlxArray> = tables.iter().collect();
    runtime
        .evaluate_arrays(&table_references)
        .expect("the tables should materialize");
    tables
}

/// Writes every row of the page into consecutive table slots as the
/// `astro_expert_cache` does: one row at a time, each write taking over the
/// table's buffer so only the row is copied.
pub(super) fn copy_page_into_tables(
    runtime: &MlxRuntime,
    tables: &mut [MlxArray],
    page: &TimedPage,
    atlas: &AtlasTestFile,
    first_table_slot: i32,
) -> Duration {
    let started_at = Instant::now();
    for (slot, table) in atlas.slots.iter().zip(tables.iter_mut()) {
        let page_tensor = &page.tensors_by_suffix[&slot.tensor_name_suffix];
        let table_shape = table.shape();
        let unit_strides = vec![1_i32; table_shape.len()];
        for page_row in 0..page_tensor.shape()[0] {
            let mut row_starts = vec![0_i32; table_shape.len()];
            let mut row_stops = table_shape.clone();
            row_starts[0] = page_row;
            row_stops[0] = page_row + 1;
            let expert_row = runtime
                .slice(page_tensor, &row_starts, &row_stops, &unit_strides)
                .expect("the row slice should build");
            row_starts[0] = first_table_slot + page_row;
            row_stops[0] = first_table_slot + page_row + 1;
            *table = runtime
                .slice_update(table, &expert_row, &row_starts, &row_stops, &unit_strides)
                .expect("the slot write should build");
        }
    }
    let table_references: Vec<&MlxArray> = tables.iter().collect();
    runtime
        .evaluate_arrays(&table_references)
        .expect("the slot writes should evaluate");
    started_at.elapsed()
}

/// The first byte of an evaluated table's storage, writable from the host.
///
/// MLX arrays on Apple silicon live in memory the processor and the graphics
/// processor share, so bytes written here are the bytes a later kernel reads.
/// MLX itself treats an array as immutable, so the caller must write only
/// while no graph that reads the table is pending.
pub(super) fn table_storage(table: &mut MlxArray) -> *mut u8 {
    // SAFETY: the table is populated and evaluated; MLX returns the address
    // of its row-contiguous storage, which lives as long as the array.
    let storage = unsafe { mlx_bridge::raw::mlx_array_data_uint8(*table.raw_mut()) };
    assert!(!storage.is_null(), "an evaluated table has storage");
    storage.cast_mut()
}

/// One `preadv` per record whose destinations are slots
/// `first_table_slot..` of the tables' own storage, a thread per record.
pub(super) fn read_records_into_tables(
    atlas: &AtlasTestFile,
    atlas_file: &std::fs::File,
    planned_page: &PlannedPage,
    tables: &mut [MlxArray],
    first_table_slot: usize,
) -> Duration {
    let mut table_storages = Vec::with_capacity(tables.len());
    for table in tables.iter_mut() {
        table_storages.push(table_storage(table));
    }
    let mut destinations_by_expert: Vec<Vec<&mut [u8]>> =
        Vec::with_capacity(planned_page.expert_ids.len());
    for page_slot in 0..planned_page.expert_ids.len() {
        let mut expert_destinations = Vec::with_capacity(atlas.slots.len());
        for (slot, storage) in atlas.slots.iter().zip(&table_storages) {
            let slot_byte_count = slot.bytes_per_expert as usize;
            // SAFETY: slot `first_table_slot + page_slot` of each table lies
            // inside that table's storage, no two page slots overlap, and no
            // pending graph reads the tables while the read runs.
            expert_destinations.push(unsafe {
                std::slice::from_raw_parts_mut(
                    storage.add((first_table_slot + page_slot) * slot_byte_count),
                    slot_byte_count,
                )
            });
        }
        destinations_by_expert.push(expert_destinations);
    }
    let atlas_layer_position = planned_page.atlas_layer_position;
    let started_at = Instant::now();
    std::thread::scope(|scope| {
        for (expert_id, expert_destinations) in planned_page
            .expert_ids
            .iter()
            .zip(destinations_by_expert.iter_mut())
        {
            let record_offset = atlas.record_offset(atlas_layer_position, *expert_id);
            scope
                .spawn(move || scatter_read_record(atlas_file, record_offset, expert_destinations));
        }
    });
    started_at.elapsed()
}

/// Helper threads of the native reader in this probe: with the caller, one
/// thread for each of the ten half-record reads a five-expert page makes.
pub(super) const NATIVE_READER_HELPER_THREAD_COUNT: u32 = 9;

/// The slot index at which a record is cut in two for overlapping reads: the
/// first slot whose start is at or past the record's middle.
fn half_record_slot_index(atlas: &AtlasTestFile) -> usize {
    let mut split_slot_index = atlas.slots.len();
    for (slot_index, slot) in atlas.slots.iter().enumerate() {
        if slot.offset_in_record * 2 >= atlas.record_byte_count {
            split_slot_index = slot_index;
            break;
        }
    }
    split_slot_index
}

/// The same read into the tables' storage through the native reader, each
/// record as `reads_per_record` reads (1 or 2).
pub(super) fn read_records_into_tables_with_native_reader(
    native_reader: &ExpertRecordReader,
    atlas: &AtlasTestFile,
    atlas_file: &std::fs::File,
    planned_page: &PlannedPage,
    tables: &mut [MlxArray],
    first_table_slot: usize,
    reads_per_record: usize,
) -> Duration {
    use std::os::fd::AsFd;
    let mut table_storages = Vec::with_capacity(tables.len());
    for table in tables.iter_mut() {
        table_storages.push(table_storage(table));
    }
    let split_slot_index = if reads_per_record == 2 {
        half_record_slot_index(atlas)
    } else {
        atlas.slots.len()
    };
    let mut record_reads = Vec::with_capacity(planned_page.expert_ids.len() * reads_per_record);
    for (page_slot, expert_id) in planned_page.expert_ids.iter().enumerate() {
        let record_offset = atlas.record_offset(planned_page.atlas_layer_position, *expert_id);
        for slot_range in [0..split_slot_index, split_slot_index..atlas.slots.len()] {
            if slot_range.is_empty() {
                continue;
            }
            let mut destinations = Vec::with_capacity(slot_range.len());
            for slot_index in slot_range.clone() {
                let slot_byte_count = atlas.slots[slot_index].bytes_per_expert as usize;
                // SAFETY: as in `read_records_into_tables`: each slot lies
                // inside its table's storage and no two destinations overlap.
                destinations.push(unsafe {
                    std::slice::from_raw_parts_mut(
                        table_storages[slot_index]
                            .add((first_table_slot + page_slot) * slot_byte_count),
                        slot_byte_count,
                    )
                });
            }
            record_reads.push(RecordRead {
                file: atlas_file.as_fd(),
                file_offset: record_offset + atlas.slots[slot_range.start].offset_in_record,
                destinations,
            });
        }
    }
    let started_at = Instant::now();
    native_reader
        .read(&mut record_reads)
        .expect("the native reader should read the page");
    started_at.elapsed()
}

/// Rows `first_table_slot..first_table_slot + row_count` of every table, as
/// lazy slices that share the tables' storage.
pub(super) fn table_rows(
    runtime: &MlxRuntime,
    tables: &[MlxArray],
    first_table_slot: usize,
    row_count: usize,
) -> Vec<MlxArray> {
    let mut rows = Vec::with_capacity(tables.len());
    for table in tables {
        let table_shape = table.shape();
        let mut row_starts = vec![0_i32; table_shape.len()];
        let mut row_stops = table_shape.clone();
        row_starts[0] = first_table_slot as i32;
        row_stops[0] = (first_table_slot + row_count) as i32;
        rows.push(
            runtime
                .slice(
                    table,
                    &row_starts,
                    &row_stops,
                    &vec![1_i32; table_shape.len()],
                )
                .expect("the row slice should build"),
        );
    }
    rows
}

/// Reads one page straight into fresh tables and checks the graphics
/// processor sees, in those slots, exactly the bytes the stock page holds.
pub(super) fn assert_tables_read_in_place_match_stock(
    runtime: &MlxRuntime,
    atlas: &AtlasTestFile,
    planned_page: &PlannedPage,
) {
    const FIRST_TABLE_SLOT: usize = 3;
    let mut tables = empty_tables(runtime, atlas);
    let atlas_file = open_bypassing_macos_file_cache(&atlas.path);
    let _ = read_records_into_tables(
        atlas,
        &atlas_file,
        planned_page,
        &mut tables,
        FIRST_TABLE_SLOT,
    );
    let rows_read_in_place = table_rows(
        runtime,
        &tables,
        FIRST_TABLE_SLOT,
        planned_page.expert_ids.len(),
    );
    let stock_page =
        super::page_read_shapes::load_stock_page(runtime, &planned_page.stock_manifest);
    for (slot, rows) in atlas.slots.iter().zip(&rows_read_in_place) {
        let stock_tensor = &stock_page.tensors_by_suffix[&slot.tensor_name_suffix];
        let rows_are_equal = runtime
            .array_equal(stock_tensor, rows, false)
            .expect("array_equal builds")
            .to_vec_bool()
            .expect("the comparison should evaluate");
        assert_eq!(
            rows_are_equal,
            vec![true],
            "{} read straight into a table must equal the stock page",
            slot.tensor_name_suffix
        );
    }
}

/// Most fetches timed at one size: enough for a steady median, few enough
/// that five sizes finish in seconds.
const FETCHES_TIMED_AT_ONE_SIZE: usize = 120;

/// Times the native reader filling table slots with `planned_pages` (all one
/// size) that the `macos_file_cache` does not hold, and prints the median.
pub(super) fn report_native_reader_fetch_time(
    runtime: &MlxRuntime,
    atlas: &AtlasTestFile,
    planned_pages: &[PlannedPage],
    residency_checker: &mut super::cold_page_plan::ResidencyChecker,
) {
    let Some(first_page) = planned_pages.first() else {
        eprintln!("[fetch-size] no pages outside the macos_file_cache at this size");
        return;
    };
    let fetch_expert_count = first_page.expert_ids.len();
    let mut tables = empty_tables(runtime, atlas);
    let atlas_file = open_bypassing_macos_file_cache(&atlas.path);
    let native_reader = ExpertRecordReader::new(NATIVE_READER_HELPER_THREAD_COUNT)
        .expect("the native reader should start");
    let mut fetch_milliseconds = Vec::with_capacity(FETCHES_TIMED_AT_ONE_SIZE);
    for planned_page in planned_pages {
        if fetch_milliseconds.len() == FETCHES_TIMED_AT_ONE_SIZE {
            break;
        }
        if residency_checker.atlas_page_is_in_macos_file_cache(
            atlas,
            planned_page.atlas_layer_position,
            &planned_page.expert_ids,
        ) {
            continue;
        }
        let fetch_elapsed = read_records_into_tables_with_native_reader(
            &native_reader,
            atlas,
            &atlas_file,
            planned_page,
            &mut tables,
            0,
            2,
        );
        fetch_milliseconds.push(fetch_elapsed.as_secs_f64() * 1_000.0);
    }
    if fetch_milliseconds.is_empty() {
        eprintln!(
            "[fetch-size] {fetch_expert_count} experts: no pages outside the macos_file_cache"
        );
        return;
    }
    fetch_milliseconds.sort_by(f64::total_cmp);
    let median_milliseconds = fetch_milliseconds[fetch_milliseconds.len() / 2];
    let fetch_megabytes =
        (atlas.record_byte_count * fetch_expert_count as u64) as f64 / 1_048_576.0;
    eprintln!(
        "[fetch-size] {fetch_expert_count:>2} experts ({fetch_megabytes:5.1} MB): {median_milliseconds:.3} ms a fetch over {} fetches, {:.1} GB/s",
        fetch_milliseconds.len(),
        fetch_megabytes / 1024.0 / (median_milliseconds / 1_000.0),
    );
}
