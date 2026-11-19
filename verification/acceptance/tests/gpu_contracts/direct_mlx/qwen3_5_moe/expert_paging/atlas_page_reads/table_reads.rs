//! Reads and writes whose destination is a layer's table: the copy the
//! `astro_expert_cache` makes today, and reads that land straight in the
//! table's own storage, through threads started for them or through the
//! native record reader.

use std::time::{Duration, Instant};

use expert_record_reader::{ExpertRecordReader, RecordRead};
use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;

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
/// record as `reads_per_record` reads (1 or 2). Page slot `n` goes to table
/// slot `first_table_slot + n * table_slot_stride`: a stride of 1 fills
/// neighbouring slots, a larger one scatters the page across the table as a
/// served fetch does.
pub(super) fn read_records_into_tables_with_native_reader(
    native_reader: &ExpertRecordReader,
    atlas: &AtlasTestFile,
    atlas_file: &std::fs::File,
    planned_page: &PlannedPage,
    tables: &mut [MlxArray],
    first_table_slot: usize,
    table_slot_stride: usize,
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
                        table_storages[slot_index].add(
                            (first_table_slot + page_slot * table_slot_stride) * slot_byte_count,
                        ),
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
/// that every size and condition finishes in seconds.
const FETCHES_TIMED_AT_ONE_SIZE: usize = 120;

/// What surrounds a timed fetch besides its size.
#[derive(Clone, Copy, Debug)]
pub(super) enum FetchCondition {
    /// Neighbouring table slots, nothing else touching the file.
    NeighbouringSlots,
    /// Slots thirteen apart, as a table that has been evicting for a while
    /// hands them out.
    ScatteredSlots,
    /// Neighbouring slots while a thread drops the file's pages from the
    /// `macos_file_cache` every 10 ms, as the cold serving probes do to keep
    /// a model's files out of that memory.
    DuringEvictionSweep,
    /// Experts that sit next to one another in the file, so the whole fetch
    /// is one unbroken run of it: what a fetch would be if experts used
    /// together were stored together.
    NeighbouringRecords,
    /// Neighbouring slots, each fetch after a millisecond in which the
    /// reader does nothing: in a served token the forward computes for about
    /// that long between one layer's fetch and the next, so the reader's
    /// threads are asleep when the fetch arrives.
    AfterAnIdleMillisecond,
    /// The same idle millisecond while another thread keeps storage busy
    /// with one 16 KiB direct read after another: if the idle millisecond
    /// costs because storage itself went quiet, this takes the cost away.
    AfterAnIdleMillisecondWithStorageKeptBusy,
    /// The same idle millisecond with the reader itself keeping storage
    /// awake, one page read every so many microseconds.
    AfterAnIdleMillisecondWithReaderKeepingStorageAwake { pause_microseconds: u64 },
}

/// A thread that keeps one file's pages out of the `macos_file_cache` until
/// it is dropped, the way the serving probes' sweep does.
struct EvictionSweep {
    should_stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    sweeping_thread: Option<std::thread::JoinHandle<()>>,
}

impl EvictionSweep {
    fn start(path: &std::path::Path) -> Self {
        use std::os::fd::AsRawFd;
        let file = std::fs::File::open(path).expect("the atlas opens for the sweep");
        let mapping_length = file.metadata().expect("the atlas stats").len() as usize;
        // SAFETY: a fresh read-only shared mapping of a live descriptor; the
        // pages are never read through it.
        let mapping_address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapping_length,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        assert_ne!(
            mapping_address,
            libc::MAP_FAILED,
            "the atlas maps for the sweep"
        );
        let mapping_address = mapping_address as usize;
        let should_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let thread_should_stop = std::sync::Arc::clone(&should_stop);
        let sweeping_thread = std::thread::spawn(move || {
            let _file = file;
            while !thread_should_stop.load(std::sync::atomic::Ordering::Relaxed) {
                // SAFETY: the address and length are exactly the live mapping.
                unsafe {
                    libc::msync(
                        mapping_address as *mut libc::c_void,
                        mapping_length,
                        libc::MS_INVALIDATE,
                    );
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            // SAFETY: unmaps exactly the mapping made above, once.
            unsafe {
                libc::munmap(mapping_address as *mut libc::c_void, mapping_length);
            }
        });
        Self {
            should_stop,
            sweeping_thread: Some(sweeping_thread),
        }
    }
}

/// A thread that reads one 16 KiB page straight from storage, over and over,
/// until it is dropped, so storage never sees a quiet moment.
struct StorageKeptBusy {
    should_stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    reading_thread: Option<std::thread::JoinHandle<()>>,
}

impl StorageKeptBusy {
    fn start(path: &std::path::Path) -> Self {
        use std::os::unix::fs::FileExt;
        let file = open_bypassing_macos_file_cache(path);
        let file_page_count = file.metadata().expect("the atlas stats").len() / 16_384;
        let should_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let thread_should_stop = std::sync::Arc::clone(&should_stop);
        let reading_thread = std::thread::spawn(move || {
            let mut page = super::macos_file_cache_bypassing_io::PageAlignedBuffer::new(16_384);
            let mut page_index = 0_u64;
            while !thread_should_stop.load(std::sync::atomic::Ordering::Relaxed) {
                // A different page each time, so no read is answered from
                // anything but storage.
                page_index = (page_index + 7_919) % file_page_count;
                let _ = file.read_exact_at(page.bytes_mut(), page_index * 16_384);
            }
        });
        Self {
            should_stop,
            reading_thread: Some(reading_thread),
        }
    }
}

impl Drop for StorageKeptBusy {
    fn drop(&mut self) {
        self.should_stop
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(reading_thread) = self.reading_thread.take() {
            let _ = reading_thread.join();
        }
    }
}

impl Drop for EvictionSweep {
    fn drop(&mut self) {
        self.should_stop
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(sweeping_thread) = self.sweeping_thread.take() {
            let _ = sweeping_thread.join();
        }
    }
}

/// A page of as many experts as `planned_page`, starting at its first expert
/// and running on through that expert's neighbours in the file.
fn page_of_neighbouring_records(atlas: &AtlasTestFile, planned_page: &PlannedPage) -> PlannedPage {
    let fetch_expert_count = planned_page.expert_ids.len();
    let first_expert_id = planned_page.expert_ids[0].min(atlas.expert_count - fetch_expert_count);
    PlannedPage {
        atlas_layer_position: planned_page.atlas_layer_position,
        expert_ids: (first_expert_id..first_expert_id + fetch_expert_count).collect(),
        stock_manifest: planned_page.stock_manifest.clone(),
    }
}

/// Times the native reader filling table slots with `planned_pages` (all one
/// size) that the `macos_file_cache` does not hold, and prints the median.
pub(super) fn report_native_reader_fetch_time(
    runtime: &MlxRuntime,
    atlas: &AtlasTestFile,
    planned_pages: &[PlannedPage],
    residency_checker: &mut super::cold_page_plan::ResidencyChecker,
    fetch_condition: FetchCondition,
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
    let table_slot_stride = match fetch_condition {
        FetchCondition::ScatteredSlots => 13,
        FetchCondition::NeighbouringSlots
        | FetchCondition::NeighbouringRecords
        | FetchCondition::DuringEvictionSweep
        | FetchCondition::AfterAnIdleMillisecond
        | FetchCondition::AfterAnIdleMillisecondWithStorageKeptBusy
        | FetchCondition::AfterAnIdleMillisecondWithReaderKeepingStorageAwake { .. } => 1,
    };
    let _eviction_sweep = matches!(fetch_condition, FetchCondition::DuringEvictionSweep)
        .then(|| EvictionSweep::start(&atlas.path));
    if let FetchCondition::AfterAnIdleMillisecondWithReaderKeepingStorageAwake {
        pause_microseconds,
    } = fetch_condition
    {
        native_reader.keep_storage_awake(
            Duration::from_micros(pause_microseconds),
            Duration::from_millis(20),
        );
    }
    let _storage_kept_busy = matches!(
        fetch_condition,
        FetchCondition::AfterAnIdleMillisecondWithStorageKeptBusy
    )
    .then(|| StorageKeptBusy::start(&atlas.path));
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
        if matches!(
            fetch_condition,
            FetchCondition::AfterAnIdleMillisecond
                | FetchCondition::AfterAnIdleMillisecondWithStorageKeptBusy
                | FetchCondition::AfterAnIdleMillisecondWithReaderKeepingStorageAwake { .. }
        ) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let neighbouring_page;
        let planned_page = if matches!(fetch_condition, FetchCondition::NeighbouringRecords) {
            neighbouring_page = page_of_neighbouring_records(atlas, planned_page);
            &neighbouring_page
        } else {
            planned_page
        };
        let fetch_elapsed = read_records_into_tables_with_native_reader(
            &native_reader,
            atlas,
            &atlas_file,
            planned_page,
            &mut tables,
            0,
            table_slot_stride,
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
        "[fetch-size] {fetch_expert_count:>2} experts ({fetch_megabytes:5.1} MB), {fetch_condition:?}: {median_milliseconds:.3} ms a fetch over {} fetches, {:.1} GB/s",
        fetch_milliseconds.len(),
        fetch_megabytes / 1024.0 / (median_milliseconds / 1_000.0),
    );
}

/// What keeping storage awake costs while a reply fetches: the reader is fed
/// one small fetch a millisecond for several seconds, as a streamed reply
/// feeds it, and the pages it reads on the side and the processor time this
/// process spends are reported. Power is not: the system load a portable
/// Mac's battery controller reports without administrator rights was tried
/// and refreshes less than once in 45 seconds, too slowly to tell two runs
/// of a few seconds apart.
pub(super) fn report_cost_of_keeping_storage_awake(
    runtime: &MlxRuntime,
    atlas: &AtlasTestFile,
    planned_page: &PlannedPage,
    pause: Option<Duration>,
) {
    const MEASURED_FETCH_COUNT: u32 = 4_000;
    let mut tables = empty_tables(runtime, atlas);
    let atlas_file = open_bypassing_macos_file_cache(&atlas.path);
    let native_reader = ExpertRecordReader::new(NATIVE_READER_HELPER_THREAD_COUNT)
        .expect("the native reader should start");
    if let Some(pause) = pause {
        native_reader.keep_storage_awake(pause, Duration::from_millis(20));
    }
    let processor_seconds_before = process_processor_seconds();
    let started_at = Instant::now();
    for _ in 0..MEASURED_FETCH_COUNT {
        std::thread::sleep(Duration::from_millis(1));
        let _ = read_records_into_tables_with_native_reader(
            &native_reader,
            atlas,
            &atlas_file,
            planned_page,
            &mut tables,
            0,
            1,
            2,
        );
    }
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    let processor_seconds = process_processor_seconds() - processor_seconds_before;
    let awake_read_count = native_reader.awake_read_count();
    eprintln!(
        "[awake-cost] awake reads {}: {:.0} pages a second ({:.0} MB/s), this process used {:.1}% of one processor core",
        match pause {
            Some(pause) => format!("every {} µs", pause.as_micros()),
            None => "off".to_owned(),
        },
        awake_read_count as f64 / elapsed_seconds,
        awake_read_count as f64 * 16_384.0 / 1_048_576.0 / elapsed_seconds,
        processor_seconds * 100.0 / elapsed_seconds,
    );
}

/// Processor time this process has used, user and system together.
fn process_processor_seconds() -> f64 {
    // SAFETY: getrusage fills a zeroed struct for this process.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: as above; the pointer is to a live, writable struct.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    timeval_seconds(usage.ru_utime) + timeval_seconds(usage.ru_stime)
}

fn timeval_seconds(time: libc::timeval) -> f64 {
    time.tv_sec as f64 + f64::from(time.tv_usec) / 1e6
}
