//! Times fetches of expert pages from an Atlas file through the native reader:
//! the rig the fetches read from and into, the background work each fetch
//! condition runs beside them, the timed fetches themselves (pages already in
//! the `macos_file_cache` are skipped), and the median fetch time they report.

use std::time::Duration;

use expert_record_reader::ExpertRecordReader;
use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;

use super::super::cold_page_plan::ResidencyChecker;
use super::{
    AtlasTestFile, EvictionSweep, FETCHES_TIMED_AT_ONE_SIZE, FetchCondition,
    NATIVE_READER_HELPER_THREAD_COUNT, PlannedPage, StorageKeptBusy, empty_tables,
    open_bypassing_macos_file_cache, page_of_neighbouring_records,
    read_records_into_tables_with_native_reader,
};

/// What one timed-fetch run reads from and into. The fields are declared so
/// they drop in the order the reader, the file, then the tables.
pub(super) struct NativeReaderFetchRig {
    pub(super) native_reader: ExpertRecordReader,
    atlas_file: std::fs::File,
    tables: Vec<MlxArray>,
}

impl NativeReaderFetchRig {
    pub(super) fn open(runtime: &MlxRuntime, atlas: &AtlasTestFile) -> Self {
        let tables = empty_tables(runtime, atlas);
        let atlas_file = open_bypassing_macos_file_cache(&atlas.path);
        let native_reader = ExpertRecordReader::new(NATIVE_READER_HELPER_THREAD_COUNT)
            .expect("the native reader should start");
        Self {
            native_reader,
            atlas_file,
            tables,
        }
    }
}

/// How far apart consecutive page slots land in the table under a condition.
fn table_slot_stride_for(fetch_condition: FetchCondition) -> usize {
    match fetch_condition {
        FetchCondition::ScatteredSlots => 13,
        FetchCondition::NeighbouringSlots
        | FetchCondition::NeighbouringRecords
        | FetchCondition::DuringEvictionSweep
        | FetchCondition::AfterAnIdleMillisecond
        | FetchCondition::AfterAnIdleMillisecondWithStorageKeptBusy
        | FetchCondition::AfterAnIdleMillisecondWithReaderKeepingStorageAwake { .. } => 1,
    }
}

/// Starts whatever runs beside the fetches under `fetch_condition`. The
/// caller holds the returned guards for as long as the fetches are timed;
/// dropping a guard stops its thread.
pub(super) fn start_fetch_condition_background_work(
    native_reader: &ExpertRecordReader,
    atlas: &AtlasTestFile,
    fetch_condition: FetchCondition,
) -> (Option<EvictionSweep>, Option<StorageKeptBusy>) {
    let eviction_sweep = matches!(fetch_condition, FetchCondition::DuringEvictionSweep)
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
    let storage_kept_busy = matches!(
        fetch_condition,
        FetchCondition::AfterAnIdleMillisecondWithStorageKeptBusy
    )
    .then(|| StorageKeptBusy::start(&atlas.path));
    (eviction_sweep, storage_kept_busy)
}

/// One timed fetch per page outside the `macos_file_cache`, up to
/// `FETCHES_TIMED_AT_ONE_SIZE`; returns each fetch's milliseconds.
pub(super) fn time_native_reader_fetches(
    atlas: &AtlasTestFile,
    planned_pages: &[PlannedPage],
    residency_checker: &mut ResidencyChecker,
    fetch_condition: FetchCondition,
    fetch_rig: &mut NativeReaderFetchRig,
) -> Vec<f64> {
    let table_slot_stride = table_slot_stride_for(fetch_condition);
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
            &fetch_rig.native_reader,
            atlas,
            &fetch_rig.atlas_file,
            planned_page,
            &mut fetch_rig.tables,
            0,
            table_slot_stride,
            2,
        );
        fetch_milliseconds.push(fetch_elapsed.as_secs_f64() * 1_000.0);
    }
    fetch_milliseconds
}

pub(super) fn print_median_fetch_time(
    atlas: &AtlasTestFile,
    fetch_expert_count: usize,
    fetch_condition: FetchCondition,
    fetch_milliseconds: &mut [f64],
) {
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
