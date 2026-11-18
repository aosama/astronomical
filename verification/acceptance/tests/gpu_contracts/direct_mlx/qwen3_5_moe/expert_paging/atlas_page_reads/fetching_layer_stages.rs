//! One fetching layer's work, stage by stage, so each stage's cost is a
//! measurement instead of what is left over after subtracting the others.
//!
//! When a reply token routes to experts a layer does not hold, serving stops
//! the graphics processor to learn the route, reads the missing experts,
//! hands them to the multiply, copies them into the layer's table, and goes
//! on. Timed as one reply those stages cannot be told apart, because MLX
//! reads lazily inside its own graph evaluation. Here each runs alone, on
//! pages the `macos_file_cache` does not hold:
//!
//! - **Today's read**: one MLX Load per tensor through the bounded reader.
//! - **A read of our own**: one `preadv` per record straight into memory that
//!   already exists, with no MLX involved, on one thread and on one thread
//!   per record.
//! - **The hand-off**: a graphics-processor pass over bytes we read into our
//!   own memory, against the same pass over bytes MLX read into its memory.
//!   The difference is what wrapping and first use of our memory costs.
//! - **A read straight into a table**: one `preadv` per record whose
//!   destinations are the slots of tables MLX already owns, so the bytes land
//!   once, where they are multiplied and kept, with no hand-off and no copy.
//! - **The table copy**: writing the page's rows into table slots, the second
//!   pass over the bytes that keeping an expert costs today.
//! - **The stop**: a chain of small layers evaluated once, against the same
//!   chain stopped after every layer to bring eight indices to the host.

use std::time::{Duration, Instant};

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use super::atlas_conversion::AtlasTestFile;
use super::cold_page_plan::{PlannedPage, ResidencyChecker};
use super::macos_file_cache_bypassing_io::{open_bypassing_macos_file_cache, scatter_read_record};
use super::page_read_shapes::{
    TimedPage, load_atlas_page_as_nine_tensors, mlx_dtype, page_tensor_shape,
};
use super::progress_report::ProgressReporter;

/// The stage sets that each need a page the `macos_file_cache` does not hold.
#[derive(Clone, Copy)]
enum ColdStageSet {
    TodaysReadThenTableCopy,
    OwnReadOnOneThread,
    OwnReadOnAThreadPerRecordThenHandOff,
    OwnReadIntoMlxTables,
}

const COLD_STAGE_SETS: [ColdStageSet; 4] = [
    ColdStageSet::TodaysReadThenTableCopy,
    ColdStageSet::OwnReadOnOneThread,
    ColdStageSet::OwnReadOnAThreadPerRecordThenHandOff,
    ColdStageSet::OwnReadIntoMlxTables,
];

/// Milliseconds of every sample of one stage.
#[derive(Default)]
pub(super) struct StageSamples {
    todays_read: Vec<f64>,
    pass_over_mlx_read_bytes: Vec<f64>,
    table_copy: Vec<f64>,
    own_read_on_one_thread: Vec<f64>,
    own_read_on_a_thread_per_record: Vec<f64>,
    pass_over_own_bytes: Vec<f64>,
    own_read_into_mlx_tables: Vec<f64>,
    pass_over_table_rows_read_in_place: Vec<f64>,
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

/// Page-aligned, already touched memory for each of a page's nine tensors:
/// the stand-in for a table slot that exists before the read begins.
fn preallocated_tensor_buffers(atlas: &AtlasTestFile, page_expert_count: usize) -> Vec<Vec<u8>> {
    let mut tensor_buffers = Vec::with_capacity(atlas.slots.len());
    for slot in &atlas.slots {
        // Writing a nonzero byte forces every page to be backed before the
        // read is timed; a zeroed allocation is only promised, not backed.
        tensor_buffers.push(vec![
            1_u8;
            slot.bytes_per_expert as usize * page_expert_count
        ]);
    }
    tensor_buffers
}

/// For each page expert, its destination slice in every tensor buffer, in
/// record order.
fn record_destinations<'buffers>(
    atlas: &AtlasTestFile,
    tensor_buffers: &'buffers mut [Vec<u8>],
    page_expert_count: usize,
) -> Vec<Vec<&'buffers mut [u8]>> {
    let mut destinations_by_expert: Vec<Vec<&mut [u8]>> = Vec::with_capacity(page_expert_count);
    for _ in 0..page_expert_count {
        destinations_by_expert.push(Vec::with_capacity(atlas.slots.len()));
    }
    for (slot, tensor_buffer) in atlas.slots.iter().zip(tensor_buffers.iter_mut()) {
        for (page_slot, expert_destination) in tensor_buffer
            .chunks_exact_mut(slot.bytes_per_expert as usize)
            .enumerate()
        {
            destinations_by_expert[page_slot].push(expert_destination);
        }
    }
    destinations_by_expert
}

/// One `preadv` per record on the calling thread.
fn read_records_on_one_thread(
    atlas: &AtlasTestFile,
    atlas_file: &std::fs::File,
    planned_page: &PlannedPage,
    tensor_buffers: &mut [Vec<u8>],
) -> Duration {
    let mut destinations_by_expert =
        record_destinations(atlas, tensor_buffers, planned_page.expert_ids.len());
    let started_at = Instant::now();
    for (expert_id, expert_destinations) in planned_page
        .expert_ids
        .iter()
        .zip(destinations_by_expert.iter_mut())
    {
        scatter_read_record(
            atlas_file,
            atlas.record_offset(planned_page.atlas_layer_position, *expert_id),
            expert_destinations,
        );
    }
    started_at.elapsed()
}

/// One `preadv` per record, each on a thread started for it. The thread
/// starts are inside the time: a long-lived pool would not pay them.
fn read_records_on_a_thread_each(
    atlas: &AtlasTestFile,
    atlas_file: &std::fs::File,
    planned_page: &PlannedPage,
    tensor_buffers: &mut [Vec<u8>],
) -> Duration {
    let mut destinations_by_expert =
        record_destinations(atlas, tensor_buffers, planned_page.expert_ids.len());
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

/// A graphics-processor pass that reads every byte of every tensor once.
fn pass_over_every_byte(runtime: &MlxRuntime, tensors: &[MlxArray]) -> Duration {
    let started_at = Instant::now();
    let mut tensor_sums = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        tensor_sums.push(runtime.sum(tensor, false).expect("the sum should build"));
    }
    let sum_references: Vec<&MlxArray> = tensor_sums.iter().collect();
    runtime
        .evaluate_arrays(&sum_references)
        .expect("the pass should evaluate");
    started_at.elapsed()
}

/// Our own memory wrapped as MLX arrays, one per tensor, without a copy.
fn wrap_own_buffers(
    atlas: &AtlasTestFile,
    tensor_buffers: Vec<Vec<u8>>,
    page_expert_count: usize,
) -> Vec<MlxArray> {
    let mut tensors = Vec::with_capacity(atlas.slots.len());
    for (slot, tensor_buffer) in atlas.slots.iter().zip(tensor_buffers) {
        let tensor_shape = page_tensor_shape(page_expert_count as i32, &slot.per_expert_shape);
        // SAFETY: the buffer holds exactly experts x slice bytes of `dtype`
        // elements, is moved into the array, and nothing else writes it.
        let tensor = unsafe {
            MlxArray::from_managed_buffer(tensor_buffer, &tensor_shape, mlx_dtype(slot.dtype))
        }
        .expect("the tensor wraps its buffer");
        tensors.push(tensor);
    }
    tensors
}

/// One table per tensor, sized for a whole layer and already in memory.
fn empty_tables(runtime: &MlxRuntime, atlas: &AtlasTestFile) -> Vec<MlxArray> {
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
fn copy_page_into_tables(
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
fn table_storage(table: &mut MlxArray) -> *mut u8 {
    // SAFETY: the table is populated and evaluated; MLX returns the address
    // of its row-contiguous storage, which lives as long as the array.
    let storage = unsafe { mlx_bridge::raw::mlx_array_data_uint8(*table.raw_mut()) };
    assert!(!storage.is_null(), "an evaluated table has storage");
    storage.cast_mut()
}

/// One `preadv` per record whose destinations are slots
/// `first_table_slot..` of the tables' own storage, a thread per record.
fn read_records_into_tables(
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

/// Rows `first_table_slot..first_table_slot + row_count` of every table, as
/// lazy slices that share the tables' storage.
fn table_rows(
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

/// What the cold stages share across pages: the open atlas, the tables the
/// table stages write, the next free table slot and the samples so far.
struct ColdStageRun<'probe> {
    runtime: &'probe MlxRuntime,
    atlas: &'probe AtlasTestFile,
    atlas_file: std::fs::File,
    tables: Vec<MlxArray>,
    next_table_slot: usize,
    samples: StageSamples,
}

impl ColdStageRun<'_> {
    /// The first of `page_expert_count` free table slots, starting over at
    /// the table's beginning when the page would run past its end.
    fn claim_table_slots(&mut self, page_expert_count: usize) -> usize {
        if self.next_table_slot + page_expert_count > self.atlas.expert_count {
            self.next_table_slot = 0;
        }
        let first_table_slot = self.next_table_slot;
        self.next_table_slot += page_expert_count;
        first_table_slot
    }

    fn todays_read_then_table_copy(&mut self, planned_page: &PlannedPage) {
        let page = load_atlas_page_as_nine_tensors(
            self.runtime,
            self.atlas,
            planned_page.atlas_layer_position,
            &planned_page.expert_ids,
        );
        self.samples
            .todays_read
            .push(milliseconds(page.page_load_elapsed));
        let mut page_tensors = Vec::with_capacity(self.atlas.slots.len());
        for slot in &self.atlas.slots {
            page_tensors.push(
                page.tensors_by_suffix[&slot.tensor_name_suffix]
                    .retain()
                    .expect("the tensor should retain"),
            );
        }
        self.samples
            .pass_over_mlx_read_bytes
            .push(milliseconds(pass_over_every_byte(
                self.runtime,
                &page_tensors,
            )));
        let first_table_slot = self.claim_table_slots(planned_page.expert_ids.len());
        self.samples
            .table_copy
            .push(milliseconds(copy_page_into_tables(
                self.runtime,
                &mut self.tables,
                &page,
                self.atlas,
                first_table_slot as i32,
            )));
    }

    fn own_read_on_one_thread(&mut self, planned_page: &PlannedPage) {
        let mut tensor_buffers =
            preallocated_tensor_buffers(self.atlas, planned_page.expert_ids.len());
        self.samples
            .own_read_on_one_thread
            .push(milliseconds(read_records_on_one_thread(
                self.atlas,
                &self.atlas_file,
                planned_page,
                &mut tensor_buffers,
            )));
    }

    fn own_read_on_a_thread_per_record_then_hand_off(&mut self, planned_page: &PlannedPage) {
        let page_expert_count = planned_page.expert_ids.len();
        let mut tensor_buffers = preallocated_tensor_buffers(self.atlas, page_expert_count);
        self.samples
            .own_read_on_a_thread_per_record
            .push(milliseconds(read_records_on_a_thread_each(
                self.atlas,
                &self.atlas_file,
                planned_page,
                &mut tensor_buffers,
            )));
        let handed_off_at = Instant::now();
        let own_tensors = wrap_own_buffers(self.atlas, tensor_buffers, page_expert_count);
        let _ = pass_over_every_byte(self.runtime, &own_tensors);
        self.samples
            .pass_over_own_bytes
            .push(milliseconds(handed_off_at.elapsed()));
    }

    fn own_read_into_mlx_tables(&mut self, planned_page: &PlannedPage) {
        let page_expert_count = planned_page.expert_ids.len();
        let first_table_slot = self.claim_table_slots(page_expert_count);
        self.samples
            .own_read_into_mlx_tables
            .push(milliseconds(read_records_into_tables(
                self.atlas,
                &self.atlas_file,
                planned_page,
                &mut self.tables,
                first_table_slot,
            )));
        let rows = table_rows(
            self.runtime,
            &self.tables,
            first_table_slot,
            page_expert_count,
        );
        self.samples
            .pass_over_table_rows_read_in_place
            .push(milliseconds(pass_over_every_byte(self.runtime, &rows)));
    }
}

/// Times every stage that needs a cold page, each on pages of its own.
pub(super) fn measure_cold_stages(
    runtime: &MlxRuntime,
    atlas: &AtlasTestFile,
    planned_pages: &[PlannedPage],
    residency_checker: &mut ResidencyChecker,
) -> StageSamples {
    let mut run = ColdStageRun {
        runtime,
        atlas,
        atlas_file: open_bypassing_macos_file_cache(&atlas.path),
        tables: empty_tables(runtime, atlas),
        next_table_slot: 0,
        samples: StageSamples::default(),
    };
    let mut progress = ProgressReporter::new("fetching-layer-stages", planned_pages.len());
    let measuring_started_at = Instant::now();
    for (planned_page_index, planned_page) in planned_pages.iter().enumerate() {
        progress.record(planned_page_index + 1, measuring_started_at);
        if residency_checker.atlas_page_is_in_macos_file_cache(
            atlas,
            planned_page.atlas_layer_position,
            &planned_page.expert_ids,
        ) {
            continue;
        }
        let layer_round = planned_page_index / atlas.layer_indexes.len();
        match COLD_STAGE_SETS[(planned_page_index + layer_round) % COLD_STAGE_SETS.len()] {
            ColdStageSet::TodaysReadThenTableCopy => run.todays_read_then_table_copy(planned_page),
            ColdStageSet::OwnReadOnOneThread => run.own_read_on_one_thread(planned_page),
            ColdStageSet::OwnReadOnAThreadPerRecordThenHandOff => {
                run.own_read_on_a_thread_per_record_then_hand_off(planned_page);
            }
            ColdStageSet::OwnReadIntoMlxTables => run.own_read_into_mlx_tables(planned_page),
        }
    }
    run.samples
}

/// Layers in the stop measurement: the anchor's reply stops about this often
/// in one token.
const STOP_CHAIN_LAYER_COUNT: usize = 28;
const STOP_CHAIN_WIDTH: i32 = 2048;
const STOP_CHAIN_PASS_COUNT: usize = 40;
const STOP_CHAIN_WARM_UP_PASS_COUNT: usize = 5;
const ROUTED_INDEX_COUNT: i32 = 8;

/// One pass down the chain. With `stops_at_every_layer` each layer's eight
/// highest outputs are brought to the host before the next layer is built,
/// as a route is; without it the chain is evaluated once at the end.
fn stop_chain_pass(
    runtime: &MlxRuntime,
    layer_weights: &[MlxArray],
    first_row: &MlxArray,
    stops_at_every_layer: bool,
) -> Duration {
    let started_at = Instant::now();
    let mut hidden_row = first_row.retain().expect("the row should retain");
    let mut routed_index_count = 0_usize;
    for layer_weight in layer_weights {
        let projected = runtime
            .matmul(&hidden_row, layer_weight)
            .expect("the layer should build");
        hidden_row = runtime
            .tanh(&projected)
            .expect("the activation should build");
        if stops_at_every_layer {
            let routed_indices = runtime
                .argpartition(&hidden_row, -ROUTED_INDEX_COUNT)
                .expect("the route should build");
            routed_index_count += routed_indices
                .to_vec_u32()
                .expect("the route should reach the host")
                .len();
        }
    }
    runtime
        .evaluate_arrays(&[&hidden_row])
        .expect("the chain should evaluate");
    std::hint::black_box(routed_index_count);
    started_at.elapsed()
}

/// Median milliseconds of the chain evaluated once and stopped at every
/// layer, in that order.
pub(super) fn measure_stop_cost(runtime: &MlxRuntime) -> (f64, f64) {
    let random_key = runtime.random_key(2329).expect("a key builds");
    let mut layer_weights = Vec::with_capacity(STOP_CHAIN_LAYER_COUNT);
    for _ in 0..STOP_CHAIN_LAYER_COUNT {
        layer_weights.push(
            runtime
                .random_normal(
                    &[STOP_CHAIN_WIDTH, STOP_CHAIN_WIDTH],
                    MlxDtype::Float32,
                    0.0,
                    0.02,
                    &random_key,
                )
                .expect("a layer weight builds"),
        );
    }
    let first_row = runtime
        .random_normal(
            &[1, STOP_CHAIN_WIDTH],
            MlxDtype::Float32,
            0.0,
            1.0,
            &random_key,
        )
        .expect("the first row builds");
    let mut prepared_arrays: Vec<&MlxArray> = layer_weights.iter().collect();
    prepared_arrays.push(&first_row);
    runtime
        .evaluate_arrays(&prepared_arrays)
        .expect("the chain's inputs materialize");
    let mut evaluated_once = Vec::with_capacity(STOP_CHAIN_PASS_COUNT);
    let mut stopped_at_every_layer = Vec::with_capacity(STOP_CHAIN_PASS_COUNT);
    for pass_index in 0..STOP_CHAIN_WARM_UP_PASS_COUNT + STOP_CHAIN_PASS_COUNT {
        let once = stop_chain_pass(runtime, &layer_weights, &first_row, false);
        let stopped = stop_chain_pass(runtime, &layer_weights, &first_row, true);
        if pass_index >= STOP_CHAIN_WARM_UP_PASS_COUNT {
            evaluated_once.push(milliseconds(once));
            stopped_at_every_layer.push(milliseconds(stopped));
        }
    }
    (
        median(&mut evaluated_once),
        median(&mut stopped_at_every_layer),
    )
}

fn median(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Prints each stage's median and what a token that fetches on
/// `fetching_layer_count` layers would spend on it.
pub(super) fn report_stages(
    samples: &mut StageSamples,
    stop_chain_milliseconds: (f64, f64),
    page_byte_count: u64,
    fetching_layer_count: usize,
) {
    let page_megabytes = page_byte_count as f64 / 1_048_576.0;
    let stages: [(&str, &mut Vec<f64>); 8] = [
        (
            "today's read (one MLX Load per tensor)",
            &mut samples.todays_read,
        ),
        ("own read, one thread", &mut samples.own_read_on_one_thread),
        (
            "own read, a thread started per record",
            &mut samples.own_read_on_a_thread_per_record,
        ),
        (
            "graphics-processor pass over bytes MLX read",
            &mut samples.pass_over_mlx_read_bytes,
        ),
        (
            "wrap and graphics-processor pass over our own bytes",
            &mut samples.pass_over_own_bytes,
        ),
        ("copy of the page into table slots", &mut samples.table_copy),
        (
            "own read straight into MLX's tables",
            &mut samples.own_read_into_mlx_tables,
        ),
        (
            "graphics-processor pass over rows read in place",
            &mut samples.pass_over_table_rows_read_in_place,
        ),
    ];
    for (stage_name, stage_samples) in stages {
        let sample_count = stage_samples.len();
        let median_milliseconds = median(stage_samples);
        eprintln!(
            "[fetching-layer] {stage_name}: {median_milliseconds:.3} ms a layer ({page_megabytes:.1} MB, {sample_count} pages), {:.1} ms a token over {fetching_layer_count} layers",
            median_milliseconds * fetching_layer_count as f64,
        );
    }
    let (evaluated_once, stopped_at_every_layer) = stop_chain_milliseconds;
    let stop_milliseconds =
        (stopped_at_every_layer - evaluated_once) / STOP_CHAIN_LAYER_COUNT as f64;
    eprintln!(
        "[fetching-layer] stopping to bring a route to the host: {stop_milliseconds:.3} ms a stop ({evaluated_once:.2} ms for {STOP_CHAIN_LAYER_COUNT} layers evaluated once, {stopped_at_every_layer:.2} ms stopped at each), {:.1} ms a token over {fetching_layer_count} layers",
        stop_milliseconds * fetching_layer_count as f64,
    );
}
