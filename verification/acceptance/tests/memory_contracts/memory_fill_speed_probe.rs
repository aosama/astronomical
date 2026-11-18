//! How fast MLX runs one fixed piece of GPU work at different memory fills.
//!
//! A large sparse model replied several times slower in the last half
//! gigabyte below the memory limit's maximum, with a healthy
//! `mlx_buffer_cache`, the same storage reads and normal system memory
//! pressure. This probe takes Astronomical's engine out of the question: it
//! fills MLX with plain tables up to a target, then times the same step at
//! every fill. A step gathers a few rows and multiplies in each of 48
//! layers, then writes freshly created pages into small tables, as a reply
//! token reads experts and the engine writes newly read ones into a table.
//!
//! What it found when it was written: the limit MLX is given makes no
//! difference, and the step slows once the memory macOS has wired across
//! the whole system passes the kernel's wire limit. The sweep test prints
//! that total beside each fill.
//!
//! MLX fixes its memory limit once per process, so every fill is its own
//! test and is run on its own (`cargo measure-memory-fill-speed`).

use std::time::Instant;

use astronomical_runtime_integration::{
    MlxMemoryLimits, MlxRuntime, maximum_recommended_gpu_working_set_size_bytes,
};
use mlx_bridge::{MlxArray, MlxDtype};

const BYTES_PER_GIGABYTE: f64 = 1_073_741_824.0;
/// One table is 32,768 rows of 4,096 single-precision values: half a
/// gigabyte, about the size of one layer's expert table.
const TABLE_ROW_COUNT: i32 = 32_768;
const HIDDEN_SIZE: i32 = 4_096;
const TABLE_BYTES: usize = 512 * 1024 * 1024;
/// The work mirrors one reply token: a gather of a few rows and a small
/// multiply in each of this many layers, evaluated once.
const LAYERS_PER_STEP: usize = 48;
const ROWS_GATHERED_PER_LAYER: i32 = 8;
/// Each step also writes freshly created pages into small tables, as the
/// engine writes experts it has just read from storage into a layer's
/// `astro_expert_cache` table: twelve pages of 160 rows, about 2.5 MB each,
/// into three tables of 32 such pages.
const PAGES_WRITTEN_PER_STEP: usize = 12;
const PAGE_ROW_COUNT: i32 = 160;
const WRITTEN_TABLE_COUNT: usize = 3;
const WRITTEN_TABLE_PAGE_SLOTS: i32 = 32;
/// Buffers left in the `mlx_buffer_cache` before the measurement, about what
/// a large sparse model's reply keeps there.
const SEEDED_MLX_BUFFER_CACHE_BYTES: usize = 512 * 1024 * 1024;
const WARMUP_STEP_COUNT: usize = 20;
const MEASURED_STEP_COUNT: usize = 150;

/// One fill to measure: the limit MLX is given and how much it then holds.
#[derive(Clone, Copy)]
struct MemoryFill {
    label: &'static str,
    memory_limit_gigabytes: f64,
    held_gigabytes: f64,
}

#[test]
#[ignore = "fills most of the GPU's memory with MLX arrays and times fixed work"]
fn should_time_fixed_work_at_a_35_gb_limit_holding_33_5_gb() {
    measure_fixed_work_speed(MemoryFill {
        label: "limit_35_held_33.5",
        memory_limit_gigabytes: 35.0,
        held_gigabytes: 33.5,
    });
}

#[test]
#[ignore = "fills most of the GPU's memory with MLX arrays and times fixed work"]
fn should_time_fixed_work_at_a_36_gb_limit_holding_34_7_gb() {
    measure_fixed_work_speed(MemoryFill {
        label: "limit_36_held_34.7",
        memory_limit_gigabytes: 36.0,
        held_gigabytes: 34.7,
    });
}

#[test]
#[ignore = "fills most of the GPU's memory with MLX arrays and times fixed work"]
fn should_time_fixed_work_at_a_36_5_gb_limit_holding_35_3_gb() {
    measure_fixed_work_speed(MemoryFill {
        label: "limit_36.5_held_35.3",
        memory_limit_gigabytes: 36.5,
        held_gigabytes: 35.3,
    });
}

/// The same limit as the slow case with the fill of the fast one: it tells
/// whether the limit or the amount held is what slows the work.
#[test]
#[ignore = "fills most of the GPU's memory with MLX arrays and times fixed work"]
fn should_time_fixed_work_at_a_36_5_gb_limit_holding_34_7_gb() {
    measure_fixed_work_speed(MemoryFill {
        label: "limit_36.5_held_34.7",
        memory_limit_gigabytes: 36.5,
        held_gigabytes: 34.7,
    });
}

/// One process, one limit, a rising fill: after each measurement another
/// tenth of a gigabyte is held, so the fill where the work slows shows as a
/// step in one printed column.
#[test]
#[ignore = "fills most of the GPU's memory with MLX arrays and times fixed work"]
fn should_time_fixed_work_as_the_fill_rises_from_34_5_to_36_gb_under_a_36_5_gb_limit() {
    const SWEEP_STEP_COUNT: usize = 60;
    const SWEEP_FILL_INCREMENT_ROWS: i32 = 6_554;
    let memory_fill = MemoryFill {
        label: "sweep_limit_36.5",
        memory_limit_gigabytes: 36.5,
        held_gigabytes: 34.5,
    };
    let runtime = initialize_runtime_as_the_worker_does(memory_fill);
    let tables = fill_with_tables(&runtime, memory_fill);
    let layer_weight = create_layer_weight(&runtime);
    let mut written_tables = create_written_tables(&runtime);
    let host_page = vec![0.5_f32; (PAGE_ROW_COUNT * HIDDEN_SIZE) as usize];
    seed_mlx_buffer_cache(&runtime);
    let mut step_index = 0;
    for sweep_index in 0..16 {
        let measurement_started_at = Instant::now();
        for _sweep_step in 0..SWEEP_STEP_COUNT {
            run_one_step(&runtime, &tables, &layer_weight, step_index);
            write_pages_into_tables(&runtime, &mut written_tables, &host_page, step_index);
            step_index += 1;
        }
        let elapsed_seconds = measurement_started_at.elapsed().as_secs_f64();
        let memory_snapshot = runtime
            .memory_snapshot()
            .expect("the memory snapshot should be readable");
        eprintln!(
            "[memory-fill-speed] status=progress sweep={}/16 system_wired_gb={:.2} held_gb={:.2} mlx_buffer_cache_gb={:.2} steps_per_second={:.1}",
            sweep_index + 1,
            system_wired_gigabytes(),
            memory_snapshot.active_memory_bytes() as f64 / BYTES_PER_GIGABYTE,
            memory_snapshot.mlx_buffer_cache_bytes() as f64 / BYTES_PER_GIGABYTE,
            SWEEP_STEP_COUNT as f64 / elapsed_seconds,
        );
        let added_fill = runtime
            .full(
                &[SWEEP_FILL_INCREMENT_ROWS, HIDDEN_SIZE],
                0.5,
                MlxDtype::Float32,
            )
            .expect("the added fill should be created");
        runtime
            .evaluate_arrays(&[&added_fill])
            .expect("the added fill should be evaluated");
        std::mem::forget(added_fill);
    }
}

fn initialize_runtime_as_the_worker_does(memory_fill: MemoryFill) -> MlxRuntime {
    let memory_limit_bytes = (memory_fill.memory_limit_gigabytes * BYTES_PER_GIGABYTE) as usize;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(memory_limit_bytes, memory_limit_bytes)
            .expect("the probe's memory limits should be valid"),
    )
    .expect("the pinned MLX runtime should initialize");
    // The serving worker claims the whole recommended working set as wired
    // memory at startup; the probe does the same so the fills compare.
    let recommended_working_set_bytes = maximum_recommended_gpu_working_set_size_bytes()
        .expect("Metal should report its recommended working set");
    mlx_bridge::set_wired_limit(recommended_working_set_bytes)
        .expect("MLX should accept the recommended working set as its wired limit");
    runtime
}

fn create_layer_weight(runtime: &MlxRuntime) -> MlxArray {
    let layer_weight = runtime
        .full(
            &[HIDDEN_SIZE, HIDDEN_SIZE],
            1.0 / HIDDEN_SIZE as f32,
            MlxDtype::Float32,
        )
        .expect("the layer weight should be created");
    runtime
        .evaluate_arrays(&[&layer_weight])
        .expect("the layer weight should be evaluated");
    layer_weight
}

fn measure_fixed_work_speed(memory_fill: MemoryFill) {
    let runtime = initialize_runtime_as_the_worker_does(memory_fill);
    let tables = fill_with_tables(&runtime, memory_fill);
    let layer_weight = create_layer_weight(&runtime);
    let mut written_tables = create_written_tables(&runtime);
    let host_page = vec![0.5_f32; (PAGE_ROW_COUNT * HIDDEN_SIZE) as usize];
    seed_mlx_buffer_cache(&runtime);

    for warmup_step_index in 0..WARMUP_STEP_COUNT {
        run_one_step(&runtime, &tables, &layer_weight, warmup_step_index);
        write_pages_into_tables(&runtime, &mut written_tables, &host_page, warmup_step_index);
    }
    let measurement_started_at = Instant::now();
    let mut last_progress_report_at = Instant::now();
    for measured_step_index in 0..MEASURED_STEP_COUNT {
        run_one_step(
            &runtime,
            &tables,
            &layer_weight,
            WARMUP_STEP_COUNT + measured_step_index,
        );
        write_pages_into_tables(
            &runtime,
            &mut written_tables,
            &host_page,
            WARMUP_STEP_COUNT + measured_step_index,
        );
        if last_progress_report_at.elapsed().as_secs() >= 5 {
            report_progress(
                memory_fill.label,
                measured_step_index + 1,
                measurement_started_at,
            );
            last_progress_report_at = Instant::now();
        }
    }
    let elapsed_seconds = measurement_started_at.elapsed().as_secs_f64();
    let memory_snapshot = runtime
        .memory_snapshot()
        .expect("the memory snapshot should be readable");
    eprintln!(
        "[memory-fill-speed] status=success fill={} steps_per_second={:.1} milliseconds_per_step={:.2} held_gb={:.2} mlx_buffer_cache_gb={:.2} system_wired_gb={:.2}",
        memory_fill.label,
        MEASURED_STEP_COUNT as f64 / elapsed_seconds,
        elapsed_seconds * 1_000.0 / MEASURED_STEP_COUNT as f64,
        memory_snapshot.active_memory_bytes() as f64 / BYTES_PER_GIGABYTE,
        memory_snapshot.mlx_buffer_cache_bytes() as f64 / BYTES_PER_GIGABYTE,
        system_wired_gigabytes(),
    );
}

fn report_progress(fill_label: &str, completed_step_count: usize, started_at: Instant) {
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    let remaining_step_count = MEASURED_STEP_COUNT - completed_step_count;
    eprintln!(
        "[memory-fill-speed] status=progress fill={fill_label} steps={completed_step_count}/{MEASURED_STEP_COUNT} percentage={:.0} ETA_seconds={:.0}",
        completed_step_count as f64 * 100.0 / MEASURED_STEP_COUNT as f64,
        elapsed_seconds / completed_step_count as f64 * remaining_step_count as f64,
    );
}

/// Creates and evaluates half-gigabyte tables until MLX holds the target.
fn fill_with_tables(runtime: &MlxRuntime, memory_fill: MemoryFill) -> Vec<MlxArray> {
    let held_target_bytes = (memory_fill.held_gigabytes * BYTES_PER_GIGABYTE) as usize;
    // The written tables are part of what is held.
    let written_tables_bytes = WRITTEN_TABLE_COUNT
        * (WRITTEN_TABLE_PAGE_SLOTS * PAGE_ROW_COUNT * HIDDEN_SIZE) as usize
        * std::mem::size_of::<f32>();
    let table_count = held_target_bytes.saturating_sub(written_tables_bytes) / TABLE_BYTES;
    let fill_started_at = Instant::now();
    let mut tables = Vec::with_capacity(table_count);
    for table_index in 0..table_count {
        let table = runtime
            .full(
                &[TABLE_ROW_COUNT, HIDDEN_SIZE],
                1.0 / (table_index + 1) as f32,
                MlxDtype::Float32,
            )
            .expect("a table should be created");
        runtime
            .evaluate_arrays(&[&table])
            .expect("a table should be evaluated");
        tables.push(table);
        if (table_index + 1) % 16 == 0 {
            eprintln!(
                "[memory-fill-speed] status=progress fill={} tables={}/{table_count} percentage={:.0} elapsed_seconds={:.0}",
                memory_fill.label,
                table_index + 1,
                (table_index + 1) as f64 * 100.0 / table_count as f64,
                fill_started_at.elapsed().as_secs_f64(),
            );
        }
    }
    // A last, smaller table brings the fill to the target exactly.
    let row_bytes = HIDDEN_SIZE as usize * std::mem::size_of::<f32>();
    let remaining_row_count = held_target_bytes
        .saturating_sub(written_tables_bytes)
        .saturating_sub(table_count * TABLE_BYTES)
        / row_bytes;
    if remaining_row_count > 64 {
        let last_table = runtime
            .full(
                &[remaining_row_count as i32, HIDDEN_SIZE],
                0.25,
                MlxDtype::Float32,
            )
            .expect("the last table should be created");
        runtime
            .evaluate_arrays(&[&last_table])
            .expect("the last table should be evaluated");
        // Never read again and never dropped: it exists only to be held
        // for the rest of the process.
        std::mem::forget(last_table);
    }
    tables
}

fn create_written_tables(runtime: &MlxRuntime) -> Vec<MlxArray> {
    let mut written_tables = Vec::with_capacity(WRITTEN_TABLE_COUNT);
    for _written_table_index in 0..WRITTEN_TABLE_COUNT {
        let written_table = runtime
            .zeros(
                &[WRITTEN_TABLE_PAGE_SLOTS * PAGE_ROW_COUNT, HIDDEN_SIZE],
                MlxDtype::Float32,
            )
            .expect("a written table should be created");
        runtime
            .evaluate_arrays(&[&written_table])
            .expect("a written table should be evaluated");
        written_tables.push(written_table);
    }
    written_tables
}

/// Allocates and frees one buffer so the `mlx_buffer_cache` holds it.
fn seed_mlx_buffer_cache(runtime: &MlxRuntime) {
    let seed_row_count =
        SEEDED_MLX_BUFFER_CACHE_BYTES / (HIDDEN_SIZE as usize * std::mem::size_of::<f32>());
    let seed = runtime
        .zeros(&[seed_row_count as i32, HIDDEN_SIZE], MlxDtype::Float32)
        .expect("the seed buffer should be created");
    runtime
        .evaluate_arrays(&[&seed])
        .expect("the seed buffer should be evaluated");
}

/// Creates pages from host memory and writes each into a slot of a table,
/// replacing the table with the updated one, then evaluates the tables.
fn write_pages_into_tables(
    runtime: &MlxRuntime,
    written_tables: &mut [MlxArray],
    host_page: &[f32],
    step_index: usize,
) {
    for page_index in 0..PAGES_WRITTEN_PER_STEP {
        let page = runtime
            .array_from_f32(host_page, &[PAGE_ROW_COUNT, HIDDEN_SIZE])
            .expect("a page should be created from host memory");
        let written_table_index = page_index % written_tables.len();
        let page_slot = ((step_index * 5 + page_index * 3) as i32) % WRITTEN_TABLE_PAGE_SLOTS;
        let updated_table = runtime
            .slice_update(
                &written_tables[written_table_index],
                &page,
                &[page_slot * PAGE_ROW_COUNT, 0],
                &[(page_slot + 1) * PAGE_ROW_COUNT, HIDDEN_SIZE],
                &[1, 1],
            )
            .expect("the page should be written into its slot");
        written_tables[written_table_index] = updated_table;
    }
    let written_table_references: Vec<&MlxArray> = written_tables.iter().collect();
    runtime
        .evaluate_arrays(&written_table_references)
        .expect("the written tables should be evaluated");
}

/// One step: in every layer, gather a few rows from one table, add them to
/// the running state and multiply by the layer weight; evaluate once.
fn run_one_step(
    runtime: &MlxRuntime,
    tables: &[MlxArray],
    layer_weight: &MlxArray,
    step_index: usize,
) {
    let mut state = runtime
        .full(&[1, HIDDEN_SIZE], 1.0, MlxDtype::Float32)
        .expect("the state should be created");
    for layer_index in 0..LAYERS_PER_STEP {
        // Every step reads different rows of different tables, as a reply
        // routes each token to different experts.
        let table = &tables[(layer_index * 7 + step_index) % tables.len()];
        let first_row = ((step_index * 131 + layer_index * 17) as i32) % (TABLE_ROW_COUNT - 64);
        let mut gathered_row_indices = Vec::with_capacity(ROWS_GATHERED_PER_LAYER as usize);
        for gathered_row_offset in 0..ROWS_GATHERED_PER_LAYER {
            gathered_row_indices.push(first_row + gathered_row_offset * 8);
        }
        let row_indices = runtime
            .array_from_i32(&gathered_row_indices, &[ROWS_GATHERED_PER_LAYER])
            .expect("the row indices should be created");
        let gathered_rows = runtime
            .take_axis(table, &row_indices, 0)
            .expect("the rows should be gathered");
        let gathered_sum = runtime
            .sum_axis(&gathered_rows, 0, true)
            .expect("the gathered rows should be summed");
        let mixed_state = runtime
            .add(&state, &gathered_sum)
            .expect("the state should take the gathered rows");
        state = runtime
            .matmul(&mixed_state, layer_weight)
            .expect("the state should be multiplied by the layer weight");
    }
    runtime
        .evaluate_arrays(&[&state])
        .expect("the step should be evaluated");
}

/// Memory macOS has wired across the whole system, from `vm_stat`: MLX's
/// buffers plus everything the kernel and other processes keep wired.
fn system_wired_gigabytes() -> f64 {
    let vm_stat_output = std::process::Command::new("vm_stat")
        .output()
        .expect("vm_stat should run");
    let vm_stat_text = String::from_utf8_lossy(&vm_stat_output.stdout);
    let page_bytes = vm_stat_text
        .split("page size of ")
        .nth(1)
        .and_then(|after_page_size| after_page_size.split(' ').next())
        .and_then(|page_size_text| page_size_text.parse::<f64>().ok())
        .expect("vm_stat should state its page size");
    let wired_page_count = vm_stat_text
        .lines()
        .find(|line| line.starts_with("Pages wired down:"))
        .and_then(|line| line.split(':').nth(1))
        .and_then(|count_text| count_text.trim().trim_end_matches('.').parse::<f64>().ok())
        .expect("vm_stat should report wired pages");
    wired_page_count * page_bytes / BYTES_PER_GIGABYTE
}
