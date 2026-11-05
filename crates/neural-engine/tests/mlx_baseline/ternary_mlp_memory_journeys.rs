//! Do many real MLP programs of the folded ternary 27B model fit, and does moving
//! between them cost anything (issue #161)?
//!
//! Prefill needs one program per layer, so one layer's numbers cannot answer
//! this: the driver wires each program's weights once it runs, and a switching
//! cost only shows when distinct programs take turns.

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use astronomical_neural_engine::core_ml_projection::CoreMlProjection;
use astronomical_neural_engine::ml_program_mlp_document::StageWeightEncoding;
use astronomical_neural_engine::process_memory::{MemoryGrowth, MemorySnapshot, mebibytes};
use astronomical_neural_engine::timing_summary::TimingSummary;
use serial_test::serial;

use super::concurrent_split_journeys::milliseconds;
use super::ternary_checkpoint::{TernaryCheckpoint, TernaryMlpLayer};
use super::ternary_mlp_journeys::{
    HIDDEN_CHANNELS, LAYER_COUNT, NEURAL_ENGINE_TOKEN_COUNT, ROWS_PER_LOOKUP_TABLE, gate_verdict,
    load_program,
};

const MAXIMUM_GIBIBYTES_PER_LAYER: f64 = 0.4;
const BYTES_PER_GIBIBYTE: f64 = 1_073_741_824.0;
// Spread across the model, so each program's weights and scale spread differ.
const PROGRAM_MEMORY_LAYERS: [usize; 8] = [0, 9, 18, 27, 36, 45, 54, 63];
const ONE_AS_FP16: u16 = 0x3c00;
// Enough to leave the rest of the Mac working when the guard stops the loading.
const MINIMUM_FREE_DISK_GIBIBYTES: f64 = 4.0;
const MINIMUM_FREE_MEMORY_PERCENT: u32 = 15;
const WARM_UP_PROGRAM_CYCLES: usize = 2;
const MEASURED_PROGRAM_CYCLES: usize = 4;
// The verdict divides the growth after the first load by the programs after it.
const MINIMUM_PROGRAMS_FOR_GROWTH: usize = 2;

/// Prefill visits every layer once per chunk, so this cycles through distinct
/// programs to expose any switching cost that one program in a loop would hide.
fn time_program_cycles(programs: &mut [CoreMlProjection]) -> TimingSummary {
    let activations = vec![ONE_AS_FP16; HIDDEN_CHANNELS * NEURAL_ENGINE_TOKEN_COUNT];
    for projection in programs.iter_mut() {
        projection
            .write_activations(&activations)
            .expect("the activations fit");
    }
    for _ in 0..WARM_UP_PROGRAM_CYCLES {
        for projection in programs.iter_mut() {
            projection.predict().expect("the warm-up prediction runs");
        }
    }
    let mut prediction_durations = Vec::with_capacity(MEASURED_PROGRAM_CYCLES * programs.len());
    for _ in 0..MEASURED_PROGRAM_CYCLES {
        for projection in programs.iter_mut() {
            prediction_durations.push(projection.predict().expect("the prediction runs"));
        }
    }
    TimingSummary::from_samples(&prediction_durations).expect("predictions were timed")
}

/// The volume's free space, read through `df` because the standard library has
/// no call for it and this journey must not fill the disk.
fn free_disk_gibibytes(volume_path: &Path) -> f64 {
    let report = Command::new("df")
        .arg("-k")
        .arg(volume_path)
        .output()
        .expect("df runs");
    let report_text = String::from_utf8_lossy(&report.stdout);
    let data_line = report_text.lines().nth(1).expect("df reports the volume");
    let available_kibibytes: f64 = data_line
        .split_whitespace()
        .nth(3)
        .expect("df has an available column")
        .parse()
        .expect("the available column is a number");
    available_kibibytes / 1_048_576.0
}

/// macOS's estimate of how much memory is free, as a percentage.
fn system_memory_free_percent() -> u32 {
    let report = Command::new("sysctl")
        .args(["-n", "kern.memorystatus_level"])
        .output()
        .expect("sysctl runs");
    String::from_utf8_lossy(&report.stdout)
        .trim()
        .parse()
        .expect("the memory level is a number")
}

/// Whether another program can load without endangering the rest of this Mac.
fn stop_reason(volume_path: &Path) -> Option<String> {
    let free_gibibytes = free_disk_gibibytes(volume_path);
    if free_gibibytes < MINIMUM_FREE_DISK_GIBIBYTES {
        return Some(format!("only {free_gibibytes:.1} GiB of disk is free"));
    }
    let free_percent = system_memory_free_percent();
    if free_percent < MINIMUM_FREE_MEMORY_PERCENT {
        return Some(format!("only {free_percent}% of memory is free"));
    }
    None
}

/// One program that stays usable with neither its package nor its compiled copy
/// on disk, which proves the temporary files can go right after the first run.
fn load_program_without_files(
    layer: &TernaryMlpLayer,
    weight_encoding: StageWeightEncoding,
) -> CoreMlProjection {
    let directory = tempfile::tempdir().expect("a scratch directory for the model");
    let (mut projection, _) = load_program(layer, weight_encoding, directory.path());
    projection.predict().expect("the first prediction runs");
    drop(directory);
    projection.discard_compiled_model_files();
    projection
        .predict()
        .expect("the program still runs without its files");
    projection
}

/// The programs a memory journey managed to load, with the snapshots its verdict
/// needs.
struct LoadedPrograms {
    programs: Vec<CoreMlProjection>,
    before_loads: MemorySnapshot,
    after_first_load: MemorySnapshot,
    total_load_seconds: f64,
}

/// Loads until the layers run out or the guard stops it to protect the Mac.
fn load_programs_within_guard(
    layer_indices: &[usize],
    weight_encoding: StageWeightEncoding,
    scratch_volume: &Path,
) -> LoadedPrograms {
    let mut checkpoint = TernaryCheckpoint::open();
    let before_loads = MemorySnapshot::capture();
    let mut after_first_load = before_loads;
    let mut programs = Vec::with_capacity(layer_indices.len());
    let journey_started_at = Instant::now();
    for (loaded_count, layer_index) in layer_indices.iter().enumerate() {
        if let Some(reason) = stop_reason(scratch_volume) {
            eprintln!("[ternary-mlp] STOPPED after {loaded_count} programs: {reason}");
            break;
        }
        let layer = checkpoint.mlp_layer(*layer_index);
        let load_started_at = Instant::now();
        programs.push(load_program_without_files(&layer, weight_encoding));
        let after_load = MemorySnapshot::capture();
        if loaded_count == 0 {
            after_first_load = after_load;
        }
        let growth = MemoryGrowth::between(before_loads, after_load);
        let finished_count = loaded_count + 1;
        let remaining_seconds = journey_started_at.elapsed().as_secs_f64() / finished_count as f64
            * (layer_indices.len() - finished_count) as f64;
        eprintln!(
            "[ternary-mlp] {finished_count}/{} programs ({:.0}%), layer {layer_index}: load {:.1} s, footprint {:+.0} MiB, wired {:+.0} MiB, memory free {}%, disk free {:.1} GiB, ETA {remaining_seconds:.0} s",
            layer_indices.len(),
            100.0 * finished_count as f64 / layer_indices.len() as f64,
            load_started_at.elapsed().as_secs_f64(),
            mebibytes(growth.process_physical_footprint_bytes),
            mebibytes(growth.system_wired_bytes),
            system_memory_free_percent(),
            free_disk_gibibytes(scratch_volume),
        );
    }
    LoadedPrograms {
        programs,
        before_loads,
        after_first_load,
        total_load_seconds: journey_started_at.elapsed().as_secs_f64(),
    }
}

/// Run alone. The first load also pays for the encoder's own allocations, so the
/// verdict uses the growth between the first load and the last.
fn measure_ternary_mlp_memory(layer_indices: &[usize], weight_encoding: StageWeightEncoding) {
    eprintln!(
        "[ternary-mlp] {} layers, {weight_encoding:?}",
        layer_indices.len()
    );
    let scratch_volume = std::env::temp_dir();
    let disk_before = free_disk_gibibytes(&scratch_volume);
    let mut loaded = load_programs_within_guard(layer_indices, weight_encoding, &scratch_volume);
    let program_count = loaded.programs.len();
    assert!(
        program_count >= MINIMUM_PROGRAMS_FOR_GROWTH,
        "the guard stopped after {program_count} programs, too few to measure growth per layer"
    );
    let cycling = time_program_cycles(&mut loaded.programs);
    eprintln!(
        "[ternary-mlp] cycling {program_count} programs of {NEURAL_ENGINE_TOKEN_COUNT} tokens: median {:.1} ms (p90 {:.1}, max {:.1}), {:.0} ms per chunk",
        milliseconds(cycling.median),
        milliseconds(cycling.ninetieth_percentile),
        milliseconds(cycling.maximum),
        milliseconds(cycling.median) * program_count as f64,
    );
    let settled = MemorySnapshot::capture();
    let later_growth = MemoryGrowth::between(loaded.after_first_load, settled);
    let total_growth = MemoryGrowth::between(loaded.before_loads, settled);
    // Once programs run, the driver wires their weights and the process footprint
    // falls back, so footprint alone reads as almost nothing at 64 programs.
    let later_bytes =
        later_growth.process_physical_footprint_bytes + later_growth.system_wired_bytes;
    let gibibytes_per_layer = later_bytes as f64 / BYTES_PER_GIBIBYTE / (program_count - 1) as f64;
    eprintln!(
        "[ternary-mlp] GATE memory {}: {gibibytes_per_layer:.3} GiB per layer (needs {MAXIMUM_GIBIBYTES_PER_LAYER}); {program_count} programs hold {:.1} GiB of footprint and {:+.1} GiB wired; loading took {:.0} s; disk {disk_before:.1} → {:.1} GiB free",
        gate_verdict(gibibytes_per_layer <= MAXIMUM_GIBIBYTES_PER_LAYER),
        total_growth.process_physical_footprint_bytes as f64 / BYTES_PER_GIBIBYTE,
        total_growth.system_wired_bytes as f64 / BYTES_PER_GIBIBYTE,
        loaded.total_load_seconds,
        free_disk_gibibytes(&scratch_volume),
    );
}

fn every_layer_index() -> Vec<usize> {
    let mut layer_indices = Vec::with_capacity(LAYER_COUNT as usize);
    for layer_index in 0..LAYER_COUNT as usize {
        layer_indices.push(layer_index);
    }
    layer_indices
}

#[test]
#[serial(neural_engine)]
#[ignore = "loads eight real 27B MLP programs as shared lookup tables and cycles through them"]
fn should_measure_the_memory_of_real_ternary_mlp_programs_as_shared_lookup_tables() {
    measure_ternary_mlp_memory(
        &PROGRAM_MEMORY_LAYERS,
        StageWeightEncoding::SharedLookupTables {
            rows_per_table: ROWS_PER_LOOKUP_TABLE,
        },
    );
}

#[test]
#[serial(neural_engine)]
#[ignore = "loads eight real 27B MLP programs as fp16 and cycles through them, for comparison"]
fn should_measure_the_memory_of_real_ternary_mlp_programs_as_fp16() {
    measure_ternary_mlp_memory(&PROGRAM_MEMORY_LAYERS, StageWeightEncoding::Fp16);
}

#[test]
#[serial(neural_engine)]
#[ignore = "loads all 64 real 27B MLP programs; about 11 minutes and over 20 GiB of memory"]
fn should_load_every_real_ternary_mlp_program_as_shared_lookup_tables() {
    measure_ternary_mlp_memory(
        &every_layer_index(),
        StageWeightEncoding::SharedLookupTables {
            rows_per_table: ROWS_PER_LOOKUP_TABLE,
        },
    );
}
