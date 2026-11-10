//! What does storing a dense MLP's weights as 8-bit lookup tables change: the cost of a
//! program's first load, and the weights themselves?
//!
//! macOS compiles a program into the chip's format the first time it is loaded from a
//! path and keeps the result in its own store; later loads from that path reuse it.
//! When the store has lost a program, as after a macOS update, the first-load cost is
//! paid again for every program of a model (issue #2064). These journeys measure that
//! cost for one program at the shape of an Ornith-1.5-9B MLP, as fp16 and as 8-bit
//! lookup tables, and how far the lookup-table builder moves dense weights.
//!
//! Every run compiles to a fresh temporary path, so the first load is always a first
//! load. macOS keeps what it compiled for those paths until it reclaims the space
//! itself: about 290 MB for the fp16 program and 150 MB for the 8-bit one per run.

use std::error::Error;
use std::path::Path;
use std::time::Instant;

use astronomical_neural_engine::core_ml::{
    ComputePlacement, CoreMlComputeUnits, CoreMlProgram, compile_model_to,
};
use astronomical_neural_engine::documents::mlp::{
    MlpFp16Weights, encode_ml_program_mlp_with_encoding,
};
use astronomical_neural_engine::documents::mlp_stages::StageWeightEncoding;
use astronomical_neural_engine::documents::package_layout::write_ml_package;
use astronomical_neural_engine::program_shape::ProgramShape;
use astronomical_neural_engine::weights::half_precision::{f16_bits_to_f32, f32_slice_to_f16_bits};
use astronomical_neural_engine::weights::shared_lookup_tables::{
    LOOKUP_TABLE_ENTRY_COUNT, SharedLookupTables,
};
use astronomical_test_harness::deterministic_values::{
    PROJECTION_WEIGHT_STANDARD_DEVIATION, bell_shaped_values,
};
use serial_test::serial;

/// The dense MLP of Ornith-1.5-9B, for the slice of a chunk serving gives the Neural
/// Engine.
const HIDDEN_CHANNELS: usize = 4096;
const INTERMEDIATE_CHANNELS: usize = 12_288;
const NEURAL_ENGINE_TOKEN_COUNT: usize = 960;
const CONTRACTION_PART_WIDTH: usize = 2048;
/// Sixteen rows per table is the size the Neural Engine runs at its full rate.
const FULL_RATE_ROWS_PER_TABLE: usize = 16;
const GATE_AND_UP_WEIGHT_SEED: u64 = 0x5eed_2064;
const DOWN_WEIGHT_SEED: u64 = 0x5eed_2065;
/// Levels on each side of zero that the first-load weights are rounded to. With 241
/// distinct values a 256-entry table holds every one of them, so both encodings compute
/// the same function and the table builder has nothing to merge: on unrounded
/// bell-shaped weights that merging alone took over eight minutes for this program.
const WEIGHT_LEVELS_PER_SIDE: f32 = 120.0;
const WEIGHT_LEVELS_PER_STANDARD_DEVIATION: f32 = 32.0;
/// What one encoding of the program cost.
struct FirstLoadEvidence {
    weight_file_bytes: usize,
    first_load_seconds: f64,
    second_load_seconds: f64,
    placement: ComputePlacement,
}

fn mlp_shape() -> ProgramShape {
    ProgramShape::new(HIDDEN_CHANNELS, HIDDEN_CHANNELS, NEURAL_ENGINE_TOKEN_COUNT)
        .expect("the MLP shape is valid")
}

fn bell_shaped_fp16_weights(seed: u64, weight_count: usize) -> Vec<u16> {
    f32_slice_to_f16_bits(&bell_shaped_values(
        seed,
        weight_count,
        PROJECTION_WEIGHT_STANDARD_DEVIATION,
    ))
}

/// Bell-shaped weights rounded to a grid coarse enough for one table to hold exactly.
fn table_exact_fp16_weights(seed: u64, weight_count: usize) -> Vec<u16> {
    let level_width = PROJECTION_WEIGHT_STANDARD_DEVIATION / WEIGHT_LEVELS_PER_STANDARD_DEVIATION;
    let mut rounded_weights =
        bell_shaped_values(seed, weight_count, PROJECTION_WEIGHT_STANDARD_DEVIATION);
    for weight in &mut rounded_weights {
        let level = (*weight / level_width)
            .round()
            .clamp(-WEIGHT_LEVELS_PER_SIDE, WEIGHT_LEVELS_PER_SIDE);
        *weight = level * level_width;
    }
    f32_slice_to_f16_bits(&rounded_weights)
}

fn timed_load(
    compiled_program: &Path,
    checks_placement: bool,
) -> Result<(f64, CoreMlProgram), Box<dyn Error>> {
    let load_started_at = Instant::now();
    let program = if checks_placement {
        CoreMlProgram::load_from_path(
            compiled_program,
            mlp_shape(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
        )?
    } else {
        CoreMlProgram::load_from_path_without_placement_check(
            compiled_program,
            mlp_shape(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
        )?
    };
    Ok((load_started_at.elapsed().as_secs_f64(), program))
}

/// Builds the program in `weight_encoding`, compiles it to a path nothing has loaded
/// from, and times the first two loads. Each load is released before the next, so the
/// second one measures what macOS kept and not a program still in memory.
fn measure_first_load(
    weight_encoding: StageWeightEncoding,
    gate_and_up: &[u16],
    down: &[u16],
    working_directory: &Path,
) -> Result<FirstLoadEvidence, Box<dyn Error>> {
    let package_contents = encode_ml_program_mlp_with_encoding(
        mlp_shape(),
        INTERMEDIATE_CHANNELS,
        MlpFp16Weights { gate_and_up, down },
        CONTRACTION_PART_WIDTH,
        weight_encoding,
    )?;
    let weight_file_bytes = package_contents.weight_blob_file.len();
    let package_path = write_ml_package(working_directory, &package_contents)?;
    let compiled_program = working_directory.join("layer.mlmodelc");
    compile_model_to(&package_path, &compiled_program)?;
    let (first_load_seconds, first_program) = timed_load(&compiled_program, false)?;
    drop(first_program);
    let (second_load_seconds, second_program) = timed_load(&compiled_program, false)?;
    drop(second_program);
    let (_, placed_program) = timed_load(&compiled_program, true)?;
    Ok(FirstLoadEvidence {
        weight_file_bytes,
        first_load_seconds,
        second_load_seconds,
        placement: placed_program.placement().preferred,
    })
}

fn print_first_load_evidence(encoding_name: &str, evidence: &FirstLoadEvidence) {
    eprintln!(
        "[neural-engine-journey] first load, {encoding_name}: weight file {:.0} MB, first load \
         {:.2} s, second load {:.3} s, placed on {:?}",
        evidence.weight_file_bytes as f64 / 1e6,
        evidence.first_load_seconds,
        evidence.second_load_seconds,
        evidence.placement
    );
}

#[test]
#[serial(neural_engine)]
#[ignore = "compiles and loads two 151-million-weight programs on the real Neural Engine"]
fn should_measure_the_first_load_of_a_dense_mlp_as_fp16_and_as_lookup_tables() {
    let gate_and_up = table_exact_fp16_weights(
        GATE_AND_UP_WEIGHT_SEED,
        2 * INTERMEDIATE_CHANNELS * HIDDEN_CHANNELS,
    );
    let down = table_exact_fp16_weights(DOWN_WEIGHT_SEED, HIDDEN_CHANNELS * INTERMEDIATE_CHANNELS);
    let fp16_directory = tempfile::tempdir().expect("a temporary directory");
    let lookup_table_directory = tempfile::tempdir().expect("a temporary directory");

    let fp16_evidence = measure_first_load(
        StageWeightEncoding::Fp16,
        &gate_and_up,
        &down,
        fp16_directory.path(),
    )
    .expect("the fp16 program compiles and loads");
    let lookup_table_evidence = measure_first_load(
        StageWeightEncoding::SharedLookupTables {
            rows_per_table: FULL_RATE_ROWS_PER_TABLE,
        },
        &gate_and_up,
        &down,
        lookup_table_directory.path(),
    )
    .expect("the lookup-table program compiles and loads");

    print_first_load_evidence("fp16", &fp16_evidence);
    print_first_load_evidence("8-bit lookup tables", &lookup_table_evidence);
    for evidence in [&fp16_evidence, &lookup_table_evidence] {
        assert_eq!(evidence.placement, ComputePlacement::NeuralEngine);
        assert!(
            evidence.second_load_seconds < evidence.first_load_seconds,
            "a second load took {:.3} s against a first load of {:.3} s",
            evidence.second_load_seconds,
            evidence.first_load_seconds
        );
    }
    assert!(
        lookup_table_evidence.weight_file_bytes * 100 < fp16_evidence.weight_file_bytes * 55,
        "{} lookup-table bytes against {} fp16 bytes",
        lookup_table_evidence.weight_file_bytes,
        fp16_evidence.weight_file_bytes
    );
}

/// The relative RMS distance between weights and what their lookup tables hold.
fn relative_table_error(
    fp16_weights: &[u16],
    output_channels: usize,
    input_channels: usize,
    rows_per_table: usize,
) -> f64 {
    let tables = SharedLookupTables::from_fp16_weights(
        fp16_weights,
        output_channels,
        input_channels,
        rows_per_table,
    )
    .expect("the tables build");
    let mut squared_error_sum = 0.0_f64;
    let mut squared_weight_sum = 0.0_f64;
    for (weight_index, original_bits) in fp16_weights.iter().enumerate() {
        let table_index = weight_index / input_channels / rows_per_table;
        let table_entry =
            table_index * LOOKUP_TABLE_ENTRY_COUNT + usize::from(tables.codes[weight_index]);
        let original = f64::from(f16_bits_to_f32(*original_bits));
        let stored = f64::from(f16_bits_to_f32(tables.fp16_tables[table_entry]));
        squared_error_sum += (stored - original) * (stored - original);
        squared_weight_sum += original * original;
    }
    (squared_error_sum / squared_weight_sum).sqrt()
}

/// On one part of a real Ornith-1.5-9B MLP layer this read 0.374 at 16 rows per table,
/// 0.323 at 4 and 0.014 at 1 (issue #2064): the builder was made for ternary weights
/// and merges neighbouring values until a table fits, which a dense population pays
/// for once a table holds more than a row. The fixture's bell-shaped weights show the
/// same order without needing the model.
#[test]
#[ignore = "builds lookup tables over eight million weights three times; about a minute"]
fn should_measure_how_far_shared_lookup_tables_move_dense_weights() {
    const PART_OUTPUT_CHANNELS: usize = 4096;
    let fp16_weights = bell_shaped_fp16_weights(
        GATE_AND_UP_WEIGHT_SEED,
        PART_OUTPUT_CHANNELS * CONTRACTION_PART_WIDTH,
    );
    let mut errors_by_rows_per_table = Vec::new();
    for rows_per_table in [FULL_RATE_ROWS_PER_TABLE, 4, 1] {
        let relative_error = relative_table_error(
            &fp16_weights,
            PART_OUTPUT_CHANNELS,
            CONTRACTION_PART_WIDTH,
            rows_per_table,
        );
        eprintln!(
            "[neural-engine-journey] dense weights in shared tables, {rows_per_table} row(s) per \
             table: relative RMS error {relative_error:.4}"
        );
        errors_by_rows_per_table.push(relative_error);
    }

    assert!(
        errors_by_rows_per_table[2] < errors_by_rows_per_table[0],
        "one row per table should hold dense weights more closely than sixteen: {errors_by_rows_per_table:?}"
    );
}
