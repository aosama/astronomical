//! Every model document format compiled by the real Core ML and run on the CPU.
//!
//! The encoders' unit tests only read back the bytes they wrote, so a document
//! Core ML itself refuses would pass them. Limiting these to CPU compute units
//! keeps the contract independent of the Neural Engine, so it holds on any Mac.

use astronomical_neural_engine::core_ml::{CoreMlComputeUnits, CoreMlProgram};
use astronomical_neural_engine::documents::mlp::MlpFp16Weights;
use astronomical_neural_engine::documents::mlp_stages::StageWeightEncoding;
use astronomical_neural_engine::documents::projection::MlProgramWeights;
use astronomical_neural_engine::documents::rotated_mlp::{RotatedMlpLayout, RotatedMlpWeights};
use astronomical_neural_engine::program_shape::ProgramShape;
use astronomical_neural_engine::weights::affine_quantized::{
    AffineQuantizedWeights, QuantizationLayout,
};
use astronomical_neural_engine::weights::half_precision::{
    f16_bits_slice_to_f32, f32_slice_to_f16_bits, f32_to_f16_bits,
};
use astronomical_neural_engine_research::accelerate_reference::reference_projection;
use astronomical_neural_engine_research::numerical_comparison::compare_values;
use astronomical_test_harness::deterministic_values::{
    PROJECTION_WEIGHT_STANDARD_DEVIATION, bell_shaped_values,
};

use crate::support::journey_constants::MINIMUM_COSINE_SIMILARITY;
use crate::support::program_loading::{
    compile_ml_program_to, load_ml_program, load_ml_program_mlp, load_ml_program_rotated_mlp,
    load_neural_network,
};

const INPUT_CHANNELS: usize = 8;
const OUTPUT_CHANNELS: usize = 4;
const TOKEN_COUNT: usize = 2;
const SINGLE_CONTRACTION_PART: usize = 1;
const WEIGHT_SEED: u64 = 0x5eed_c001;
const ACTIVATION_SEED: u64 = 0x5eed_c002;
const MLP_HIDDEN_CHANNELS: usize = 8;
const MLP_INTERMEDIATE_CHANNELS: usize = 16;
const MLP_PART_WIDTH: usize = 4;
const ROTATION_BLOCK_SIZE: usize = 4;
const ROWS_PER_LOOKUP_TABLE: usize = 4;
const EIGHT_BIT_CODES: u8 = 8;

fn projection_shape() -> ProgramShape {
    ProgramShape::new(INPUT_CHANNELS, OUTPUT_CHANNELS, TOKEN_COUNT).expect("valid shape")
}

fn projection_weights() -> Vec<f32> {
    bell_shaped_values(
        WEIGHT_SEED,
        OUTPUT_CHANNELS * INPUT_CHANNELS,
        PROJECTION_WEIGHT_STANDARD_DEVIATION,
    )
}

fn unit_activations(channel_count: usize) -> Vec<f32> {
    bell_shaped_values(ACTIVATION_SEED, channel_count * TOKEN_COUNT, 1.0)
}

fn predict_on_the_cpu(program: &mut CoreMlProgram, activations: &[f32]) -> Vec<f32> {
    program
        .write_activations(&f32_slice_to_f16_bits(activations))
        .expect("the activations fit the input surface");
    program.predict().expect("the CPU prediction runs");
    f16_bits_slice_to_f32(&program.read_projected().expect("the output reads back"))
}

fn assert_matches_reference(program: &mut CoreMlProgram, weights: &[f32]) {
    let activations = unit_activations(INPUT_CHANNELS);
    let projected = predict_on_the_cpu(program, &activations);
    let reference = reference_projection(projection_shape(), weights, &activations)
        .expect("the reference has matching sizes");
    let comparison = compare_values(&reference, &projected).expect("the outputs have one length");
    assert!(
        comparison.cosine_similarity >= MINIMUM_COSINE_SIMILARITY,
        "Core ML's result drifted from the reference: {comparison:?}"
    );
}

fn assert_all_finite_and_not_all_zero(output_values: &[f32]) {
    let mut has_nonzero_value = false;
    for output_value in output_values {
        assert!(output_value.is_finite(), "the output holds {output_value}");
        has_nonzero_value = has_nonzero_value || *output_value != 0.0;
    }
    assert!(has_nonzero_value, "the output is entirely zero");
}

#[test]
fn should_compile_and_run_a_neural_network_document() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let weights = projection_weights();

    let mut program = load_neural_network(
        projection_shape(),
        &f32_slice_to_f16_bits(&weights),
        CoreMlComputeUnits::CpuOnly,
        SINGLE_CONTRACTION_PART,
        model_directory.path(),
    )
    .expect("Core ML accepts the document");

    assert_matches_reference(&mut program, &weights);
}

#[test]
fn should_compile_and_run_an_ml_program_with_fp16_weights() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let weights = projection_weights();
    let fp16_weights = f32_slice_to_f16_bits(&weights);

    let mut program = load_ml_program(
        projection_shape(),
        MlProgramWeights::Fp16(&fp16_weights),
        CoreMlComputeUnits::CpuOnly,
        SINGLE_CONTRACTION_PART,
        model_directory.path(),
    )
    .expect("Core ML accepts the program");

    assert_matches_reference(&mut program, &weights);
}

#[test]
fn should_compile_and_run_an_ml_program_split_into_contraction_parts() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let weights = projection_weights();
    let fp16_weights = f32_slice_to_f16_bits(&weights);

    let mut program = load_ml_program(
        projection_shape(),
        MlProgramWeights::Fp16(&fp16_weights),
        CoreMlComputeUnits::CpuOnly,
        2,
        model_directory.path(),
    )
    .expect("Core ML accepts the split program");

    assert_matches_reference(&mut program, &weights);
}

#[test]
fn should_compile_and_run_an_ml_program_with_lookup_table_weights() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let quantized_weights = AffineQuantizedWeights::quantize(
        &projection_weights(),
        OUTPUT_CHANNELS,
        INPUT_CHANNELS,
        QuantizationLayout {
            code_bits: EIGHT_BIT_CODES,
            block_output_rows: OUTPUT_CHANNELS,
            block_input_columns: INPUT_CHANNELS,
        },
    )
    .expect("the weights cut into one block");

    let mut program = load_ml_program(
        projection_shape(),
        MlProgramWeights::LookupTable(&quantized_weights),
        CoreMlComputeUnits::CpuOnly,
        SINGLE_CONTRACTION_PART,
        model_directory.path(),
    )
    .expect("Core ML accepts the compressed program");

    assert_matches_reference(
        &mut program,
        &f16_bits_slice_to_f32(&quantized_weights.dequantized_fp16()),
    );
}

#[test]
fn should_compile_and_run_a_whole_mlp_program() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let shape = ProgramShape::new(MLP_HIDDEN_CHANNELS, MLP_HIDDEN_CHANNELS, TOKEN_COUNT)
        .expect("valid shape");

    let mut program = load_ml_program_mlp(
        shape,
        MLP_INTERMEDIATE_CHANNELS,
        MlpFp16Weights {
            gate_and_up: &vec![
                f32_to_f16_bits(0.25);
                2 * MLP_INTERMEDIATE_CHANNELS * MLP_HIDDEN_CHANNELS
            ],
            down: &vec![f32_to_f16_bits(0.25); MLP_HIDDEN_CHANNELS * MLP_INTERMEDIATE_CHANNELS],
        },
        MLP_PART_WIDTH,
        CoreMlComputeUnits::CpuOnly,
        model_directory.path(),
    )
    .expect("Core ML accepts the MLP program");

    assert_all_finite_and_not_all_zero(&predict_on_the_cpu(
        &mut program,
        &unit_activations(MLP_HIDDEN_CHANNELS),
    ));
}

#[test]
fn should_compile_and_run_a_rotated_mlp_program() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let layout = RotatedMlpLayout {
        shape: ProgramShape::new(MLP_HIDDEN_CHANNELS, MLP_HIDDEN_CHANNELS, TOKEN_COUNT)
            .expect("valid shape"),
        intermediate_channels: MLP_INTERMEDIATE_CHANNELS,
        contraction_part_width: MLP_PART_WIDTH,
        rotation_block_size: ROTATION_BLOCK_SIZE,
        weight_encoding: StageWeightEncoding::SharedLookupTables {
            rows_per_table: ROWS_PER_LOOKUP_TABLE,
        },
    };

    let mut program = load_ml_program_rotated_mlp(
        layout,
        RotatedMlpWeights {
            gate_and_up: &vec![
                f32_to_f16_bits(0.25);
                2 * MLP_INTERMEDIATE_CHANNELS * MLP_HIDDEN_CHANNELS
            ],
            down: &vec![f32_to_f16_bits(0.25); MLP_HIDDEN_CHANNELS * MLP_INTERMEDIATE_CHANNELS],
            intermediate_signs: &[f32_to_f16_bits(1.0); MLP_INTERMEDIATE_CHANNELS],
        },
        CoreMlComputeUnits::CpuOnly,
        model_directory.path(),
    )
    .expect("Core ML accepts the rotated program");

    assert_all_finite_and_not_all_zero(&predict_on_the_cpu(
        &mut program,
        &unit_activations(MLP_HIDDEN_CHANNELS),
    ));
}

#[test]
fn should_load_a_program_compiled_earlier_and_leave_its_files_in_place() {
    let scratch_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = scratch_directory.path().join("kept/program.mlmodelc");
    let weights = projection_weights();
    let fp16_weights = f32_slice_to_f16_bits(&weights);
    compile_ml_program_to(
        projection_shape(),
        MlProgramWeights::Fp16(&fp16_weights),
        SINGLE_CONTRACTION_PART,
        scratch_directory.path(),
        &kept_program_path,
    )
    .expect("Core ML compiles the program");

    let mut program = CoreMlProgram::load_from_path(
        &kept_program_path,
        projection_shape(),
        CoreMlComputeUnits::CpuOnly,
    )
    .expect("the kept program loads");
    assert_matches_reference(&mut program, &weights);
    drop(program);

    assert!(kept_program_path.is_dir());
}
