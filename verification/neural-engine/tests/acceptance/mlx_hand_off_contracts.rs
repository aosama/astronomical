//! Data passing between MLX and a loaded program, which serving relies on and
//! which holds the crate's only raw MLX calls.
//!
//! The program runs on the CPU, because the hand-off's layout and ownership rules
//! do not depend on which engine produced the output. Tokens stay at a multiple of
//! 64, the alignment that lets MLX read the surface as one array.
//!
//! These are ignored because the MLX runtime needs its compiled Metal library
//! beside the test executable, and the archive CI runs tests from does not carry
//! it. Run them with `cargo test-neural-engine-journeys --filter mlx_hand_off`.

use astronomical_neural_engine::core_ml::{CoreMlComputeUnits, CoreMlProgram};
use astronomical_neural_engine::documents::projection::MlProgramWeights;
use astronomical_neural_engine::mlx_hand_off::{
    MlxHandOffError, OutputHandOff, SplitAxis, channel_major_fp16, merge_surface_with_gpu_slice,
    surface_as_token_major, write_prepared_input,
};
use astronomical_neural_engine::program_shape::ProgramShape;
use astronomical_neural_engine::weights::half_precision::f32_to_f16_bits;
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

use crate::support::mlx_projection::{initialize_runtime, upload};
use crate::support::program_loading::load_ml_program;

const CHANNEL_COUNT: usize = 4;
const ALIGNED_TOKEN_COUNT: usize = 64;
const SINGLE_CONTRACTION_PART: usize = 1;

/// `[channels][channels]` with ones on the diagonal, so the program's output is
/// its input and a wrong layout shows up as wrong values.
fn identity_weights(channel_count: usize) -> Vec<u16> {
    let mut fp16_weights = vec![0_u16; channel_count * channel_count];
    for channel_index in 0..channel_count {
        fp16_weights[channel_index * channel_count + channel_index] = f32_to_f16_bits(1.0);
    }
    fp16_weights
}

fn load_identity_program(token_count: usize, model_directory: &std::path::Path) -> CoreMlProgram {
    load_ml_program(
        ProgramShape::new(CHANNEL_COUNT, CHANNEL_COUNT, token_count).expect("valid shape"),
        MlProgramWeights::Fp16(&identity_weights(CHANNEL_COUNT)),
        CoreMlComputeUnits::CpuOnly,
        SINGLE_CONTRACTION_PART,
        model_directory,
    )
    .expect("Core ML accepts the identity program")
}

fn counting_values(token_count: usize) -> Vec<f32> {
    let mut token_major_values = Vec::with_capacity(token_count * CHANNEL_COUNT);
    for value_index in 0..token_count * CHANNEL_COUNT {
        token_major_values.push(value_index as f32);
    }
    token_major_values
}

fn prepared_activations(runtime: &MlxRuntime, token_major: &MlxArray) -> MlxArray {
    let prepared = channel_major_fp16(runtime, token_major).expect("the reorder builds");
    runtime
        .evaluate_arrays(&[&prepared])
        .expect("the reorder evaluates");
    prepared
}

fn run_identity_program(runtime: &MlxRuntime, program: &mut CoreMlProgram, token_count: usize) {
    let token_major = upload(
        runtime,
        &counting_values(token_count),
        &[1, token_count as i32, CHANNEL_COUNT as i32],
        MlxDtype::Float32,
    );
    write_prepared_input(&prepared_activations(runtime, &token_major), program)
        .expect("the activations fit the input surface");
    program.predict().expect("the CPU prediction runs");
}

fn values_of(runtime: &MlxRuntime, values: &MlxArray) -> Vec<f32> {
    let single_precision = runtime
        .astype(values, MlxDtype::Float32)
        .expect("the values convert");
    runtime
        .array_to_vec_f32(&single_precision)
        .expect("the values read back")
}

#[test]
#[ignore = "needs the MLX runtime; run through cargo test-neural-engine-journeys"]
#[serial]
fn should_reorder_token_major_activations_into_channel_major_fp16() {
    let runtime = initialize_runtime();
    let token_major = upload(
        &runtime,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1, 2, 3],
        MlxDtype::Float32,
    );

    let prepared = prepared_activations(&runtime, &token_major);

    assert_eq!(prepared.dtype(), MlxDtype::Float16);
    assert_eq!(prepared.shape(), [3, 2]);
    assert_eq!(
        values_of(&runtime, &prepared),
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
#[ignore = "needs the MLX runtime; run through cargo test-neural-engine-journeys"]
#[serial]
fn should_refuse_activations_that_are_not_fp16() {
    let runtime = initialize_runtime();
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let mut program = load_identity_program(ALIGNED_TOKEN_COUNT, model_directory.path());
    let single_precision = upload(
        &runtime,
        &counting_values(ALIGNED_TOKEN_COUNT),
        &[CHANNEL_COUNT as i32, ALIGNED_TOKEN_COUNT as i32],
        MlxDtype::Float32,
    );

    let write_result = write_prepared_input(&single_precision, &mut program);

    assert!(matches!(
        write_result,
        Err(MlxHandOffError::PreparedActivationsNotContiguous)
    ));
}

#[test]
#[ignore = "needs the MLX runtime; run through cargo test-neural-engine-journeys"]
#[serial]
fn should_refuse_prepared_activations_of_the_wrong_size() {
    let runtime = initialize_runtime();
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let mut program = load_identity_program(ALIGNED_TOKEN_COUNT, model_directory.path());
    let too_few_tokens = upload(
        &runtime,
        &counting_values(1),
        &[1, 1, CHANNEL_COUNT as i32],
        MlxDtype::Float32,
    );

    let write_result = write_prepared_input(
        &prepared_activations(&runtime, &too_few_tokens),
        &mut program,
    );

    assert!(matches!(write_result, Err(MlxHandOffError::Program(_))));
}

#[test]
#[ignore = "needs the MLX runtime; run through cargo test-neural-engine-journeys"]
#[serial]
fn should_return_the_program_output_as_token_major_in_both_hand_off_forms() {
    let runtime = initialize_runtime();
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let mut program = load_identity_program(ALIGNED_TOKEN_COUNT, model_directory.path());
    run_identity_program(&runtime, &mut program, ALIGNED_TOKEN_COUNT);

    for output_hand_off in [OutputHandOff::CopiedSurface, OutputHandOff::ZeroCopySurface] {
        let token_major =
            surface_as_token_major(&runtime, &program, MlxDtype::Float32, output_hand_off)
                .expect("the surface reaches MLX");

        assert_eq!(
            token_major.shape(),
            [1, ALIGNED_TOKEN_COUNT as i32, CHANNEL_COUNT as i32]
        );
        assert_eq!(
            values_of(&runtime, &token_major),
            counting_values(ALIGNED_TOKEN_COUNT)
        );
    }
}

#[test]
#[ignore = "needs the MLX runtime; run through cargo test-neural-engine-journeys"]
#[serial]
fn should_refuse_a_surface_whose_rows_are_padded() {
    let runtime = initialize_runtime();
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let unaligned_token_count = 3;
    let mut program = load_identity_program(unaligned_token_count, model_directory.path());
    run_identity_program(&runtime, &mut program, unaligned_token_count);

    let hand_off_result = surface_as_token_major(
        &runtime,
        &program,
        MlxDtype::Float32,
        OutputHandOff::CopiedSurface,
    );

    assert!(matches!(
        hand_off_result,
        Err(MlxHandOffError::PaddedOutputSurface)
    ));
}

#[test]
#[ignore = "needs the MLX runtime; run through cargo test-neural-engine-journeys"]
#[serial]
fn should_merge_the_program_output_before_a_gpu_slice_along_tokens() {
    let runtime = initialize_runtime();
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let mut program = load_identity_program(ALIGNED_TOKEN_COUNT, model_directory.path());
    run_identity_program(&runtime, &mut program, ALIGNED_TOKEN_COUNT);
    let gpu_tokens = upload(
        &runtime,
        &[-1.0; CHANNEL_COUNT],
        &[1, 1, CHANNEL_COUNT as i32],
        MlxDtype::Float32,
    );

    let merged = merge_surface_with_gpu_slice(
        &runtime,
        &program,
        &gpu_tokens,
        SplitAxis::Tokens,
        OutputHandOff::CopiedSurface,
    )
    .expect("the two engines' tokens merge");

    assert_eq!(
        merged.shape(),
        [1, ALIGNED_TOKEN_COUNT as i32 + 1, CHANNEL_COUNT as i32]
    );
    let mut expected_values = counting_values(ALIGNED_TOKEN_COUNT);
    expected_values.extend([-1.0; CHANNEL_COUNT]);
    assert_eq!(values_of(&runtime, &merged), expected_values);
}

#[test]
#[ignore = "needs the MLX runtime; run through cargo test-neural-engine-journeys"]
#[serial]
fn should_merge_the_program_output_before_a_gpu_slice_along_channels() {
    let runtime = initialize_runtime();
    let model_directory = tempfile::tempdir().expect("a scratch directory");
    let mut program = load_identity_program(ALIGNED_TOKEN_COUNT, model_directory.path());
    run_identity_program(&runtime, &mut program, ALIGNED_TOKEN_COUNT);
    let gpu_channels = upload(
        &runtime,
        &[-1.0; ALIGNED_TOKEN_COUNT],
        &[1, ALIGNED_TOKEN_COUNT as i32, 1],
        MlxDtype::Float32,
    );

    let merged = merge_surface_with_gpu_slice(
        &runtime,
        &program,
        &gpu_channels,
        SplitAxis::Channels,
        OutputHandOff::CopiedSurface,
    )
    .expect("the two engines' channels merge");

    assert_eq!(
        merged.shape(),
        [1, ALIGNED_TOKEN_COUNT as i32, CHANNEL_COUNT as i32 + 1]
    );
}
