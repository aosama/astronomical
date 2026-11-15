//! Component contract of the row-batched 4-bit matmul against MLX's own
//! quantized matmul, on the Qwen3.8 27B's projection shapes.
//!
//! Two promises are checked. The values match MLX's within bf16 rounding,
//! because both accumulate in float32 and round once. And a row's result is
//! bit-identical whether it is computed alone or inside a block of rows,
//! which is what lets a drafted verify pass keep exactly the tokens plain
//! decoding would.

use astronomical_model_serving::row_batched_matmul::{
    RowBatchedMatmulKernels, RowBatchedProjectionError, row_batched_quantized_matmul,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::serial;

const ACTIVE_MEMORY_LIMIT_BYTES: usize = 4 * 1024 * 1024 * 1024;
const MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
/// `(output, input)`: wide, narrow-and-long, small, and the mixer output.
const PROJECTION_SHAPES: [(i32, i32); 4] =
    [(17408, 5120), (5120, 17408), (1024, 5120), (5120, 6144)];
/// One row, a count inside a block, one block, just past it, and the most.
const ROW_COUNTS: [i32; 5] = [1, 5, 16, 17, 32];
/// bf16 keeps 8 significant bits, so two float32 sums of the same products
/// taken in different orders can round to neighbouring bf16 values.
const MINIMUM_COSINE_SIMILARITY: f64 = 0.9999;
/// Two float32 sums of the same products in a different order, each rounded
/// once to bf16 (8 significant bits), can land one bf16 step apart; for
/// values a few times the RMS that step is a few hundredths of the RMS.
const MAXIMUM_DIFFERENCE_OVER_RMS: f64 = 0.05;

fn initialize_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(ACTIVE_MEMORY_LIMIT_BYTES, MLX_BUFFER_CACHE_LIMIT_BYTES)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes")
}

/// One quantized weight in MLX's layout, which both kernels read as stored.
struct ShapeFixture {
    packed_weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

fn shape_fixture(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    output_width: i32,
    input_width: i32,
) -> ShapeFixture {
    let weight = runtime
        .random_normal(
            &[output_width, input_width],
            MlxDtype::BFloat16,
            0.0,
            0.02,
            random_key,
        )
        .expect("weights build");
    let (packed_weight, scales, biases) = runtime
        .quantize_affine(&weight, 64, 4)
        .expect("weights quantize");
    runtime
        .evaluate_arrays(&[&packed_weight, &scales, &biases])
        .expect("weights materialize");
    ShapeFixture {
        packed_weight,
        scales,
        biases,
    }
}

impl ShapeFixture {
    fn row_batched(
        &self,
        runtime: &MlxRuntime,
        kernels: &RowBatchedMatmulKernels,
        rows: &MlxArray,
    ) -> Result<MlxArray, RowBatchedProjectionError> {
        row_batched_quantized_matmul(
            runtime,
            kernels,
            rows,
            &self.packed_weight,
            &self.scales,
            &self.biases,
        )
    }
}

fn values(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    let single_precision = runtime
        .astype(array, MlxDtype::Float32)
        .expect("values convert");
    runtime
        .array_to_vec_f32(&single_precision)
        .expect("values read back")
}

/// How two outputs of the same projection differ value by value. Cosine alone
/// could hide a few wrong columns; the largest single difference cannot.
struct OutputDifference {
    differing_value_count: usize,
    /// The largest absolute difference over the reference output's RMS.
    largest_difference_over_rms: f64,
}

impl OutputDifference {
    fn between(candidate: &[f32], reference: &[f32]) -> Self {
        let mut differing_value_count = 0;
        let mut largest_difference = 0.0_f64;
        let mut reference_square_sum = 0.0_f64;
        for (candidate_value, reference_value) in candidate.iter().zip(reference) {
            if candidate_value.to_bits() != reference_value.to_bits() {
                differing_value_count += 1;
            }
            largest_difference =
                largest_difference.max(f64::from(*candidate_value - *reference_value).abs());
            reference_square_sum += f64::from(*reference_value).powi(2);
        }
        let reference_rms = (reference_square_sum / reference.len() as f64).sqrt();
        Self {
            differing_value_count,
            largest_difference_over_rms: largest_difference / reference_rms,
        }
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    let (mut dot, mut left_norm, mut right_norm) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (left_value, right_value) in left.iter().zip(right) {
        dot += f64::from(*left_value) * f64::from(*right_value);
        left_norm += f64::from(*left_value).powi(2);
        right_norm += f64::from(*right_value).powi(2);
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_match_mlx_quantized_matmul_for_every_27b_shape_and_row_count() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(7).expect("a key builds");
    for (output_width, input_width) in PROJECTION_SHAPES {
        let fixture = shape_fixture(&runtime, &random_key, output_width, input_width);
        for row_count in ROW_COUNTS {
            let rows = runtime
                .random_normal(
                    &[row_count, input_width],
                    MlxDtype::BFloat16,
                    0.0,
                    1.0,
                    &random_key,
                )
                .expect("rows build");
            let kernel_output = fixture
                .row_batched(&runtime, &kernels, &rows)
                .expect("the kernel runs");
            let mlx_output = runtime
                .quantized_matmul_affine(
                    &rows,
                    &fixture.packed_weight,
                    &fixture.scales,
                    &fixture.biases,
                    true,
                    64,
                    4,
                )
                .expect("MLX's matmul builds");
            let kernel_values = values(&runtime, &kernel_output);
            let mlx_values = values(&runtime, &mlx_output);
            let similarity = cosine_similarity(&kernel_values, &mlx_values);
            let difference = OutputDifference::between(&kernel_values, &mlx_values);
            eprintln!(
                "[row-batched] {output_width}x{input_width} K={row_count}: cosine {similarity:.7}, \
                 {} of {} values differ, largest difference {:.4} of the output's RMS",
                difference.differing_value_count,
                kernel_values.len(),
                difference.largest_difference_over_rms
            );
            assert!(
                similarity >= MINIMUM_COSINE_SIMILARITY,
                "{output_width}x{input_width} at {row_count} rows: cosine {similarity}"
            );
            assert!(
                difference.largest_difference_over_rms <= MAXIMUM_DIFFERENCE_OVER_RMS,
                "{output_width}x{input_width} at {row_count} rows: a value differs by {} of the RMS",
                difference.largest_difference_over_rms
            );
        }
    }
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_give_each_row_the_same_bits_alone_and_inside_a_sixteen_row_block() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(11).expect("a key builds");
    for (output_width, input_width) in PROJECTION_SHAPES {
        let fixture = shape_fixture(&runtime, &random_key, output_width, input_width);
        let rows = runtime
            .random_normal(
                &[16, input_width],
                MlxDtype::BFloat16,
                0.0,
                1.0,
                &random_key,
            )
            .expect("rows build");
        let block_values = values(
            &runtime,
            &fixture
                .row_batched(&runtime, &kernels, &rows)
                .expect("the block runs"),
        );
        for row_index in [0, 7, 15] {
            let single_row = runtime
                .slice(
                    &rows,
                    &[row_index, 0],
                    &[row_index + 1, input_width],
                    &[1, 1],
                )
                .expect("one row slices");
            let alone_values = values(
                &runtime,
                &fixture
                    .row_batched(&runtime, &kernels, &single_row)
                    .expect("one row runs"),
            );
            let width = output_width as usize;
            let block_row =
                &block_values[row_index as usize * width..(row_index as usize + 1) * width];
            assert_eq!(
                block_row,
                alone_values.as_slice(),
                "{output_width}x{input_width} row {row_index}"
            );
        }
    }
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_refuse_more_rows_than_the_kernel_takes() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(13).expect("a key builds");
    let fixture = shape_fixture(&runtime, &random_key, 1024, 5120);
    let rows = runtime
        .random_normal(&[33, 5120], MlxDtype::BFloat16, 0.0, 1.0, &random_key)
        .expect("rows build");
    assert!(matches!(
        fixture.row_batched(&runtime, &kernels, &rows),
        Err(RowBatchedProjectionError::Refused(_))
    ));
}

#[test]
#[ignore = "runs MLX on the GPU"]
#[serial]
fn should_refuse_group_32_and_float32_scale_packings_instead_of_misreading_them() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(17).expect("a key builds");
    let rows = runtime
        .random_normal(&[4, 5120], MlxDtype::BFloat16, 0.0, 1.0, &random_key)
        .expect("rows build");
    let weight = runtime
        .random_normal(&[1024, 5120], MlxDtype::BFloat16, 0.0, 0.02, &random_key)
        .expect("weights build");
    let (group_32_weight, group_32_scales, group_32_biases) = runtime
        .quantize_affine(&weight, 32, 4)
        .expect("group 32 quantizes");
    assert!(matches!(
        row_batched_quantized_matmul(
            &runtime,
            &kernels,
            &rows,
            &group_32_weight,
            &group_32_scales,
            &group_32_biases
        ),
        Err(RowBatchedProjectionError::UnsupportedQuantization { .. })
    ));
    let (packed_weight, scales, biases) = runtime
        .quantize_affine(&weight, 64, 4)
        .expect("group 64 quantizes");
    let float32_scales = runtime
        .astype(&scales, MlxDtype::Float32)
        .expect("scales widen");
    assert!(matches!(
        row_batched_quantized_matmul(
            &runtime,
            &kernels,
            &rows,
            &packed_weight,
            &float32_scales,
            &biases
        ),
        Err(RowBatchedProjectionError::UnsupportedQuantization { .. })
    ));
}
