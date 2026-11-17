//! The lowest rung under draft-tree verification: whether one MLX kernel
//! gives a row the same bits inside a window of rows as alone. A draft-tree
//! round verifies up to 16 rows at once, and its greedy picks equal plain
//! decoding only when every row equals its one-row step. MLX chooses its
//! quantized matmul and its attention kernel by the number of rows, so this
//! measures each kernel at the 35B-A3B mixture-of-experts target's own
//! shapes, row by row, for windows of 2, 4, 8 and 16 rows.
//!
//! Random bf16 inputs stand in for activations: the question is which
//! reduction the kernel runs, not what the model computes. It reports; it
//! asserts nothing.

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::serial;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
};

const WINDOW_ROW_COUNTS: [i32; 4] = [2, 4, 8, 16];
const QUANTIZATION_GROUP_SIZE: i32 = 64;
const HIDDEN_SIZE: i32 = 2_048;
/// (bits, output width, what that width is on the 35B-A3B target).
const PROJECTION_SHAPES: [(i32, i32, &str); 6] = [
    (8, 256, "router, 8-bit"),
    (4, 512, "shared expert gate or up, 4-bit"),
    (8, 512, "shared expert gate or up, 8-bit"),
    (4, 4_096, "full-attention query and gate, 4-bit"),
    (8, 4_096, "full-attention query and gate, 8-bit"),
    (8, 8_192, "DeltaNet input projection, 8-bit"),
];
const QUERY_HEAD_COUNT: i32 = 16;
const KEY_VALUE_HEAD_COUNT: i32 = 2;
const HEAD_SIZE: i32 = 256;
const STORED_KEY_COUNT: i32 = 1_024;

#[test]
#[ignore = "runs MLX kernels on the GPU; run through cargo measure-batched-kernel-row-exactness"]
#[serial]
fn should_report_how_far_a_quantized_matmul_row_drifts_inside_a_window() {
    let runtime = direct_mlx_runtime();
    for (bits, output_width, label) in PROJECTION_SHAPES {
        let float_weights = normal(&runtime, &[output_width, HIDDEN_SIZE], 0.02, 11);
        let (packed, scales, biases) = runtime
            .quantize_affine(&float_weights, QUANTIZATION_GROUP_SIZE, bits)
            .expect("the weights quantize");
        let inputs = normal(&runtime, &[16, HIDDEN_SIZE], 1.0, 23);
        let multiply = |rows: &MlxArray| {
            runtime
                .quantized_matmul_affine(
                    rows,
                    &packed,
                    &scales,
                    &biases,
                    true,
                    QUANTIZATION_GROUP_SIZE,
                    bits,
                )
                .expect("the quantized matmul builds")
        };
        report_window_drift(&runtime, label, &inputs, 0, &multiply);
    }
}

#[test]
#[ignore = "runs MLX kernels on the GPU; run through cargo measure-batched-kernel-row-exactness"]
#[serial]
fn should_report_how_far_an_attention_row_drifts_inside_a_window() {
    let runtime = direct_mlx_runtime();
    let queries = normal(&runtime, &[1, QUERY_HEAD_COUNT, 16, HEAD_SIZE], 1.0, 31);
    let keys = normal(
        &runtime,
        &[1, KEY_VALUE_HEAD_COUNT, STORED_KEY_COUNT, HEAD_SIZE],
        1.0,
        37,
    );
    let values = normal(
        &runtime,
        &[1, KEY_VALUE_HEAD_COUNT, STORED_KEY_COUNT, HEAD_SIZE],
        1.0,
        41,
    );
    let scale = 1.0 / (HEAD_SIZE as f32).sqrt();
    let attend = |rows: &MlxArray| {
        runtime
            .scaled_dot_product_attention(rows, &keys, &values, scale)
            .expect("the attention builds")
    };
    report_window_drift(
        &runtime,
        "full attention, 16 query heads over 2 key-value heads, 1,024 keys",
        &queries,
        2,
        &attend,
    );
}

/// Prints, for each window size, the largest difference between a row
/// computed inside the window and the same row computed alone. `row_axis`
/// is the axis that holds the rows in `inputs` and in the kernel's output.
fn report_window_drift(
    runtime: &MlxRuntime,
    label: &str,
    inputs: &MlxArray,
    row_axis: usize,
    kernel: &dyn Fn(&MlxArray) -> MlxArray,
) {
    let largest_window = WINDOW_ROW_COUNTS[WINDOW_ROW_COUNTS.len() - 1];
    let alone_rows: Vec<Vec<f32>> = (0..largest_window)
        .map(|row_index| {
            host_values(
                runtime,
                &kernel(&rows_of(runtime, inputs, row_axis, row_index, 1)),
            )
        })
        .collect();
    let mut summary = format!("[batched-kernel-row-exactness] {label}:");
    for window_row_count in WINDOW_ROW_COUNTS {
        let window_output = kernel(&rows_of(runtime, inputs, row_axis, 0, window_row_count));
        let mut largest_difference = 0.0_f32;
        let mut differing_row_count = 0;
        for row_index in 0..window_row_count {
            let window_row = host_values(
                runtime,
                &rows_of(runtime, &window_output, row_axis, row_index, 1),
            );
            let row_difference =
                largest_absolute_difference(&alone_rows[row_index as usize], &window_row);
            if row_difference > 0.0 {
                differing_row_count += 1;
            }
            largest_difference = largest_difference.max(row_difference);
        }
        summary.push_str(&format!(
            " | {window_row_count} rows: max diff {largest_difference:.5}, {differing_row_count} rows differ"
        ));
    }
    eprintln!("{summary}");
}

fn rows_of(
    runtime: &MlxRuntime,
    array: &MlxArray,
    row_axis: usize,
    first_row: i32,
    row_count: i32,
) -> MlxArray {
    let shape = array.shape();
    let mut start = vec![0; shape.len()];
    let mut stop = shape.to_vec();
    start[row_axis] = first_row;
    stop[row_axis] = first_row + row_count;
    let strides = vec![1; shape.len()];
    runtime
        .slice(array, &start, &stop, &strides)
        .expect("the rows slice")
}

fn normal(runtime: &MlxRuntime, shape: &[i32], standard_deviation: f32, seed: u64) -> MlxArray {
    let key = runtime.random_key(seed).expect("a random key");
    runtime
        .random_normal(shape, MlxDtype::BFloat16, 0.0, standard_deviation, &key)
        .expect("normal values")
}

fn host_values(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    let single_precision = runtime
        .astype(array, MlxDtype::Float32)
        .expect("values convert");
    runtime
        .array_to_vec_f32(&single_precision)
        .expect("values read back")
}

fn largest_absolute_difference(reference: &[f32], candidate: &[f32]) -> f32 {
    let mut largest = 0.0_f32;
    for (reference_value, candidate_value) in reference.iter().zip(candidate) {
        largest = largest.max((reference_value - candidate_value).abs());
    }
    largest
}

fn direct_mlx_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
        )
        .expect("the memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}
