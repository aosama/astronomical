//! Whether converting a kept layer to register order costs the rows that are
//! not a draft tree (issues #2585 and #2568): one reply token, and a prompt
//! chunk.
//!
//! Three ways to multiply the dense Qwen3.8 27B's 4-bit projection shapes,
//! 16 distinct copies each so every pass reads DRAM:
//! - stock: MLX's quantized matmul on MLX's layout;
//! - register: the register-ordered kernel on converted weights, a chunk
//!   going 128 rows a dispatch;
//! - restored: converted weights restored to MLX's layout for the pass and
//!   multiplied by MLX's kernel, bit for bit the stock result.
//!
//! The paths are alternated round by round and each one's first round is
//! dropped, so none is timed cold. Asserts nothing; prints a median and a
//! range for each. First reading (2026-11-26): register order costs 5 to 20%
//! at one row, a fixed 6 to 10 microseconds a projection, and 4 to 33% on a
//! 2,048-row chunk; restoring for the chunk costs 0 to 7%, and three times
//! the stock cost at one row.
//!
//! Run with `cargo measure-register-order-served-rows`. No model; about a
//! minute and a half.

use std::time::{Duration, Instant};

use mlx_bridge::MlxArray;
use model_serving::row_batched_matmul::RowBatchedMatmulKernels;
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::k_row_verify_cost::{
    MEASURED_PASS_COUNT, MatmulPath, MeasuredPass, MeasuredPassTimes, PROJECTION_SHAPES,
    PackedProjection, WARM_UP_PASS_COUNT, distinct_shape_copies, initialize_runtime,
    median_pass_time, random_rows, shape_copies_pass,
};

/// One reply token and a resident prompt chunk, each with its rounds of
/// alternated timing: ten at one row, where a pass takes milliseconds and a
/// range is cheap, and two at a prompt chunk, where a round is already the
/// median of five passes of about 0.2 s. A streamed span of 4,096 rows cost
/// twice a 2,048-row chunk on both kernels, so it is not timed.
const SERVED_ROW_COUNTS_AND_ROUNDS: [(i32, usize); 2] = [(1, 10), (2_048, 2)];

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-register-order-served-rows"]
#[serial]
fn should_compare_stock_register_ordered_and_restored_layouts_at_served_row_counts() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(1234).expect("a key builds");
    for projection_shape in PROJECTION_SHAPES {
        let shape_copies = distinct_shape_copies(&runtime, &kernels, &random_key, projection_shape);
        let served_shape = ServedShape {
            runtime: &runtime,
            random_key: &random_key,
            shape_copies: &shape_copies,
            input_width: projection_shape.1,
        };
        for (row_count, round_count) in SERVED_ROW_COUNTS_AND_ROUNDS {
            let mut stock_times = Vec::with_capacity(round_count);
            let mut register_times = Vec::with_capacity(round_count);
            let mut restored_times = Vec::with_capacity(round_count);
            for round_index in 0..=round_count {
                let stock_time = served_shape.median_time(row_count, MatmulPath::Mlx);
                let register_time =
                    served_shape.median_time(row_count, MatmulPath::RegisterOrdered(&kernels));
                let restored_time = served_shape.median_restored_time(row_count);
                if round_index > 0 {
                    stock_times.push(stock_time);
                    register_times.push(register_time);
                    restored_times.push(restored_time);
                }
            }
            eprintln!(
                "[served-rows] {}x{} K={row_count}: stock {} | register {} | restored {}",
                projection_shape.0,
                projection_shape.1,
                spread_text(&mut stock_times),
                spread_text(&mut register_times),
                spread_text(&mut restored_times)
            );
        }
    }
}

/// One projection shape's distinct copies and what times a pass over them.
struct ServedShape<'fixture> {
    runtime: &'fixture MlxRuntime,
    random_key: &'fixture MlxArray,
    shape_copies: &'fixture [PackedProjection],
    input_width: i32,
}

impl ServedShape<'_> {
    fn median_time(&self, row_count: i32, matmul_path: MatmulPath<'_>) -> Duration {
        median_pass_time(
            self.runtime,
            self.random_key,
            row_count,
            &MeasuredPass::ShapeCopies(self.shape_copies, self.input_width),
            matmul_path,
        )
    }

    fn median_restored_time(&self, row_count: i32) -> Duration {
        let rows = random_rows(self.runtime, self.random_key, row_count, self.input_width);
        median_restored_pass_time(self.runtime, &rows, self.shape_copies)
    }
}

/// The median time of `rows` through every copy with its weight restored
/// to MLX's layout inside the pass: the restorations are built again for
/// each repetition, so each pays them as a served chunk would, and the
/// restored weights then take the stock path's own pass.
fn median_restored_pass_time(
    runtime: &MlxRuntime,
    rows: &MlxArray,
    shape_copies: &[PackedProjection],
) -> Duration {
    runtime
        .evaluate_arrays(&[rows])
        .expect("the rows materialize");
    let mut pass_times = MeasuredPassTimes::new();
    for repetition_index in 0..WARM_UP_PASS_COUNT + MEASURED_PASS_COUNT {
        let repetition_started_at = Instant::now();
        let mut restored_copies = Vec::with_capacity(shape_copies.len());
        for shape_copy in shape_copies {
            let restored = shape_copy
                .register_ordered
                .as_ref()
                .expect("every 27B shape tiles")
                .restored_mlx_layout(runtime)
                .expect("the weight restores");
            restored_copies.push(PackedProjection {
                weight: restored.packed_weight,
                scales: restored.scales,
                biases: restored.biases,
                register_ordered: None,
                stored_bytes: shape_copy.stored_bytes,
            });
        }
        let restored_output = shape_copies_pass(runtime, MatmulPath::Mlx, &restored_copies, rows);
        pass_times.keep_evaluated(
            runtime,
            &restored_output,
            repetition_index,
            repetition_started_at,
        );
    }
    pass_times.median()
}

/// "median 2.53 ms (2.41 to 2.70)" for 16 copies of a shape.
fn spread_text(round_times: &mut [Duration]) -> String {
    round_times.sort();
    format!(
        "median {:.2} ms ({:.2} to {:.2})",
        milliseconds(round_times[round_times.len() / 2]),
        milliseconds(round_times[0]),
        milliseconds(round_times[round_times.len() - 1])
    )
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
