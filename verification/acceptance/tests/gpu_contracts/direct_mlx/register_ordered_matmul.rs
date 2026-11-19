//! Component contract of the register-ordered 4-bit matmul: its one-time
//! conversion against a CPU transcription of TensorFold's `regBlock`, its
//! values against MLX's quantized matmul, a row's bits at every row count,
//! and the restoration back to MLX's layout bit for bit.

use mlx_bridge::{MlxArray, MlxDtype};
use model_serving::row_batched_matmul::{
    RegisterOrderedWeight, RowBatchedMatmulKernels, register_ordered_matmul,
};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::row_batched_matmul::{
    MAXIMUM_DIFFERENCE_OVER_RMS, MINIMUM_COSINE_SIMILARITY, OutputDifference, PROJECTION_SHAPES,
    ShapeFixture, cosine_similarity, initialize_runtime, shape_fixture, values,
};

/// One row, a count inside a block, one block, past it, and a whole dispatch.
const ROW_COUNTS: [i32; 6] = [1, 5, 16, 17, 64, 128];

fn converted(
    runtime: &MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
    fixture: &ShapeFixture,
) -> RegisterOrderedWeight {
    let weight = RegisterOrderedWeight::convert(
        runtime,
        kernels,
        &fixture.packed_weight,
        &fixture.scales,
        &fixture.biases,
    )
    .expect("the weight converts");
    runtime
        .evaluate_arrays(&[&weight.codes, &weight.scale_bias_words])
        .expect("the converted weight materializes");
    weight
}

/// TensorFold's `regBlock` (zig/src/core/lane_projection.zig), transcribed
/// from the Zig rather than from this crate's Metal so the two are checked
/// against each other: one 32-column x 64-code block in tile order (column
/// `c`'s eight words at `8 c`) into the tensor op's register order.
fn register_block(tile_order_block: &[u32; 256]) -> [u32; 256] {
    let mut register_ordered = [0_u32; 256];
    for (word_position, ordered_word) in register_ordered.iter_mut().enumerate() {
        let lane = (word_position >> 2) & 31;
        let word_slot = 4 * (word_position >> 7) + (word_position & 3);
        let at = (((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (word_slot >> 1)) * 8
            + ((lane >> 3) & 1)
            + 4 * (word_slot & 1);
        let shift = 16 * (lane & 1);
        let a = (tile_order_block[at] >> shift) & 0xffff;
        let c = (tile_order_block[at + 2] >> shift) & 0xffff;
        *ordered_word = (a & 0xf)
            | ((a >> 4) & 0xf0)
            | ((c & 0xf) << 8)
            | ((c << 4) & 0xf000)
            | (((a >> 4) & 0xf) << 16)
            | (((a >> 8) & 0xf0) << 16)
            | (((c >> 4) & 0xf) << 24)
            | (((c >> 8) & 0xf0) << 24);
    }
    register_ordered
}

fn bf16_bits(runtime: &MlxRuntime, array: &MlxArray) -> Vec<u16> {
    runtime
        .view(array, MlxDtype::UInt16)
        .expect("bf16 views as uint16")
        .to_vec_u16()
        .expect("bits read back")
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_convert_every_block_and_scale_bias_word_exactly_as_tensorfold_orders_them() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(19).expect("a key builds");
    let (output_width, input_width) = (96, 640);
    let fixture = shape_fixture(&runtime, &random_key, output_width, input_width);
    let weight = converted(&runtime, &kernels, &fixture);
    let mlx_words = fixture.packed_weight.to_vec_u32().expect("words read back");
    let converted_codes = weight.codes.to_vec_u32().expect("codes read back");
    let group_count = (input_width / 64) as usize;
    let row_words = (input_width / 8) as usize;
    for tile in 0..(output_width / 32) as usize {
        for group in 0..group_count {
            let mut tile_order_block = [0_u32; 256];
            for column in 0..32 {
                let source = (tile * 32 + column) * row_words + group * 8;
                tile_order_block[column * 8..column * 8 + 8]
                    .copy_from_slice(&mlx_words[source..source + 8]);
            }
            let block_start = (tile * group_count + group) * 256;
            assert_eq!(
                &converted_codes[block_start..block_start + 256],
                register_block(&tile_order_block).as_slice(),
                "tile {tile} group {group}"
            );
        }
    }
    let scale_bits = bf16_bits(&runtime, &fixture.scales);
    let bias_bits = bf16_bits(&runtime, &fixture.biases);
    let words = weight
        .scale_bias_words
        .to_vec_u32()
        .expect("words read back");
    for group in 0..group_count {
        for column in 0..output_width as usize {
            let stored = column * group_count + group;
            let expected = u32::from(scale_bits[stored]) | (u32::from(bias_bits[stored]) << 16);
            assert_eq!(words[group * output_width as usize + column], expected);
        }
    }
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_match_mlx_quantized_matmul_for_every_27b_shape_and_row_count() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(23).expect("a key builds");
    for (output_width, input_width) in PROJECTION_SHAPES {
        let fixture = shape_fixture(&runtime, &random_key, output_width, input_width);
        let weight = converted(&runtime, &kernels, &fixture);
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
            let kernel_values = values(
                &runtime,
                &register_ordered_matmul(&runtime, &kernels, &rows, &weight)
                    .expect("the kernel runs"),
            );
            let mlx_values = values(
                &runtime,
                &runtime
                    .quantized_matmul_affine(
                        &rows,
                        &fixture.packed_weight,
                        &fixture.scales,
                        &fixture.biases,
                        true,
                        64,
                        4,
                    )
                    .expect("MLX's matmul builds"),
            );
            let similarity = cosine_similarity(&kernel_values, &mlx_values);
            let difference = OutputDifference::between(&kernel_values, &mlx_values);
            eprintln!(
                "[register-ordered] {output_width}x{input_width} K={row_count}: cosine {similarity:.7}, \
                 {} of {} values differ, largest difference {:.4} of the RMS",
                difference.differing_value_count,
                kernel_values.len(),
                difference.largest_difference_over_rms
            );
            assert!(similarity >= MINIMUM_COSINE_SIMILARITY);
            assert!(difference.largest_difference_over_rms <= MAXIMUM_DIFFERENCE_OVER_RMS);
        }
    }
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_give_each_row_the_same_bits_alone_and_inside_a_full_dispatch() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(29).expect("a key builds");
    for (output_width, input_width) in PROJECTION_SHAPES {
        let fixture = shape_fixture(&runtime, &random_key, output_width, input_width);
        let weight = converted(&runtime, &kernels, &fixture);
        let rows = runtime
            .random_normal(
                &[128, input_width],
                MlxDtype::BFloat16,
                0.0,
                1.0,
                &random_key,
            )
            .expect("rows build");
        let block_values = values(
            &runtime,
            &register_ordered_matmul(&runtime, &kernels, &rows, &weight).expect("rows run"),
        );
        for row_index in [0, 7, 15, 16, 127] {
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
                &register_ordered_matmul(&runtime, &kernels, &single_row, &weight)
                    .expect("one row runs"),
            );
            let width = output_width as usize;
            let start = row_index as usize * width;
            assert_eq!(
                &block_values[start..start + width],
                alone_values.as_slice(),
                "{output_width}x{input_width} row {row_index}"
            );
        }
    }
}

fn taken_rows(runtime: &MlxRuntime, array: &MlxArray, row_id_array: &MlxArray) -> MlxArray {
    runtime
        .take_axis(array, row_id_array, 0)
        .expect("rows are taken")
}

fn words_of(runtime: &MlxRuntime, array: &MlxArray) -> Vec<u32> {
    runtime
        .evaluate_arrays(&[array])
        .expect("the array materializes");
    array.to_vec_u32().expect("words read back")
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_restore_mlx_layout_bit_for_bit_from_a_converted_weight() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(31).expect("a key builds");
    let fixture = shape_fixture(&runtime, &random_key, 1024, 5120);
    let weight = converted(&runtime, &kernels, &fixture);
    let restored = weight
        .restored_mlx_layout(&runtime)
        .expect("the weight restores");
    assert_eq!(
        words_of(&runtime, &restored.packed_weight),
        words_of(&runtime, &fixture.packed_weight)
    );
    assert_eq!(
        bf16_bits(&runtime, &restored.scales),
        bf16_bits(&runtime, &fixture.scales)
    );
    assert_eq!(
        bf16_bits(&runtime, &restored.biases),
        bf16_bits(&runtime, &fixture.biases)
    );
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_restore_scattered_rows_as_taking_them_from_the_original() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(37).expect("a key builds");
    let fixture = shape_fixture(&runtime, &random_key, 1024, 5120);
    let weight = converted(&runtime, &kernels, &fixture);
    // Rows in different tiles, a repeat, out of order, and both tile edges.
    let row_ids = [1023, 0, 31, 32, 500, 31];
    let restored = weight
        .restored_rows(&runtime, &row_ids)
        .expect("the rows restore");
    let row_id_array = runtime
        .array_from_i32(&row_ids, &[row_ids.len() as i32])
        .expect("row ids build");
    assert_eq!(
        words_of(&runtime, &restored.packed_weight),
        words_of(
            &runtime,
            &taken_rows(&runtime, &fixture.packed_weight, &row_id_array)
        )
    );
    assert_eq!(
        bf16_bits(&runtime, &restored.scales),
        bf16_bits(
            &runtime,
            &taken_rows(&runtime, &fixture.scales, &row_id_array)
        )
    );
    assert_eq!(
        bf16_bits(&runtime, &restored.biases),
        bf16_bits(
            &runtime,
            &taken_rows(&runtime, &fixture.biases, &row_id_array)
        )
    );
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_refuse_to_restore_a_row_outside_the_weight() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(41).expect("a key builds");
    let fixture = shape_fixture(&runtime, &random_key, 64, 256);
    let weight = converted(&runtime, &kernels, &fixture);
    assert!(weight.restored_rows(&runtime, &[64]).is_err());
    assert!(weight.restored_rows(&runtime, &[-1]).is_err());
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_pad_a_48_wide_projection_to_a_whole_tile_and_return_only_its_48_columns() {
    let runtime = initialize_runtime();
    let kernels = RowBatchedMatmulKernels::declare().expect("the kernels are declared");
    let random_key = runtime.random_key(43).expect("a key builds");
    let fixture = shape_fixture(&runtime, &random_key, 48, 5120);
    let weight = converted(&runtime, &kernels, &fixture);
    assert_eq!((weight.output_width, weight.logical_output_width), (64, 48));
    let rows = runtime
        .random_normal(&[16, 5120], MlxDtype::BFloat16, 0.0, 1.0, &random_key)
        .expect("rows build");
    let kernel_output =
        register_ordered_matmul(&runtime, &kernels, &rows, &weight).expect("the kernel runs");
    assert_eq!(kernel_output.shape(), vec![16, 48]);
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
    let similarity = cosine_similarity(
        &values(&runtime, &kernel_output),
        &values(&runtime, &mlx_output),
    );
    assert!(
        similarity >= MINIMUM_COSINE_SIMILARITY,
        "cosine {similarity}"
    );
    let restored = weight
        .restored_mlx_layout(&runtime)
        .expect("the weight restores");
    assert_eq!(
        words_of(&runtime, &restored.packed_weight),
        words_of(&runtime, &fixture.packed_weight)
    );
    assert_eq!(
        bf16_bits(&runtime, &restored.scales),
        bf16_bits(&runtime, &fixture.scales)
    );
}
