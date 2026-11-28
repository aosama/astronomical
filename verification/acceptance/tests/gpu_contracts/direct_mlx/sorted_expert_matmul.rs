//! Component contract of the sorted-expert 4-bit matmul against MLX's own
//! `gather_qmm`: the same bits, for every tile layout and for the token-row
//! variant, on a small layer whose routing reaches every awkward case at
//! once.
//!
//! The routing gives experts no rows, one row, a few rows, exactly a tile,
//! more than a 96-row pass (so a threadgroup makes two passes), and a total
//! that is no multiple of any tile, so the bound checks at an expert's end
//! and at the matrix's end both run.

use mlx_bridge::{MlxArray, MlxDtype};
use model_serving::sorted_expert_matmul::{
    SortedExpertLayout, SortedExpertMatmulKernels, SortedExpertProjectionError,
    sorted_expert_quantized_matmul_from_token_rows, sorted_expert_quantized_matmul_with_layout,
};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::row_batched_matmul::initialize_runtime;
use crate::sorted_expert_fixtures::{
    mismatched_values, mlx_sorted_gather_qmm, quantized_expert_stack, stable_expert_sort, stack_of,
    tiled_layout,
};

const OUTPUT_WIDTH: i32 = 256;
const INPUT_WIDTH: i32 = 128;
/// Rows per expert, in expert order: none, one, a few, a tile, two passes'
/// worth, none again, a partial tile, and the last expert's rows.
const ROWS_PER_EXPERT: [i32; 8] = [0, 1, 7, 64, 130, 0, 33, 20];
const EXPERT_COUNT: i32 = ROWS_PER_EXPERT.len() as i32;
const EXPERTS_A_TOKEN: i32 = 2;
const TOKEN_COUNT: i32 = 300;
const LAYOUTS: [SortedExpertLayout; 4] = [
    tiled_layout(4, true, 96),
    tiled_layout(4, true, 64),
    tiled_layout(2, false, 96),
    tiled_layout(2, false, 64),
];

// --- positive ---

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_give_mlx_gather_qmm_bits_for_every_layout_and_uneven_routing() {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(2_712).expect("a key builds");
    let kernels = SortedExpertMatmulKernels::declare().expect("the kernels declare");
    let experts = quantized_expert_stack(
        &runtime,
        &random_key,
        (EXPERT_COUNT, OUTPUT_WIDTH, INPUT_WIDTH),
        4,
    );
    let sorted_expert_ids = sorted_ids_from_counts(&runtime);
    let row_count: i32 = ROWS_PER_EXPERT.iter().sum();
    let sorted_rows = runtime
        .random_normal(
            &[row_count, 1, INPUT_WIDTH],
            MlxDtype::BFloat16,
            0.0,
            1.0,
            &random_key,
        )
        .expect("rows are drawn");
    let mlx_output =
        mlx_sorted_gather_qmm(&runtime, &sorted_rows, &experts, (&sorted_expert_ids, 4));
    for layout in LAYOUTS {
        let astronomical_output = sorted_expert_quantized_matmul_with_layout(
            &runtime,
            &kernels,
            (&sorted_rows, &sorted_expert_ids),
            &stack_of(&experts),
            layout,
        )
        .expect("the kernel projects");
        assert_eq!(
            mismatched_values(&runtime, &mlx_output, &astronomical_output),
            0,
            "{layout:?} gave different bits from MLX"
        );
    }
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_give_mlx_bits_reading_rows_straight_from_the_token_rows() {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(2_713).expect("a key builds");
    let kernels = SortedExpertMatmulKernels::declare().expect("the kernels declare");
    let experts = quantized_expert_stack(
        &runtime,
        &random_key,
        (EXPERT_COUNT, OUTPUT_WIDTH, INPUT_WIDTH),
        4,
    );
    let token_rows = runtime
        .random_normal(
            &[TOKEN_COUNT, 1, INPUT_WIDTH],
            MlxDtype::BFloat16,
            0.0,
            1.0,
            &random_key,
        )
        .expect("token rows are drawn");
    let (sorted_order, sorted_token_indices, sorted_expert_ids) = token_routing(&runtime);
    let sorted_rows = runtime
        .take_axis(&token_rows, &sorted_token_indices, 0)
        .expect("the sorted copy is taken");
    let mlx_output =
        mlx_sorted_gather_qmm(&runtime, &sorted_rows, &experts, (&sorted_expert_ids, 4));
    let astronomical_output = sorted_expert_quantized_matmul_from_token_rows(
        &runtime,
        &kernels,
        (&token_rows, &sorted_order, &sorted_expert_ids),
        EXPERTS_A_TOKEN,
        &stack_of(&experts),
    )
    .expect("the token-row kernel projects");
    assert_eq!(
        mismatched_values(&runtime, &mlx_output, &astronomical_output),
        0
    );
}

// --- negative ---

#[test]
#[ignore = "runs MLX on the GPU"]
#[serial]
fn should_refuse_an_eight_bit_stack_instead_of_misreading_it() {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(2_714).expect("a key builds");
    let kernels = SortedExpertMatmulKernels::declare().expect("the kernels declare");
    let eight_bit_experts = quantized_expert_stack(
        &runtime,
        &random_key,
        (EXPERT_COUNT, OUTPUT_WIDTH, INPUT_WIDTH),
        8,
    );
    let rows = runtime
        .random_normal(
            &[4, 1, INPUT_WIDTH],
            MlxDtype::BFloat16,
            0.0,
            1.0,
            &random_key,
        )
        .expect("rows are drawn");
    let ids = runtime
        .array_from_i32(&[0, 1, 2, 3], &[4])
        .expect("ids upload");
    let refusal = sorted_expert_quantized_matmul_with_layout(
        &runtime,
        &kernels,
        (&rows, &ids),
        &stack_of(&eight_bit_experts),
        LAYOUTS[0],
    );
    assert!(matches!(
        refusal,
        Err(SortedExpertProjectionError::UnsupportedQuantization { .. })
    ));
}

#[test]
#[ignore = "runs MLX on the GPU"]
#[serial]
fn should_refuse_rows_of_the_wrong_width() {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(2_715).expect("a key builds");
    let kernels = SortedExpertMatmulKernels::declare().expect("the kernels declare");
    let experts = quantized_expert_stack(
        &runtime,
        &random_key,
        (EXPERT_COUNT, OUTPUT_WIDTH, INPUT_WIDTH),
        4,
    );
    let narrow_rows = runtime
        .random_normal(
            &[4, 1, INPUT_WIDTH / 2],
            MlxDtype::BFloat16,
            0.0,
            1.0,
            &random_key,
        )
        .expect("rows are drawn");
    let ids = runtime
        .array_from_i32(&[0, 1, 2, 3], &[4])
        .expect("ids upload");
    assert!(
        sorted_expert_quantized_matmul_with_layout(
            &runtime,
            &kernels,
            (&narrow_rows, &ids),
            &stack_of(&experts),
            LAYOUTS[0],
        )
        .is_err()
    );
}

fn sorted_ids_from_counts(runtime: &MlxRuntime) -> MlxArray {
    let mut expert_ids = Vec::new();
    for (expert, row_count) in ROWS_PER_EXPERT.iter().enumerate() {
        expert_ids.extend(std::iter::repeat_n(expert as i32, *row_count as usize));
    }
    runtime
        .array_from_i32(&expert_ids, &[expert_ids.len() as i32])
        .expect("ids upload")
}

/// Each token's two distinct experts from a fixed stream over the eight,
/// sorted stably as the engine's assignment sort does: (sorted order, each
/// sorted assignment's token, its expert).
fn token_routing(runtime: &MlxRuntime) -> (MlxArray, MlxArray, MlxArray) {
    let mut flat_expert_ids = Vec::with_capacity((TOKEN_COUNT * EXPERTS_A_TOKEN) as usize);
    for token_index in 0..TOKEN_COUNT {
        let first_expert = (token_index * 5 + token_index / 7) % EXPERT_COUNT;
        flat_expert_ids.push(first_expert);
        flat_expert_ids.push((first_expert + 1 + token_index % 3) % EXPERT_COUNT);
    }
    let (sorted_order, sorted_tokens, sorted_experts) =
        stable_expert_sort(&flat_expert_ids, EXPERTS_A_TOKEN);
    let shape = [sorted_order.len() as i32];
    (
        runtime
            .array_from_i32(&sorted_order, &shape)
            .expect("order uploads"),
        runtime
            .array_from_i32(&sorted_tokens, &shape)
            .expect("tokens upload"),
        runtime
            .array_from_i32(&sorted_experts, &shape)
            .expect("experts upload"),
    )
}
