//! Component contract of the tiled sorted-expert multiplies against MLX:
//! the gate/up kernel's SwiGLU against MLX's two sorted `gather_qmm`s on the
//! sorted copy of the rows followed by MLX's compiled SwiGLU, and the plain
//! kernel against MLX's sorted `gather_qmm`, bit for bit.
//!
//! The small routing gives experts no rows, one row, a few, exactly a 64-row
//! tile, more than one 96-row tile, none again, a partial tile and a last
//! expert, so tile cuts at an expert's end and at the matrix's end both run;
//! a second case uses a 35B-A3B layer's real widths and expert count.

use mlx_bridge::{MlxArray, MlxCompiledSwiGlu};
use model_serving::sorted_expert_matmul::{
    SortedExpertProjectionError, TiledExpertKernels, sorted_expert_swiglu_from_token_rows,
    tiled_sorted_expert_matmul,
};
use serial_test::serial;

use super::row_batched_matmul::initialize_runtime;
use crate::sorted_expert_fixtures::{
    mismatched_values, mlx_sorted_gather_qmm, quantized_expert_stack, stack_of,
    standard_normal_rows, uploaded_stable_routing,
};

/// Experts each token picks, in the small routing.
const SMALL_EXPERTS_A_TOKEN: i32 = 2;
/// Rows per expert the small routing aims for, in expert order.
const SMALL_ROWS_PER_EXPERT: [i32; 8] = [0, 1, 7, 64, 130, 0, 33, 25];

/// Widths of one routed-expert layer: (experts, hidden, expert width).
#[derive(Clone, Copy)]
struct LayerWidths {
    expert_count: i32,
    hidden_width: i32,
    expert_width: i32,
}

const SMALL_LAYER: LayerWidths = LayerWidths {
    expert_count: SMALL_ROWS_PER_EXPERT.len() as i32,
    hidden_width: 128,
    expert_width: 64,
};

const ORNITH_35B_A3B_LAYER: LayerWidths = LayerWidths {
    expert_count: 256,
    hidden_width: 2_048,
    expert_width: 512,
};

// --- positive ---

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_give_mlx_bits_for_gate_up_swiglu_and_down_on_uneven_routing() {
    let flat_expert_ids = small_flat_expert_ids();
    assert_layer_matches_mlx(SMALL_LAYER, &flat_expert_ids, SMALL_EXPERTS_A_TOKEN, 2_716);
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_give_mlx_bits_at_a_35b_a3b_layer_over_a_2048_token_chunk() {
    let flat_expert_ids = skewed_flat_expert_ids(2_048, 8, ORNITH_35B_A3B_LAYER.expert_count);
    assert_layer_matches_mlx(ORNITH_35B_A3B_LAYER, &flat_expert_ids, 8, 2_717);
}

#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_give_mlx_bits_at_a_35b_a3b_layer_over_an_8192_token_chunk() {
    let flat_expert_ids = skewed_flat_expert_ids(8_192, 8, ORNITH_35B_A3B_LAYER.expert_count);
    assert_layer_matches_mlx(ORNITH_35B_A3B_LAYER, &flat_expert_ids, 8, 2_722);
}

// --- edge ---

/// Every row on one expert and the others empty: the tile list holds that
/// expert's tiles only, the last of them partial. (A handful of rows is no
/// case for this kernel: serving takes it from 32 rows an expert on
/// average, and MLX itself answers a few rows with its vector kernel, whose
/// sums run in another order.)
#[test]
#[ignore = "runs custom Metal kernels on the GPU"]
#[serial]
fn should_give_mlx_bits_when_every_row_goes_to_one_expert() {
    assert_layer_matches_mlx(SMALL_LAYER, &[5; 300], 1, 2_718);
}

// --- negative ---

#[test]
#[ignore = "runs MLX on the GPU"]
#[serial]
fn should_refuse_an_output_width_the_tiles_do_not_cover() {
    assert!(project_four_rows((8, 96, 128), 4, 2_720).is_err());
}

#[test]
#[ignore = "runs MLX on the GPU"]
#[serial]
fn should_refuse_an_eight_bit_stack_instead_of_misreading_it() {
    assert!(matches!(
        project_four_rows((8, 128, 128), 8, 2_719),
        Err(SortedExpertProjectionError::UnsupportedQuantization { .. })
    ));
}

/// Four rows, one an expert, through the tiled kernel and a `bits` stack of
/// `(experts, output, input)`.
fn project_four_rows(
    stack_shape: (i32, i32, i32),
    bits: i32,
    seed: u64,
) -> Result<MlxArray, SortedExpertProjectionError> {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(seed).expect("a key builds");
    let kernels = TiledExpertKernels::declare().expect("the kernels declare");
    let experts = quantized_expert_stack(&runtime, &random_key, stack_shape, bits);
    let rows = standard_normal_rows(&runtime, &random_key, (4, stack_shape.2));
    let expert_ids = runtime
        .array_from_i32(&[0, 1, 2, 3], &[4])
        .expect("the expert ids upload");
    tiled_sorted_expert_matmul(&runtime, &kernels, &rows, &stack_of(&experts), &expert_ids)
}

/// Draws a layer's gate, up and down stacks and token rows, routes the
/// tokens by `flat_expert_ids`, and asserts that both kernels give MLX's bits.
fn assert_layer_matches_mlx(
    widths: LayerWidths,
    flat_expert_ids: &[i32],
    experts_per_token: i32,
    seed: u64,
) {
    let runtime = initialize_runtime();
    let random_key = runtime.random_key(seed).expect("a key builds");
    let kernels = TiledExpertKernels::declare().expect("the kernels declare");
    let gate_up_shape = (
        widths.expert_count,
        widths.expert_width,
        widths.hidden_width,
    );
    let gate = quantized_expert_stack(&runtime, &random_key, gate_up_shape, 4);
    let up_key = runtime.random_key(seed + 1).expect("a key builds");
    let up = quantized_expert_stack(&runtime, &up_key, gate_up_shape, 4);
    let down_key = runtime.random_key(seed + 2).expect("a key builds");
    let down_shape = (
        widths.expert_count,
        widths.hidden_width,
        widths.expert_width,
    );
    let down = quantized_expert_stack(&runtime, &down_key, down_shape, 4);
    let token_count = flat_expert_ids.len() as i32 / experts_per_token;
    let token_rows =
        standard_normal_rows(&runtime, &random_key, (token_count, widths.hidden_width));
    let (sorted_order, sorted_token_indices, sorted_expert_ids) =
        uploaded_stable_routing(&runtime, flat_expert_ids, experts_per_token);
    let sorted_rows = runtime
        .take_axis(&token_rows, &sorted_token_indices, 0)
        .expect("the sorted copy is taken");

    let mlx_gate = mlx_sorted_gather_qmm(&runtime, &sorted_rows, &gate, &sorted_expert_ids);
    let mlx_up = mlx_sorted_gather_qmm(&runtime, &sorted_rows, &up, &sorted_expert_ids);
    let compiled_swiglu = MlxCompiledSwiGlu::new().expect("the SwiGLU compiles");
    let mlx_activated = runtime
        .apply_compiled_swiglu(&compiled_swiglu, &mlx_gate, &mlx_up)
        .expect("MLX's SwiGLU runs");
    let astronomical_activated = sorted_expert_swiglu_from_token_rows(
        &runtime,
        &kernels,
        (&token_rows, &sorted_order, &sorted_expert_ids),
        experts_per_token,
        (&stack_of(&gate), &stack_of(&up)),
    )
    .expect("the gate/up kernel projects");
    assert_eq!(
        mismatched_values(&runtime, &mlx_activated, &astronomical_activated),
        0,
        "the gate/up SwiGLU differs from MLX's"
    );

    let mlx_down = mlx_sorted_gather_qmm(&runtime, &mlx_activated, &down, &sorted_expert_ids);
    let astronomical_down = tiled_sorted_expert_matmul(
        &runtime,
        &kernels,
        &mlx_activated,
        &stack_of(&down),
        &sorted_expert_ids,
    )
    .expect("the tiled kernel projects");
    assert_eq!(
        mismatched_values(&runtime, &mlx_down, &astronomical_down),
        0,
        "the down projection differs from MLX's"
    );
}

/// The small routing: token t's experts walk the eight in an order that
/// gives each expert about its row count in `SMALL_ROWS_PER_EXPERT`.
fn small_flat_expert_ids() -> Vec<i32> {
    let mut assignments_left = SMALL_ROWS_PER_EXPERT.to_vec();
    let total: i32 = assignments_left.iter().sum();
    let mut flat_expert_ids = Vec::with_capacity(total as usize);
    while flat_expert_ids.len() < (total - total % SMALL_EXPERTS_A_TOKEN) as usize {
        let mut picked_this_token = Vec::new();
        for (expert, left) in assignments_left.iter_mut().enumerate().rev() {
            if *left > 0 && picked_this_token.len() < SMALL_EXPERTS_A_TOKEN as usize {
                *left -= 1;
                picked_this_token.push(expert as i32);
            }
        }
        if picked_this_token.len() < SMALL_EXPERTS_A_TOKEN as usize {
            break;
        }
        flat_expert_ids.extend(picked_this_token);
    }
    flat_expert_ids
}

/// Distinct experts for every token, favouring low expert numbers as a real
/// router favours some experts: a fixed arithmetic stream, so the routing is
/// the same on every run.
fn skewed_flat_expert_ids(token_count: i32, experts_per_token: i32, expert_count: i32) -> Vec<i32> {
    let mut flat_expert_ids = Vec::with_capacity((token_count * experts_per_token) as usize);
    for token in 0..token_count {
        let mut picked: Vec<i32> = Vec::new();
        let mut draw = i64::from(token) * 7_919 + 13;
        while picked.len() < experts_per_token as usize {
            draw = (draw * 1_103 + 12_345) % 65_521;
            let expert = ((draw % i64::from(expert_count)) * (draw % 3 + 1) / 3) as i32;
            if !picked.contains(&expert) {
                picked.push(expert);
            }
        }
        flat_expert_ids.extend(picked);
    }
    flat_expert_ids
}
