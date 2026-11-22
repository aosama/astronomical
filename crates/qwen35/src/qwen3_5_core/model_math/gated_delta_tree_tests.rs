//! The tree recurrence against the sequence kernel it must agree with bit for
//! bit: a row's output equals the sequence kernel's last output over that
//! row's chain alone.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use super::gated_delta_sequence::{qwen3_5_gated_delta_kernel, qwen3_5_gated_delta_sequence};
use super::gated_delta_tree::{
    TreeRowChains, qwen3_5_gated_delta_replay, qwen3_5_gated_delta_replay_kernel,
    qwen3_5_gated_delta_tree, qwen3_5_gated_delta_tree_kernel,
};

const KEY_HEAD_COUNT: i32 = 2;
const VALUE_HEAD_COUNT: i32 = 4;
/// The kernels' lane layout reads 128-wide keys.
const KEY_HEAD_DIMENSION: i32 = 128;
const VALUE_HEAD_DIMENSION: i32 = 32;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the tree recurrence test memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}

/// One step's inputs for `row_count` rows, shaped as the layer passes them.
struct RecurrenceInputs {
    queries: MlxArray,
    keys: MlxArray,
    values: MlxArray,
    decays: MlxArray,
    update_rates: MlxArray,
    recurrent_state: MlxArray,
}

fn recurrence_inputs(runtime: &MlxRuntime, row_count: i32, seed: u64) -> RecurrenceInputs {
    let key = runtime.random_key(seed).expect("a key builds");
    let normal = |shape: &[i32], dtype: MlxDtype, scale: f32| {
        runtime
            .random_normal(shape, dtype, 0.0, scale, &key)
            .expect("random inputs build")
    };
    let unit_interval = |shape: &[i32]| {
        runtime
            .sigmoid(&normal(shape, MlxDtype::Float32, 1.0))
            .expect("rates build")
    };
    let key_shape = [1, row_count, KEY_HEAD_COUNT, KEY_HEAD_DIMENSION];
    RecurrenceInputs {
        queries: normal(&key_shape, MlxDtype::BFloat16, 0.1),
        keys: normal(&key_shape, MlxDtype::BFloat16, 0.1),
        values: normal(
            &[1, row_count, VALUE_HEAD_COUNT, VALUE_HEAD_DIMENSION],
            MlxDtype::BFloat16,
            1.0,
        ),
        decays: unit_interval(&[1, row_count, VALUE_HEAD_COUNT]),
        update_rates: unit_interval(&[1, row_count, VALUE_HEAD_COUNT]),
        recurrent_state: normal(
            &[
                1,
                VALUE_HEAD_COUNT,
                VALUE_HEAD_DIMENSION,
                KEY_HEAD_DIMENSION,
            ],
            MlxDtype::Float32,
            0.5,
        ),
    }
}

fn bits(runtime: &MlxRuntime, array: &MlxArray) -> Vec<u32> {
    let widened = runtime
        .astype(array, MlxDtype::Float32)
        .expect("the output widens");
    runtime
        .array_to_vec_f32(&widened)
        .expect("the output reads back")
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

/// Rows `rows` of every input, in order, along the row axis.
fn gathered(runtime: &MlxRuntime, inputs: &RecurrenceInputs, rows: &[i32]) -> RecurrenceInputs {
    let row_ids = runtime
        .array_from_i32(rows, &[rows.len() as i32])
        .expect("row ids build");
    let take = |array: &MlxArray| runtime.take_axis(array, &row_ids, 1).expect("rows gather");
    RecurrenceInputs {
        queries: take(&inputs.queries),
        keys: take(&inputs.keys),
        values: take(&inputs.values),
        decays: take(&inputs.decays),
        update_rates: take(&inputs.update_rates),
        recurrent_state: runtime
            .multiply_scalar(&inputs.recurrent_state, 1.0)
            .expect("the state copies"),
    }
}

fn tree_outputs(
    runtime: &MlxRuntime,
    inputs: &RecurrenceInputs,
    parents: &[Option<usize>],
) -> MlxArray {
    let tree_kernel = qwen3_5_gated_delta_tree_kernel().expect("the tree kernel declares");
    qwen3_5_gated_delta_tree(
        runtime,
        &tree_kernel,
        &inputs.queries,
        &inputs.keys,
        &inputs.values,
        &inputs.decays,
        &inputs.update_rates,
        &inputs.recurrent_state,
        &TreeRowChains::from_parent_rows(parents)
            .upload(runtime)
            .expect("the chains upload"),
    )
    .expect("the tree recurrence runs")
}

fn sequence_outputs(runtime: &MlxRuntime, inputs: &RecurrenceInputs) -> MlxArray {
    let sequence_kernel = qwen3_5_gated_delta_kernel().expect("the sequence kernel declares");
    qwen3_5_gated_delta_sequence(
        runtime,
        Some(&sequence_kernel),
        &inputs.queries,
        &inputs.keys,
        &inputs.values,
        &inputs.decays,
        &inputs.update_rates,
        &inputs.recurrent_state,
    )
    .expect("the sequence recurrence runs")
    .0
}

#[test]
#[serial]
fn should_match_the_sequence_kernel_bit_for_bit_on_a_chain() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_match_the_sequence_kernel_bit_for_bit_on_a_chain",
    ) {
        return;
    }
    let runtime = test_runtime();
    let inputs = recurrence_inputs(&runtime, 6, 3);
    let chain = [None, Some(0), Some(1), Some(2), Some(3), Some(4)];
    assert_eq!(
        bits(&runtime, &tree_outputs(&runtime, &inputs, &chain)),
        bits(&runtime, &sequence_outputs(&runtime, &inputs))
    );
}

#[test]
#[serial]
fn should_give_every_branch_row_what_its_chain_alone_would_give() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_give_every_branch_row_what_its_chain_alone_would_give",
    ) {
        return;
    }
    let runtime = test_runtime();
    let inputs = recurrence_inputs(&runtime, 7, 5);
    // Two branches under the anchor, one three deep, and a sibling pair.
    let parents = [None, Some(0), Some(0), Some(1), Some(3), Some(2), Some(3)];
    let tree = bits(&runtime, &tree_outputs(&runtime, &inputs, &parents));
    let row_width = (VALUE_HEAD_COUNT * VALUE_HEAD_DIMENSION) as usize;
    let chains: [&[i32]; 7] = [
        &[0],
        &[0, 1],
        &[0, 2],
        &[0, 1, 3],
        &[0, 1, 3, 4],
        &[0, 2, 5],
        &[0, 1, 3, 6],
    ];
    for (row, chain) in chains.iter().enumerate() {
        let alone = bits(
            &runtime,
            &sequence_outputs(&runtime, &gathered(&runtime, &inputs, chain)),
        );
        let last_output = &alone[(chain.len() - 1) * row_width..];
        assert_eq!(
            &tree[row * row_width..(row + 1) * row_width],
            last_output,
            "row {row}"
        );
    }
}

#[test]
#[serial]
fn should_give_a_copied_chain_below_a_branch_what_its_chain_alone_gives() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_give_a_copied_chain_below_a_branch_what_its_chain_alone_gives",
    ) {
        return;
    }
    let runtime = test_runtime();
    let inputs = recurrence_inputs(&runtime, 8, 7);
    // A small tree, then a copied run continuing row 3 for four rows.
    let parents = [
        None,
        Some(0),
        Some(0),
        Some(1),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
    ];
    let tree = bits(&runtime, &tree_outputs(&runtime, &inputs, &parents));
    let row_width = (VALUE_HEAD_COUNT * VALUE_HEAD_DIMENSION) as usize;
    let chains: [&[i32]; 3] = [&[0, 2], &[0, 1, 3, 4, 5], &[0, 1, 3, 4, 5, 6, 7]];
    for (row, chain) in [(2, chains[0]), (5, chains[1]), (7, chains[2])] {
        let alone = bits(
            &runtime,
            &sequence_outputs(&runtime, &gathered(&runtime, &inputs, chain)),
        );
        assert_eq!(
            &tree[row * row_width..(row + 1) * row_width],
            &alone[(chain.len() - 1) * row_width..],
            "row {row}"
        );
    }
}

#[test]
fn should_pad_every_chain_to_the_longest_and_list_it_from_the_anchor() {
    let chains = TreeRowChains::from_parent_rows(&[None, Some(0), Some(0), Some(1)]);
    // The longest chain (3) is padded to a power of two.
    assert_eq!(chains.maximum_chain, 4);
    assert_eq!(chains.chain_lengths, vec![1, 2, 2, 3]);
    assert_eq!(
        chains.row_chains,
        vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 3, 0]
    );
}

#[test]
#[serial]
fn should_replay_kept_rows_to_the_sequence_kernels_final_state_bit_for_bit() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_replay_kept_rows_to_the_sequence_kernels_final_state_bit_for_bit",
    ) {
        return;
    }
    let runtime = test_runtime();
    let inputs = recurrence_inputs(&runtime, 7, 11);
    let replay_kernel = qwen3_5_gated_delta_replay_kernel().expect("the replay kernel declares");
    let sequence_kernel = qwen3_5_gated_delta_kernel().expect("the sequence kernel declares");
    for kept in [vec![0], vec![0, 2, 5], vec![0, 1, 3, 4, 6]] {
        let kept_rows = runtime
            .array_from_i32(&kept, &[kept.len() as i32])
            .expect("kept rows build");
        let replayed = qwen3_5_gated_delta_replay(
            &runtime,
            &replay_kernel,
            &inputs.queries,
            &inputs.keys,
            &inputs.values,
            &inputs.decays,
            &inputs.update_rates,
            &inputs.recurrent_state,
            &kept_rows,
        )
        .expect("the replay runs");
        let alone = gathered(&runtime, &inputs, &kept);
        let (_outputs, expected_state) = qwen3_5_gated_delta_sequence(
            &runtime,
            Some(&sequence_kernel),
            &alone.queries,
            &alone.keys,
            &alone.values,
            &alone.decays,
            &alone.update_rates,
            &alone.recurrent_state,
        )
        .expect("the sequence recurrence runs");
        assert_eq!(
            bits(&runtime, &replayed),
            bits(&runtime, &expected_state),
            "kept rows {kept:?}"
        );
    }
}
