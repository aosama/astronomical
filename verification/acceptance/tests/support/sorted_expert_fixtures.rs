//! Fixtures the sorted-expert matmul's contract and measurements share: a
//! quantized expert stack, a tile layout, the engine's stable sort of routed assignments,
//! and a bit-for-bit comparison of two outputs.

use mlx_bridge::{MlxArray, MlxDtype};
use model_serving::sorted_expert_matmul::SortedExpertLayout;
use runtime_integration::MlxRuntime;

/// MLX's quantization group, the only one the kernel reads.
pub const QUANTIZATION_GROUP_SIZE: i32 = 64;

/// A stack of random bf16 experts `[experts, output, input]`, quantized by
/// MLX at `bits` in groups of 64 and evaluated: packed weights, scales and
/// biases, the scales and biases bf16 as a bf16 model's are.
pub fn quantized_expert_stack(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    (expert_count, output_width, input_width): (i32, i32, i32),
    bits: i32,
) -> (MlxArray, MlxArray, MlxArray) {
    let weights = runtime
        .random_normal(
            &[expert_count, output_width, input_width],
            MlxDtype::BFloat16,
            0.0,
            0.02,
            random_key,
        )
        .expect("the weights are drawn");
    let (packed_weight, scales, biases) = runtime
        .quantize_affine(&weights, QUANTIZATION_GROUP_SIZE, bits)
        .expect("the experts quantize");
    runtime
        .evaluate_arrays(&[&packed_weight, &scales, &biases])
        .expect("the quantized experts evaluate");
    (packed_weight, scales, biases)
}

/// A tile layout of the sorted-expert matmul, always splitting experts into
/// row tiles as serving does.
pub const fn tiled_layout(
    simdgroups_across_columns: i32,
    stages_rows: bool,
    rows_per_pass: i32,
) -> SortedExpertLayout {
    SortedExpertLayout {
        simdgroups_across_columns,
        stages_rows,
        rows_per_pass,
        splits_experts_into_tiles: true,
    }
}

/// The engine's assignment sort on the host: the stable sort order of the
/// flat expert ids, each sorted assignment's token, and its expert.
pub fn stable_expert_sort(
    flat_expert_ids: &[i32],
    experts_per_token: i32,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut sorted_order: Vec<i32> = (0..flat_expert_ids.len() as i32).collect();
    sorted_order.sort_by_key(|assignment| flat_expert_ids[*assignment as usize]);
    let sorted_tokens = sorted_order
        .iter()
        .map(|assignment| assignment / experts_per_token)
        .collect();
    let sorted_experts = sorted_order
        .iter()
        .map(|assignment| flat_expert_ids[*assignment as usize])
        .collect();
    (sorted_order, sorted_tokens, sorted_experts)
}

/// Values of `actual` that differ from `expected`, compared as numbers with
/// NaN equal to NaN; 0 when the bits agree.
pub fn mismatched_values(runtime: &MlxRuntime, expected: &MlxArray, actual: &MlxArray) -> usize {
    assert_eq!(expected.shape(), actual.shape(), "the shapes match");
    let equality = runtime
        .array_equal(expected, actual, true)
        .expect("the outputs compare");
    runtime
        .array_to_vec_f32(
            &runtime
                .astype(&equality, MlxDtype::Float32)
                .expect("flags cast"),
        )
        .expect("flags read back")
        .iter()
        .filter(|flag| **flag == 0.0)
        .count()
}
