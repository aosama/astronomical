//! Fixtures the sorted-expert matmul's contract and measurements share: a
//! quantized expert stack, a tile layout, the engine's stable sort of routed assignments,
//! and a bit-for-bit comparison of two outputs.

use mlx_bridge::{MlxArray, MlxDtype};
use model_serving::sorted_expert_matmul::{ExpertStack, SortedExpertLayout};
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

/// A quantized expert stack: packed weights, scales and biases.
pub type QuantizedStack = (MlxArray, MlxArray, MlxArray);

/// The kernels' view of a quantized stack.
pub fn stack_of(experts: &QuantizedStack) -> ExpertStack<'_> {
    ExpertStack {
        packed_weight: &experts.0,
        scales: &experts.1,
        biases: &experts.2,
    }
}

/// MLX's own sorted 4-bit `gather_qmm` of `sorted_rows` by each row's expert.
pub fn mlx_sorted_gather_qmm(
    runtime: &MlxRuntime,
    sorted_rows: &MlxArray,
    experts: &QuantizedStack,
    sorted_expert_ids: &MlxArray,
) -> MlxArray {
    runtime
        .gather_quantized_matmul_affine(
            sorted_rows,
            &experts.0,
            &experts.1,
            &experts.2,
            None,
            Some(sorted_expert_ids),
            true,
            QUANTIZATION_GROUP_SIZE,
            4,
            true,
        )
        .expect("MLX projects")
}

/// `row_count` standard-normal bf16 rows `[rows, 1, width]`, the shape the
/// expert kernels read.
pub fn standard_normal_rows(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    (row_count, width): (i32, i32),
) -> MlxArray {
    let rows_shape = [row_count, 1, width];
    runtime
        .random_normal(&rows_shape, MlxDtype::BFloat16, 0.0, 1.0, random_key)
        .expect("rows are drawn")
}

/// The engine's stable sort of `flat_expert_ids`, uploaded as int32
/// arrays: (sorted order, each sorted assignment's token, its expert).
pub fn uploaded_stable_routing(
    runtime: &MlxRuntime,
    flat_expert_ids: &[i32],
    experts_per_token: i32,
) -> (MlxArray, MlxArray, MlxArray) {
    let (sorted_order, sorted_tokens, sorted_experts) =
        stable_expert_sort(flat_expert_ids, experts_per_token);
    let routing_shape = [sorted_order.len() as i32];
    let mut uploaded = Vec::with_capacity(3);
    for routing_values in [&sorted_order, &sorted_tokens, &sorted_experts] {
        uploaded.push(
            runtime
                .array_from_i32(routing_values, &routing_shape)
                .expect("the routing uploads"),
        );
    }
    let sorted_experts = uploaded.pop().expect("three arrays were uploaded");
    let sorted_tokens = uploaded.pop().expect("three arrays were uploaded");
    let sorted_order = uploaded.pop().expect("three arrays were uploaded");
    (sorted_order, sorted_tokens, sorted_experts)
}
