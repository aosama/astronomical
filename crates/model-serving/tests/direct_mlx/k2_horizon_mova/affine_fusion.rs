//! Fused q/k/v projection parity against the separate projection route.

use astronomical_model_serving::K2HorizonMoVAAffineLinear;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const HIDDEN_SIZE: usize = 64;
const GROUP_SIZE: usize = 64;
const BITS: u32 = 4;
/// 4-bit elements packed into 32-bit words for one 64-element group.
const PACKED_WORDS_PER_ROW: usize = GROUP_SIZE * BITS as usize / 32;

#[tokio::test]
async fn should_match_fused_qkv_projection_exactly() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();

    // Row counts mirror a real member's attention shape class: q rows differ
    // from k/v rows, and the parts share affine geometry.
    let query_row_count = 8_usize;
    let key_row_count = 4_usize;
    let value_row_count = 4_usize;

    let build_part = |row_count: usize, row_seed: usize| {
        let packed_values = (0..row_count * PACKED_WORDS_PER_ROW)
            .map(|index| (index as u32 + row_seed as u32) | 0x0F0F_0F0F)
            .collect::<Vec<_>>();
        let packed = runtime
            .array_from_u32(
                &packed_values,
                &[row_count as i32, PACKED_WORDS_PER_ROW as i32],
            )
            .expect("packed weights");
        let scale_values = (0..row_count)
            .map(|row| 0.5 + (row_seed + row) as f32 * 0.01)
            .collect::<Vec<_>>();
        let scales = runtime
            .array_from_f32(&scale_values, &[row_count as i32, 1])
            .expect("scales");
        let bias_values = (0..row_count)
            .map(|row| -0.25 + (row_seed + row) as f32 * 0.02)
            .collect::<Vec<_>>();
        let biases = runtime
            .array_from_f32(&bias_values, &[row_count as i32, 1])
            .expect("biases");
        K2HorizonMoVAAffineLinear::new(packed, scales, biases, BITS, GROUP_SIZE as u32, None)
    };
    let query_part = build_part(query_row_count, 0);
    let key_part = build_part(key_row_count, 100);
    let value_part = build_part(value_row_count, 200);

    let fused = K2HorizonMoVAAffineLinear::fuse_output_rows(
        &runtime,
        &[&query_part, &key_part, &value_part],
    )
    .expect("fused projection")
    .expect("matching geometry must fuse");

    let input_values = (0..3 * HIDDEN_SIZE)
        .map(|index| (index as f32 * 0.05).cos())
        .collect::<Vec<_>>();
    let input = runtime
        .array_from_f32(&input_values, &[1, 3, HIDDEN_SIZE as i32])
        .expect("projection input");

    let fused_output = fused.project(&runtime, &input).expect("fused projection");
    let query_output = query_part.project(&runtime, &input).expect("q projection");
    let key_output = key_part.project(&runtime, &input).expect("k projection");
    let value_output = value_part.project(&runtime, &input).expect("v projection");

    let split_parts = K2HorizonMoVAAffineLinear::split_projection_output(
        &runtime,
        &fused_output,
        &[query_row_count, key_row_count, value_row_count],
    )
    .expect("split fused output");
    let split_parts = split_parts
        .iter()
        .map(|part| runtime.array_to_vec_f32(part).expect("part values"))
        .collect::<Vec<_>>();
    for (separate, fused_values) in [
        (&query_output, &split_parts[0]),
        (&key_output, &split_parts[1]),
        (&value_output, &split_parts[2]),
    ] {
        assert_eq!(
            separate.to_vec_f32().expect("separate values"),
            *fused_values,
            "the fused route must be bit-identical to the separate route"
        );
    }
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("test memory limits"),
    )
    .expect("direct MLX runtime")
}
