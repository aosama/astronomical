use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
};
use mlx_bridge::{MlxCompiledElementwiseGraphs, MlxDtype};

use super::vision_rope_operands::{HEAD_COUNT, HEAD_DIMENSION, VisionRopeOperands};

const PATCH_COUNT: i32 = 4;

#[tokio::test]
async fn should_match_composed_vision_rope_bit_for_bit_through_the_compiled_graph() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
        )
        .expect("the vision rope test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the compiled elementwise graphs should initialize");

    let operands = VisionRopeOperands::patterned(&runtime, PATCH_COUNT);

    let composed_output = operands.composed_rope(&runtime);
    let compiled_output = runtime
        .apply_compiled_vision_rope(
            &compiled_elementwise_graphs,
            &operands.attention_states,
            &operands.rotary_cosines,
            &operands.rotary_sines,
            &operands.first_half,
            &operands.second_half,
        )
        .expect("the compiled vision rope should build a valid graph");

    assert_eq!(
        compiled_output.shape(),
        vec![PATCH_COUNT, HEAD_COUNT, HEAD_DIMENSION],
        "the compiled vision rope must preserve the attention state shape"
    );
    assert_eq!(
        compiled_output.dtype(),
        MlxDtype::BFloat16,
        "the compiled vision rope must restore the attention state dtype"
    );

    let composed_values = runtime
        .astype(&composed_output, MlxDtype::Float32)
        .expect("the composed output should cast to float32")
        .to_vec_f32()
        .expect("the composed output should evaluate as float32");
    let compiled_values = runtime
        .astype(&compiled_output, MlxDtype::Float32)
        .expect("the compiled output should cast to float32")
        .to_vec_f32()
        .expect("the compiled output should evaluate as float32");

    let maximum_difference = composed_values
        .iter()
        .zip(compiled_values.iter())
        .map(|(composed_value, compiled_value)| (composed_value - compiled_value).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        maximum_difference == 0.0,
        "the compiled vision rope must be bit-exact with the composed path; max diff {maximum_difference}"
    );
}
