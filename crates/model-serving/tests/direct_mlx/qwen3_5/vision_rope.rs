use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};
use astronomical_mlx_c_rust::{MlxCompiledElementwiseGraphs, MlxDtype};

// Qwen3.5 vision tower geometry: hidden 1152 over 16 heads is head dimension 72.
const HEAD_COUNT: i32 = 16;
const HEAD_DIMENSION: i32 = 72;
const PATCH_COUNT: i32 = 4;

#[tokio::test]
async fn should_match_composed_vision_rope_bit_for_bit_through_the_compiled_graph() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the vision rope test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the compiled elementwise graphs should initialize");

    let attention_states = bf16_array(
        &runtime,
        &patterned_values(
            (PATCH_COUNT * HEAD_COUNT * HEAD_DIMENSION) as usize,
            0.01,
            -0.5,
        ),
        &[PATCH_COUNT, HEAD_COUNT, HEAD_DIMENSION],
    );
    let rotary_cosines = f32_array(
        &runtime,
        &patterned_values((PATCH_COUNT * HEAD_DIMENSION) as usize, 0.02, -1.0),
        &[PATCH_COUNT, 1, HEAD_DIMENSION],
    );
    let rotary_sines = f32_array(
        &runtime,
        &patterned_values((PATCH_COUNT * HEAD_DIMENSION) as usize, 0.03, 0.25),
        &[PATCH_COUNT, 1, HEAD_DIMENSION],
    );

    let half_head_dimension = HEAD_DIMENSION / 2;
    let first_half = runtime
        .slice(
            &attention_states,
            &[0, 0, 0],
            &[PATCH_COUNT, HEAD_COUNT, half_head_dimension],
            &[1, 1, 1],
        )
        .expect("the first head half should slice");
    let second_half = runtime
        .slice(
            &attention_states,
            &[0, 0, half_head_dimension],
            &[PATCH_COUNT, HEAD_COUNT, HEAD_DIMENSION],
            &[1, 1, 1],
        )
        .expect("the second head half should slice");

    let composed_output = composed_vision_rope(
        &runtime,
        &attention_states,
        &rotary_cosines,
        &rotary_sines,
        &first_half,
        &second_half,
    );
    let compiled_output = runtime
        .apply_compiled_vision_rope(
            &compiled_elementwise_graphs,
            &attention_states,
            &rotary_cosines,
            &rotary_sines,
            &first_half,
            &second_half,
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

// Mirrors the production `apply_rotary_embedding` op sequence exactly so the
// contract compares the compiled graph against the reference it replaces.
fn composed_vision_rope(
    runtime: &MlxRuntime,
    attention_states: &astronomical_mlx_c_rust::MlxArray,
    rotary_cosines: &astronomical_mlx_c_rust::MlxArray,
    rotary_sines: &astronomical_mlx_c_rust::MlxArray,
    first_half: &astronomical_mlx_c_rust::MlxArray,
    second_half: &astronomical_mlx_c_rust::MlxArray,
) -> astronomical_mlx_c_rust::MlxArray {
    let negative_second_half = runtime
        .negative(second_half)
        .expect("the second half should negate");
    let rotated_states = runtime
        .concatenate_axis(&[&negative_second_half, first_half], 2)
        .expect("the rotated halves should concatenate");
    let cosine_component = runtime
        .multiply(attention_states, rotary_cosines)
        .expect("the cosine component should build");
    let sine_component = runtime
        .multiply(&rotated_states, rotary_sines)
        .expect("the sine component should build");
    let summed_components = runtime
        .add(&cosine_component, &sine_component)
        .expect("the components should add");
    runtime
        .astype(&summed_components, attention_states.dtype())
        .expect("the composed rope should restore the input dtype")
}

fn bf16_array(
    runtime: &MlxRuntime,
    values: &[f32],
    shape: &[i32],
) -> astronomical_mlx_c_rust::MlxArray {
    runtime
        .astype(
            &runtime
                .array_from_f32(values, shape)
                .expect("the bfloat16 source should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the array should cast to bfloat16")
}

fn f32_array(
    runtime: &MlxRuntime,
    values: &[f32],
    shape: &[i32],
) -> astronomical_mlx_c_rust::MlxArray {
    runtime
        .array_from_f32(values, shape)
        .expect("the float32 array should be valid")
}

fn patterned_values(element_count: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..element_count)
        .map(|value_index| ((value_index % 23) as f32 - 11.0) * scale + offset)
        .collect()
}
