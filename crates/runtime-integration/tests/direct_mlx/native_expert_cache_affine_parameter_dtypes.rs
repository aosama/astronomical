use std::fs;

use astronomical_runtime_integration::{
    MlxDtype, MlxNativeExpertCache, MlxNativeExpertLayerDescriptor, MlxNativeExpertParameter,
    MlxNativeExpertProjection, MlxNativeExpertTensorSourceDescriptor,
};

use crate::common::runtime_test_support::runtime;

const INPUT_DIMENSION: i32 = 64;
const OUTPUT_DIMENSION: i32 = 32;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 8;

#[test]
fn should_preserve_every_supported_affine_parameter_dtype_in_native_slots() {
    let runtime = runtime();
    for affine_parameter_dtype in [MlxDtype::Float16, MlxDtype::BFloat16, MlxDtype::Float32] {
        eprintln!("[native-expert-affine-dtype] status=progress dtype={affine_parameter_dtype:?}");
        assert_native_affine_profile_matches_mlx(
            &runtime,
            affine_parameter_dtype,
            affine_parameter_dtype,
            affine_parameter_dtype,
            INPUT_DIMENSION,
            OUTPUT_DIMENSION,
            affine_parameter_dtype,
            false,
        );
    }
}

#[test]
fn should_promote_bfloat16_activations_and_float16_affine_parameters_like_mlx_gather_qmm() {
    let runtime = runtime();
    assert_native_affine_profile_matches_mlx(
        &runtime,
        MlxDtype::BFloat16,
        MlxDtype::Float16,
        MlxDtype::Float16,
        INPUT_DIMENSION,
        OUTPUT_DIMENSION,
        MlxDtype::Float32,
        true,
    );
}

#[test]
fn should_promote_independent_scale_and_bias_dtypes_like_mlx_gather_qmm() {
    let runtime = runtime();
    assert_native_affine_profile_matches_mlx(
        &runtime,
        MlxDtype::BFloat16,
        MlxDtype::Float16,
        MlxDtype::BFloat16,
        INPUT_DIMENSION,
        OUTPUT_DIMENSION,
        MlxDtype::Float32,
        false,
    );
}

#[test]
fn should_promote_the_fast_eight_bit_decode_kernel_like_mlx_gather_qmm() {
    let runtime = runtime();
    assert_native_affine_profile_matches_mlx(
        &runtime,
        MlxDtype::BFloat16,
        MlxDtype::Float16,
        MlxDtype::Float16,
        512,
        8,
        MlxDtype::Float32,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
fn assert_native_affine_profile_matches_mlx(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    activation_dtype: MlxDtype,
    scale_dtype: MlxDtype,
    bias_dtype: MlxDtype,
    input_dimension: i32,
    output_dimension: i32,
    expected_output_dtype: MlxDtype,
    should_exercise_sorted_prefill: bool,
) {
    let fixture_directory =
        tempfile::tempdir().expect("the mixed affine dtype fixture should exist");
    let source_file_path = fixture_directory.path().join("expert-source.bin");
    let packed_words_per_output_row = input_dimension * BITS / i32::BITS as i32;
    let packed_weight_values: Vec<u32> =
        vec![0x1111_1111; (output_dimension * packed_words_per_output_row) as usize];
    let packed_weight_bytes = packed_weight_values
        .iter()
        .flat_map(|packed_weight_value| packed_weight_value.to_ne_bytes())
        .collect::<Vec<_>>();
    let affine_groups_per_output_row = input_dimension / GROUP_SIZE;
    let affine_value_count = (output_dimension * affine_groups_per_output_row) as usize;
    let scale_bytes = affine_parameter_bytes(scale_dtype, 1.0, affine_value_count);
    let bias_bytes = affine_parameter_bytes(bias_dtype, 0.0, affine_value_count);
    let mut source_file_bytes = Vec::new();
    let mut tensor_sources = Vec::new();
    for projection in [
        MlxNativeExpertProjection::Gate,
        MlxNativeExpertProjection::Up,
        MlxNativeExpertProjection::Down,
    ] {
        append_source(
            &mut source_file_bytes,
            &mut tensor_sources,
            &source_file_path,
            projection,
            MlxNativeExpertParameter::PackedWeight,
            &packed_weight_bytes,
            vec![1, output_dimension, packed_words_per_output_row],
            MlxDtype::UInt32,
        );
        append_source(
            &mut source_file_bytes,
            &mut tensor_sources,
            &source_file_path,
            projection,
            MlxNativeExpertParameter::Scales,
            &scale_bytes,
            vec![1, output_dimension, affine_groups_per_output_row],
            scale_dtype,
        );
        append_source(
            &mut source_file_bytes,
            &mut tensor_sources,
            &source_file_path,
            projection,
            MlxNativeExpertParameter::Biases,
            &bias_bytes,
            vec![1, output_dimension, affine_groups_per_output_row],
            bias_dtype,
        );
    }
    fs::write(&source_file_path, &source_file_bytes)
        .expect("the mixed affine dtype source fixture should be writable");
    let native_expert_cache = MlxNativeExpertCache::new(
        runtime,
        &[MlxNativeExpertLayerDescriptor::new(1, tensor_sources)],
        source_file_bytes.len() as u64,
    )
    .expect("the native cache should preserve each affine parameter dtype");
    let selected_expert_indices = runtime
        .array_from_i32(&[0], &[1])
        .expect("the selected expert index should be valid");
    let (native_snapshot, _) = native_expert_cache
        .prepare_layer(runtime, 0, &selected_expert_indices, true)
        .expect("the mixed affine dtype expert should load");
    let activation_values = (1..=input_dimension)
        .map(|activation_position| activation_position as f32)
        .collect::<Vec<_>>();
    let float32_activations = runtime
        .array_from_f32(&activation_values, &[1, input_dimension])
        .expect("the source activations should be valid");
    let activations = runtime
        .astype(&float32_activations, activation_dtype)
        .expect("the activations should use the model dtype");
    let packed_weights = runtime
        .array_from_u32(
            &packed_weight_values,
            &[1, output_dimension, packed_words_per_output_row],
        )
        .expect("the reference packed weights should be valid");
    let float32_scales = runtime
        .array_from_f32(
            &vec![1.0; affine_value_count],
            &[1, output_dimension, affine_groups_per_output_row],
        )
        .expect("the reference scales should be valid");
    let float32_biases = runtime
        .array_from_f32(
            &vec![0.0; affine_value_count],
            &[1, output_dimension, affine_groups_per_output_row],
        )
        .expect("the reference biases should be valid");
    let scales = runtime
        .astype(&float32_scales, scale_dtype)
        .expect("the reference scales should preserve the artifact dtype");
    let biases = runtime
        .astype(&float32_biases, bias_dtype)
        .expect("the reference biases should preserve the artifact dtype");

    assert_projection_matches_mlx(
        runtime,
        &native_snapshot,
        &activations,
        &selected_expert_indices,
        &packed_weights,
        &scales,
        &biases,
        expected_output_dtype,
        false,
    );

    if should_exercise_sorted_prefill {
        let sorted_assignment_count = 64;
        let sorted_selected_expert_indices = runtime
            .array_from_i32(
                &vec![0; sorted_assignment_count],
                &[sorted_assignment_count as i32],
            )
            .expect("the sorted prefill route should be valid");
        let sorted_float32_activations = runtime
            .array_from_f32(
                &(0..sorted_assignment_count)
                    .flat_map(|assignment_position| {
                        let row_multiplier = (assignment_position % 4 + 1) as f32;
                        (1..=input_dimension).map(move |activation_position| {
                            row_multiplier * activation_position as f32
                        })
                    })
                    .collect::<Vec<_>>(),
                &[sorted_assignment_count as i32, 1, input_dimension],
            )
            .expect("the sorted prefill activations should be valid");
        let sorted_activations = runtime
            .astype(&sorted_float32_activations, activation_dtype)
            .expect("the sorted activations should use the model dtype");
        assert_projection_matches_mlx(
            runtime,
            &native_snapshot,
            &sorted_activations,
            &sorted_selected_expert_indices,
            &packed_weights,
            &scales,
            &biases,
            expected_output_dtype,
            true,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_projection_matches_mlx(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    native_snapshot: &astronomical_runtime_integration::MlxNativeExpertCacheSnapshot,
    activations: &astronomical_runtime_integration::MlxArray,
    selected_expert_indices: &astronomical_runtime_integration::MlxArray,
    packed_weights: &astronomical_runtime_integration::MlxArray,
    scales: &astronomical_runtime_integration::MlxArray,
    biases: &astronomical_runtime_integration::MlxArray,
    expected_output_dtype: MlxDtype,
    sorted_indices: bool,
) {
    let native_output = native_snapshot
        .gather_matmul(
            runtime,
            MlxNativeExpertProjection::Gate,
            activations,
            selected_expert_indices,
            true,
            sorted_indices,
        )
        .expect("the native projection should promote mixed floating dtypes");
    let mlx_gather_qmm_output = runtime
        .gather_quantized_matmul_affine(
            activations,
            packed_weights,
            scales,
            biases,
            None,
            Some(selected_expert_indices),
            true,
            GROUP_SIZE,
            BITS,
            sorted_indices,
        )
        .expect("MLX gather_qmm should define the affine dtype reference");
    assert_eq!(native_output.dtype(), expected_output_dtype);
    assert_eq!(mlx_gather_qmm_output.dtype(), expected_output_dtype);
    let native_output_float32 = runtime
        .astype(&native_output, MlxDtype::Float32)
        .expect("the native affine dtype output should cast for comparison");
    let mlx_gather_qmm_output_float32 = runtime
        .astype(&mlx_gather_qmm_output, MlxDtype::Float32)
        .expect("the MLX affine dtype output should cast for comparison");
    assert_eq!(
        native_output_float32
            .to_vec_f32()
            .expect("the native affine dtype output should evaluate"),
        mlx_gather_qmm_output_float32
            .to_vec_f32()
            .expect("the MLX affine dtype output should evaluate")
    );
}

#[allow(clippy::too_many_arguments)]
fn append_source(
    source_file_bytes: &mut Vec<u8>,
    tensor_sources: &mut Vec<MlxNativeExpertTensorSourceDescriptor>,
    source_file_path: &std::path::Path,
    projection: MlxNativeExpertProjection,
    parameter: MlxNativeExpertParameter,
    parameter_bytes: &[u8],
    expert_shape: Vec<i32>,
    dtype: MlxDtype,
) {
    let tensor_payload_offset = source_file_bytes.len() as u64;
    source_file_bytes.extend_from_slice(parameter_bytes);
    tensor_sources.push(MlxNativeExpertTensorSourceDescriptor::new(
        projection,
        parameter,
        GROUP_SIZE,
        BITS,
        source_file_path.to_path_buf(),
        tensor_payload_offset,
        parameter_bytes.len(),
        expert_shape,
        dtype,
    ));
}

fn affine_parameter_bytes(dtype: MlxDtype, value: f32, count: usize) -> Vec<u8> {
    match dtype {
        MlxDtype::Float16 => vec![if value == 0.0 { 0x0000_u16 } else { 0x3c00_u16 }; count]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        MlxDtype::BFloat16 => vec![if value == 0.0 { 0x0000_u16 } else { 0x3f80_u16 }; count]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        MlxDtype::Float32 => vec![value; count]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        _ => unreachable!("the test only supplies supported affine parameter dtypes"),
    }
}
