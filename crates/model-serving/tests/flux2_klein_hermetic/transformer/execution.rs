//! Direct transformer contracts reuse the process-wide direct-MLX memory limits.

use std::collections::BTreeMap;

use astronomical_model_serving::{
    Flux2KleinBlockGroupEvent, Flux2KleinTransformer, Flux2KleinTransformerGeometry,
    Flux2KleinTransformerInputs, Flux2KleinTransformerWeights, apply_rope_for_component_oracle,
};

#[test]
fn should_rotate_adjacent_real_imaginary_pairs_against_an_asymmetric_scalar_oracle() {
    let runtime = test_runtime();
    let source_values = [1.0, 2.0, -3.0, 5.0, 7.0, -11.0, 13.0, 17.0];
    let cosine_values = [0.8, 0.8, 0.6, 0.6, -0.4, -0.4, 0.2, 0.2];
    let sine_values = [0.6, 0.6, -0.8, -0.8, 0.3, 0.3, -0.5, -0.5];
    let input = runtime
        .array_from_f32(&source_values, &[1, 1, 1, 8])
        .and_then(|values| runtime.astype(&values, MlxDtype::BFloat16))
        .expect("the asymmetric BF16 RoPE input should build");
    let cosines = runtime
        .array_from_f32(&cosine_values, &[1, 8])
        .expect("the cosine coefficients should build");
    let sines = runtime
        .array_from_f32(&sine_values, &[1, 8])
        .expect("the sine coefficients should build");

    let rotated = apply_rope_for_component_oracle(&runtime, &input, &cosines, &sines)
        .expect("adjacent-pair RoPE should build");
    let rotated = runtime
        .astype(&rotated, MlxDtype::Float32)
        .expect("adjacent-pair RoPE should evaluate")
        .to_vec_f32()
        .expect("the rotated values should materialize");
    let expected = adjacent_pair_rope_scalar_oracle(&source_values, &cosine_values, &sine_values);

    for (actual_value, expected_value) in rotated.iter().zip(expected) {
        assert!((actual_value - expected_value).abs() <= 0.08);
    }
}

fn adjacent_pair_rope_scalar_oracle(
    source_values: &[f32],
    cosine_values: &[f32],
    sine_values: &[f32],
) -> Vec<f32> {
    source_values
        .chunks_exact(2)
        .enumerate()
        .flat_map(|(pair_index, pair)| {
            let coefficient_index = pair_index * 2;
            let cosine = cosine_values[coefficient_index];
            let sine = sine_values[coefficient_index];
            [
                pair[0] * cosine - pair[1] * sine,
                pair[1] * cosine + pair[0] * sine,
            ]
        })
        .collect()
}
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[test]
fn should_complete_the_transformer_journey_in_observable_bounded_groups() {
    let runtime = test_runtime();
    let geometry = small_geometry();
    let weights = zero_weights(&runtime, &geometry);
    let transformer = Flux2KleinTransformer::new(runtime, geometry, weights)
        .expect("the injected transformer should be valid");
    let image = transformer
        .runtime()
        .zeros(&[1, 2, 4], MlxDtype::BFloat16)
        .expect("the image-token fixture should be valid");
    let text = transformer
        .runtime()
        .zeros(&[1, 3, 6], MlxDtype::BFloat16)
        .expect("the text fixture should be valid");
    let timestep = transformer
        .runtime()
        .array_from_f32(&[0.5], &[1])
        .expect("the timestep fixture should be valid");
    let image_ids = transformer
        .runtime()
        .zeros(&[2, 4], MlxDtype::Float32)
        .expect("the image positions should be valid");
    let text_ids = transformer
        .runtime()
        .zeros(&[3, 4], MlxDtype::Float32)
        .expect("the text positions should be valid");
    let mut events = Vec::new();
    let mut is_cancelled = || false;
    let mut record_event = |event| events.push(event);

    let output = transformer
        .forward_in_block_groups(
            Flux2KleinTransformerInputs::new(&image, &text, &timestep, &image_ids, &text_ids),
            1,
            &mut is_cancelled,
            &mut record_event,
        )
        .expect("the complete image denoising transformer journey should succeed");

    assert_eq!(output.sample().shape(), vec![1, 2, 4]);
    assert_eq!(output.sample().dtype(), MlxDtype::BFloat16);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Flux2KleinBlockGroupEvent::Completed { .. }))
            .count(),
        3
    );
    let sample_f32 = transformer
        .runtime()
        .astype(output.sample(), MlxDtype::Float32)
        .expect("the oracle output should cast to float32");
    assert_eq!(
        sample_f32.to_vec_f32().expect("the output should evaluate"),
        vec![0.0; 8]
    );
}

#[test]
fn should_expose_component_oracles_and_stop_before_a_cancelled_group() {
    let runtime = test_runtime();
    let geometry = small_geometry();
    let weights = zero_weights(&runtime, &geometry);
    let transformer = Flux2KleinTransformer::new(runtime, geometry, weights)
        .expect("the injected transformer should be valid");
    let image = transformer
        .runtime()
        .zeros(&[1, 1, 4], MlxDtype::BFloat16)
        .expect("valid image");
    let text = transformer
        .runtime()
        .zeros(&[1, 1, 6], MlxDtype::BFloat16)
        .expect("valid text");
    let timestep = transformer
        .runtime()
        .array_from_f32(&[0.0], &[1])
        .expect("valid timestep");
    let image_ids = transformer
        .runtime()
        .array_from_f32(&[0.0, 1.0, 2.0, 3.0], &[1, 4])
        .expect("valid image IDs");
    let text_ids = transformer
        .runtime()
        .zeros(&[1, 4], MlxDtype::Float32)
        .expect("valid text IDs");
    let inputs = Flux2KleinTransformerInputs::new(&image, &text, &timestep, &image_ids, &text_ids);

    let oracle = transformer
        .component_oracle(inputs)
        .expect("component boundary outputs should be available");
    assert_eq!(oracle.timestep_embedding().shape(), vec![1, 16]);
    assert_eq!(oracle.image_projection().shape(), vec![1, 1, 16]);
    assert_eq!(oracle.text_projection().shape(), vec![1, 1, 16]);
    assert_eq!(oracle.rope_cosines().shape(), vec![2, 8]);
    assert_eq!(oracle.rope_cosines().dtype(), MlxDtype::Float32);

    let mut cancellation_checks = 0;
    let mut is_cancelled = || {
        cancellation_checks += 1;
        cancellation_checks > 1
    };
    let mut events = Vec::new();
    let mut record_event = |event| events.push(event);
    let cancelled =
        transformer.forward_in_block_groups(inputs, 1, &mut is_cancelled, &mut record_event);
    assert!(cancelled.is_err());
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Flux2KleinBlockGroupEvent::Completed { .. }))
            .count(),
        1
    );
}

#[test]
fn should_cancel_after_a_middle_group_without_executing_later_groups_and_reuse_transformer() {
    let runtime = test_runtime();
    let geometry = small_geometry();
    let weights = zero_weights(&runtime, &geometry);
    let transformer = Flux2KleinTransformer::new(runtime, geometry, weights)
        .expect("the injected transformer should be valid");
    let image = transformer
        .runtime()
        .zeros(&[1, 2, 4], MlxDtype::BFloat16)
        .expect("the image fixture should be valid");
    let conditioning_value = f32::from(ROMEO_AND_JULIET_SOURCE.as_bytes()[0]) / 1_024.0;
    let text = transformer
        .runtime()
        .full(&[1, 2, 6], conditioning_value, MlxDtype::BFloat16)
        .expect("the Romeo and Juliet conditioning fixture should be valid");
    let timestep = transformer
        .runtime()
        .array_from_f32(&[0.5], &[1])
        .expect("the timestep fixture should be valid");
    let image_ids = transformer
        .runtime()
        .zeros(&[2, 4], MlxDtype::Float32)
        .expect("the image positions should be valid");
    let text_ids = transformer
        .runtime()
        .zeros(&[2, 4], MlxDtype::Float32)
        .expect("the text positions should be valid");
    let inputs = Flux2KleinTransformerInputs::new(&image, &text, &timestep, &image_ids, &text_ids);
    let mut completed_groups = Vec::new();
    let mut record_completed_group = |event| {
        if let Flux2KleinBlockGroupEvent::Completed {
            kind,
            first_block_index,
            ..
        } = event
        {
            completed_groups.push((kind, first_block_index));
        }
    };

    let first_advance = transformer
        .advance_one_block_group(
            transformer
                .start_forward(inputs)
                .expect("the forward should start"),
            1,
            &mut record_completed_group,
        )
        .expect("the first block group should complete");
    let second_advance = transformer
        .advance_one_block_group(
            first_advance
                .into_forward_state()
                .expect("the first group must retain state"),
            1,
            &mut record_completed_group,
        )
        .expect("the middle block group should complete");
    let cancelled_forward_state = second_advance
        .into_forward_state()
        .expect("a middle group must not publish partial output");
    drop(cancelled_forward_state);
    transformer
        .runtime()
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .expect("cancellation cleanup should retire work and clear allocator cache");

    assert_eq!(completed_groups.len(), 2);
    let mut never_cancelled = || false;
    let mut ignore_group_event = |_| {};
    let reusable_output = transformer
        .forward_in_block_groups(inputs, 1, &mut never_cancelled, &mut ignore_group_event)
        .expect("the transformer should remain reusable after cancellation");
    assert_eq!(reusable_output.sample().shape(), [1, 2, 4]);
}

#[test]
fn should_preserve_exact_output_with_complete_zero_and_partial_block_residency() {
    let complete_output = execute_with_resident_blocks(&[0, 1, 2], 1);
    let zero_residency_output = execute_with_resident_blocks(&[], 4);
    let partial_residency_output = execute_with_resident_blocks(&[1], 4);

    assert_eq!(zero_residency_output, complete_output);
    assert_eq!(partial_residency_output, complete_output);
}

fn execute_with_resident_blocks(
    resident_block_indices: &[usize],
    forward_count: usize,
) -> Vec<f32> {
    let runtime = test_runtime();
    let geometry = small_geometry();
    let weights = patterned_weights(&runtime, &geometry, resident_block_indices);
    let resident_weight_shapes = geometry
        .expected_weight_shapes()
        .filter(|(tensor_name, _)| {
            geometry.block_index_for_weight_name(tensor_name).is_none()
                || geometry
                    .block_index_for_weight_name(tensor_name)
                    .is_some_and(|block_index| resident_block_indices.contains(&block_index))
        })
        .collect::<Vec<_>>();
    let expected_resident_payload_bytes = resident_weight_shapes
        .iter()
        .map(|(_, shape)| shape.iter().product::<usize>() as u64 * 2)
        .sum::<u64>();
    assert_eq!(weights.retained_block_count(), resident_block_indices.len());
    assert_eq!(
        weights.resident_tensor_count(),
        resident_weight_shapes.len()
    );
    assert_eq!(
        weights.resident_payload_bytes(),
        expected_resident_payload_bytes
    );

    let transformer = Flux2KleinTransformer::new(runtime, geometry, weights)
        .expect("the residency-specific transformer should be valid");
    let image = transformer
        .runtime()
        .full(&[1, 2, 4], 0.25, MlxDtype::BFloat16)
        .expect("the image fixture should be valid");
    let text = transformer
        .runtime()
        .full(&[1, 2, 6], -0.125, MlxDtype::BFloat16)
        .expect("the text fixture should be valid");
    let timestep = transformer
        .runtime()
        .array_from_f32(&[0.5], &[1])
        .expect("the timestep fixture should be valid");
    let image_ids = transformer
        .runtime()
        .zeros(&[2, 4], MlxDtype::Float32)
        .expect("the image positions should be valid");
    let text_ids = transformer
        .runtime()
        .zeros(&[2, 4], MlxDtype::Float32)
        .expect("the text positions should be valid");
    let mut output_values = Vec::new();
    for _forward_index in 0..forward_count {
        let mut is_cancelled = || false;
        let mut record_event = |_| {};
        let output = transformer
            .forward_in_block_groups(
                Flux2KleinTransformerInputs::new(&image, &text, &timestep, &image_ids, &text_ids),
                1,
                &mut is_cancelled,
                &mut record_event,
            )
            .expect("every residency mode should execute the complete denoising journey");
        let current_output_values = transformer
            .runtime()
            .astype(output.sample(), MlxDtype::Float32)
            .expect("the output should cast to FP32")
            .to_vec_f32()
            .expect("the output should materialize");
        if !output_values.is_empty() {
            assert_eq!(current_output_values, output_values);
        }
        output_values = current_output_values;
    }
    assert_eq!(
        transformer.weights().retained_block_count(),
        resident_block_indices.len()
    );
    output_values
}

fn small_geometry() -> Flux2KleinTransformerGeometry {
    Flux2KleinTransformerGeometry::new(16, 2, 8, 4, 6, [2, 2, 2, 2], 2_000.0, 1, 2, 4, 1.0e-6)
        .expect("the reduced geometry should preserve every architectural relation")
}

fn zero_weights(
    runtime: &MlxRuntime,
    geometry: &Flux2KleinTransformerGeometry,
) -> Flux2KleinTransformerWeights {
    let tensors = geometry
        .expected_weight_shapes()
        .map(|(name, shape)| {
            let signed_shape = shape
                .iter()
                .map(|dimension| i32::try_from(*dimension).expect("small fixture dimension"))
                .collect::<Vec<_>>();
            let tensor = runtime
                .zeros(&signed_shape, MlxDtype::BFloat16)
                .expect("the BF16 fixture tensor should be valid");
            (name, tensor)
        })
        .collect::<BTreeMap<_, _>>();
    Flux2KleinTransformerWeights::bind_injected(tensors, geometry)
        .expect("the exact injected tensor set should bind")
}

fn patterned_weights(
    runtime: &MlxRuntime,
    geometry: &Flux2KleinTransformerGeometry,
    resident_block_indices: &[usize],
) -> Flux2KleinTransformerWeights {
    let tensors = geometry
        .expected_weight_shapes()
        .enumerate()
        .map(|(tensor_index, (name, shape))| {
            let signed_shape = shape
                .iter()
                .map(|dimension| i32::try_from(*dimension).expect("small fixture dimension"))
                .collect::<Vec<_>>();
            let tensor = runtime
                .full(
                    &signed_shape,
                    (tensor_index as f32 + 1.0) / 4_096.0,
                    MlxDtype::BFloat16,
                )
                .expect("the patterned BF16 fixture tensor should be valid");
            (name, tensor)
        })
        .collect::<BTreeMap<_, _>>();
    Flux2KleinTransformerWeights::bind_injected_with_residency(
        tensors,
        geometry,
        resident_block_indices,
    )
    .expect("the residency-specific injected tensor set should bind")
}

fn test_runtime() -> MlxRuntime {
    let limits = MlxMemoryLimits::new(
        DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
        DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
    )
    .expect("the bounded test MLX limits should be valid");
    MlxRuntime::initialize(limits).expect("the native MLX test runtime should initialize")
}
