//! Direct MLX text-conditioning oracles share the process-wide suite memory limits.

use std::fs::File;
use std::time::Duration;

use super::error::Flux2KleinTextConditioningError;
use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime, MlxSafetensors,
};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[allow(dead_code)]
#[path = "../../../src/flux2_klein/text_conditioning/layer.rs"]
mod layer;

#[allow(dead_code)]
mod weights {
    use super::MlxArray;

    pub(super) const HEAD_WIDTH: i32 = 128;
    pub(super) const HIDDEN_WIDTH: i32 = 2_560;
    pub(super) const KEY_VALUE_HEAD_COUNT: i32 = 8;
    pub(super) const QUERY_HEAD_COUNT: i32 = 32;

    pub(super) struct Flux2KleinDecoderLayerWeights {
        pub(super) input_norm: MlxArray,
        pub(super) query: MlxArray,
        pub(super) key: MlxArray,
        pub(super) value: MlxArray,
        pub(super) output: MlxArray,
        pub(super) query_norm: MlxArray,
        pub(super) key_norm: MlxArray,
        pub(super) post_attention_norm: MlxArray,
        pub(super) gate: MlxArray,
        pub(super) up: MlxArray,
        pub(super) down: MlxArray,
    }
}

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);
const DIRECT_MLX_TEXT_CONDITIONING_TIMEOUT: Duration = Duration::from_secs(115);
const REDUCED_LAYER_COUNT: usize = 3;

#[test]
fn should_build_the_combined_causal_and_right_padding_mask_on_mlx() {
    let runtime = test_runtime();
    let source_word_count = ROMEO_AND_JULIET_SOURCE.split_whitespace().take(2).count();
    let attention_mask = [1, 1, 0, 0];

    let combined_mask = layer::build_causal_padding_mask(
        &runtime,
        &attention_mask,
        1,
        i32::try_from(source_word_count + 2).expect("the small sequence should fit i32"),
    )
    .expect("the combined attention mask should build");
    let mask_bytes = runtime
        .astype(&combined_mask, MlxDtype::UInt8)
        .expect("the boolean mask should cast to bytes");
    let mask_values = mask_bytes
        .to_vec_u8()
        .expect("the MLX mask should materialize as boolean bytes");

    assert_eq!(combined_mask.shape(), [1, 1, 4, 4]);
    assert_eq!(
        mask_values,
        vec![1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,]
    );
}

#[test]
fn should_match_native_nontraditional_rope_to_a_scalar_oracle() {
    let runtime = test_runtime();
    let source_values = ROMEO_AND_JULIET_SOURCE
        .bytes()
        .take(8)
        .map(|source_byte| f32::from(source_byte) / 128.0)
        .collect::<Vec<_>>();
    let input = runtime
        .array_from_f32(&source_values, &[1, 1, 2, 4])
        .expect("the Romeo and Juliet RoPE input should build");

    let rotated = runtime
        .rope(&input, 4, 1_000_000.0, 0)
        .expect("native MLX RoPE should build")
        .to_vec_f32()
        .expect("native MLX RoPE should evaluate");
    let expected = nontraditional_rope_oracle(&source_values);

    for (actual_value, expected_value) in rotated.iter().zip(expected) {
        assert!((actual_value - expected_value).abs() <= 1.0e-5);
    }
}

#[tokio::test]
async fn should_match_streamed_and_complete_conditioning_then_cancel_and_reuse_the_stream() {
    tokio::time::timeout(DIRECT_MLX_TEXT_CONDITIONING_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = test_runtime();
        let fixture_directory = tempfile::tempdir()
            .expect("the reduced text-conditioning fixture directory should exist");
        let shard_path = fixture_directory.path().join("reduced-text.safetensors");
        let initial_hidden_states = reduced_bf16_array(&runtime, 16, &[1, 4, 4]);
        let source_weights = (0..REDUCED_LAYER_COUNT)
            .map(|layer_index| {
                let source_offset = layer_index * 16;
                reduced_bf16_array_with_offset(&runtime, 16, &[4, 4], source_offset)
            })
            .collect::<Vec<_>>();
        let tensor_names = (0..REDUCED_LAYER_COUNT)
            .map(|layer_index| format!("model.layers.{layer_index}.weight"))
            .collect::<Vec<_>>();
        let named_weights = tensor_names
            .iter()
            .zip(source_weights.iter())
            .map(|(tensor_name, weight)| (tensor_name.as_str(), weight))
            .collect::<Vec<_>>();
        runtime
            .save_safetensors(
                File::create(&shard_path)
                    .expect("the reduced SafeTensors shard should be creatable"),
                &named_weights,
                &[],
            )
            .expect("the reduced BF16 weights should serialize");
        drop(named_weights);
        drop(source_weights);
        runtime
            .clear_allocator_cache()
            .expect("the serialized fixture sources should release before qualification");

        let complete_map = load_reduced_map(&runtime, &shard_path);
        let complete_weights = tensor_names
            .iter()
            .map(|tensor_name| {
                complete_map
                    .tensor(tensor_name)
                    .expect("the complete reduced layer should bind")
            })
            .collect::<Vec<_>>();
        let complete_weight_references = complete_weights.iter().collect::<Vec<_>>();
        runtime
            .evaluate_arrays(&complete_weight_references)
            .expect("all complete reduced weights should materialize");
        let complete_conditioning = execute_reduced_layers(
            &runtime,
            initial_hidden_states
                .retain()
                .expect("the complete input should retain"),
            complete_weights.iter(),
        );
        drop(complete_weights);
        drop(complete_map);
        runtime
            .clear_allocator_cache()
            .expect("the complete reduced weights should release before streaming");

        let mut cancelled_hidden_states = initial_hidden_states
            .retain()
            .expect("the cancellable Romeo and Juliet input should retain");
        let mut cancelled_taps = Vec::new();
        let mut cancelled_layer_reads = Vec::new();
        let cancelled_map = load_reduced_map(&runtime, &shard_path);
        cancelled_layer_reads.push(0_usize);
        let cancelled_layer_weight = cancelled_map
            .tensor(&tensor_names[0])
            .expect("the first cancellable reduced layer should bind");
        runtime
            .evaluate_arrays(&[&cancelled_layer_weight])
            .expect("the cancellable layer should materialize");
        drop(cancelled_map);
        cancelled_hidden_states =
            reduced_layer_forward(&runtime, &cancelled_hidden_states, &cancelled_layer_weight);
        runtime
            .evaluate_arrays(&[&cancelled_hidden_states])
            .expect("the intermediate cancellable state should materialize");
        cancelled_taps.push(
            cancelled_hidden_states
                .retain()
                .expect("the intermediate cancellable tap should retain"),
        );
        drop(cancelled_layer_weight);
        drop(cancelled_hidden_states);
        drop(cancelled_taps);
        runtime
            .synchronize_gpu_stream_and_clear_allocator_cache()
            .expect("cancellation should synchronize and clear reclaimable MLX storage");
        assert_eq!(cancelled_layer_reads, vec![0]);
        assert_eq!(
            runtime
                .memory_snapshot()
                .expect("the post-cancellation memory snapshot should be available")
                .allocator_cache_memory_bytes(),
            0,
        );

        // Reusing the request owner starts descriptor reads from layer zero again.
        let mut streamed_hidden_states = initial_hidden_states;
        let mut streamed_taps = Vec::with_capacity(REDUCED_LAYER_COUNT);
        let mut streamed_layer_reads = Vec::with_capacity(REDUCED_LAYER_COUNT);
        let mut retained_layer_count = 0_usize;
        let mut peak_retained_layer_count = 0_usize;
        for (layer_index, tensor_name) in tensor_names.iter().enumerate() {
            let streamed_map = load_reduced_map(&runtime, &shard_path);
            streamed_layer_reads.push(layer_index);
            let layer_weight = streamed_map
                .tensor(tensor_name)
                .expect("the streamed reduced layer should bind");
            runtime
                .evaluate_arrays(&[&layer_weight])
                .expect("one streamed reduced layer should materialize");
            drop(streamed_map);
            retained_layer_count += 1;
            peak_retained_layer_count = peak_retained_layer_count.max(retained_layer_count);
            streamed_hidden_states =
                reduced_layer_forward(&runtime, &streamed_hidden_states, &layer_weight);
            runtime
                .evaluate_arrays(&[&streamed_hidden_states])
                .expect("the streamed hidden state should materialize before release");
            streamed_taps.push(
                streamed_hidden_states
                    .retain()
                    .expect("the streamed tap should retain"),
            );
            drop(layer_weight);
            retained_layer_count -= 1;
            runtime
                .clear_allocator_cache()
                .expect("the released layer cache should be reclaimable");
        }
        let streamed_tap_references = streamed_taps.iter().collect::<Vec<_>>();
        let streamed_conditioning = runtime
            .concatenate_axis(&streamed_tap_references, 2)
            .expect("the streamed taps should concatenate");
        runtime
            .evaluate_arrays(&[&streamed_conditioning])
            .expect("the streamed conditioning should materialize");

        assert_eq!(peak_retained_layer_count, 1);
        assert_eq!(retained_layer_count, 0);
        assert_eq!(streamed_layer_reads, vec![0, 1, 2]);
        assert_eq!(streamed_conditioning.dtype(), MlxDtype::BFloat16);
        assert_eq!(streamed_conditioning.shape(), [1, 4, 12]);
        assert_eq!(
            bf16_values_as_f32(&runtime, &streamed_conditioning),
            bf16_values_as_f32(&runtime, &complete_conditioning),
        );
    })
    .await
    .expect("reduced streamed text conditioning should finish within 115 seconds");
}

fn execute_reduced_layers<'weights>(
    runtime: &MlxRuntime,
    mut hidden_states: MlxArray,
    layer_weights: impl Iterator<Item = &'weights MlxArray>,
) -> MlxArray {
    let mut hidden_state_taps = Vec::with_capacity(REDUCED_LAYER_COUNT);
    for layer_weight in layer_weights {
        hidden_states = reduced_layer_forward(runtime, &hidden_states, layer_weight);
        runtime
            .evaluate_arrays(&[&hidden_states])
            .expect("the complete reduced hidden state should materialize");
        hidden_state_taps.push(
            hidden_states
                .retain()
                .expect("the complete reduced tap should retain"),
        );
    }
    let tap_references = hidden_state_taps.iter().collect::<Vec<_>>();
    let conditioning = runtime
        .concatenate_axis(&tap_references, 2)
        .expect("the complete reduced taps should concatenate");
    runtime
        .evaluate_arrays(&[&conditioning])
        .expect("the complete reduced conditioning should materialize");
    conditioning
}

fn reduced_layer_forward(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    layer_weight: &MlxArray,
) -> MlxArray {
    let transposed_weight = runtime
        .transpose_axes(layer_weight, &[1, 0])
        .expect("the reduced layer weight should transpose");
    let projected = runtime
        .matmul(hidden_states, &transposed_weight)
        .expect("the reduced BF16 layer should project");
    runtime
        .add(hidden_states, &projected)
        .expect("the reduced residual should build")
}

fn reduced_bf16_array(runtime: &MlxRuntime, element_count: usize, shape: &[i32]) -> MlxArray {
    reduced_bf16_array_with_offset(runtime, element_count, shape, 0)
}

fn reduced_bf16_array_with_offset(
    runtime: &MlxRuntime,
    element_count: usize,
    shape: &[i32],
    source_offset: usize,
) -> MlxArray {
    let source_values = ROMEO_AND_JULIET_SOURCE
        .bytes()
        .skip(source_offset)
        .take(element_count)
        .map(|source_byte| f32::from(source_byte) / 256.0)
        .collect::<Vec<_>>();
    let float_array = runtime
        .array_from_f32(&source_values, shape)
        .expect("the Romeo and Juliet reduced array should build");
    runtime
        .astype(&float_array, MlxDtype::BFloat16)
        .expect("the reduced array should preserve BF16 execution")
}

fn load_reduced_map(runtime: &MlxRuntime, shard_path: &std::path::Path) -> MlxSafetensors {
    runtime
        .load_safetensors(
            File::open(shard_path).expect("the reduced SafeTensors shard should reopen"),
            None,
        )
        .expect("the reduced descriptor-backed map should load")
}

fn bf16_values_as_f32(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    runtime
        .astype(array, MlxDtype::Float32)
        .expect("the BF16 evidence should cast")
        .to_vec_f32()
        .expect("the BF16 evidence should materialize on the host")
}

fn nontraditional_rope_oracle(source_values: &[f32]) -> Vec<f32> {
    let mut rotated_values = Vec::with_capacity(source_values.len());
    for (token_position, token_values) in source_values.chunks_exact(4).enumerate() {
        let first_angle = token_position as f32;
        let second_angle = token_position as f32 / 1_000.0;
        let angles = [first_angle, second_angle];
        for feature_index in 0..2 {
            rotated_values.push(
                token_values[feature_index] * angles[feature_index].cos()
                    - token_values[feature_index + 2] * angles[feature_index].sin(),
            );
        }
        for feature_index in 0..2 {
            rotated_values.push(
                token_values[feature_index] * angles[feature_index].sin()
                    + token_values[feature_index + 2] * angles[feature_index].cos(),
            );
        }
    }
    rotated_values
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the direct-MLX oracle limits should be valid"),
    )
    .expect("the pinned MLX runtime should initialize")
}
