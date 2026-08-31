use astronomical_model_serving::{
    qwen3_5_aggregate_speculative_prefill_attention_weights,
    qwen3_5_select_speculative_prefill_token_positions_on_gpu,
};
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_pool_each_attention_head_before_selecting_the_maximum_head_scores() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the speculative-prefill test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let layer_head_attention_weights = runtime
        .array_from_f32(&[0.0, 0.0, 9.0, 8.0, 0.0, 0.0], &[2, 1, 3])
        .expect("the controlled attention weights should be valid");

    let importance_scores = qwen3_5_aggregate_speculative_prefill_attention_weights(
        &runtime,
        &layer_head_attention_weights,
        3,
    )
    .expect("the attention weights should aggregate");
    let importance_score_values = importance_scores
        .to_vec_f32()
        .expect("the importance scores should evaluate as float32");

    let expected_importance_scores = [8.0 / 3.0, 3.0, 3.0];
    for (importance_score, expected_importance_score) in importance_score_values
        .iter()
        .zip(expected_importance_scores)
    {
        assert!((importance_score - expected_importance_score).abs() < 1e-5);
    }
}

#[tokio::test]
async fn should_apply_native_mlx_rope_at_each_selected_prompt_position() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the rotary-embedding test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let float_input = runtime
        .array_from_f32(
            &[
                0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 2.25, -2.5, 2.75, -3.0,
            ],
            &[1, 1, 3, 4],
        )
        .expect("the controlled rotary input should be valid");
    let input = runtime
        .astype(&float_input, MlxDtype::BFloat16)
        .expect("the rotary input should use model precision");
    let selected_prompt_position_offsets = runtime
        .array_from_i32(&[5, 6, 7], &[3])
        .expect("the selected prompt positions should be valid");

    let actual_output = runtime
        .rope_with_token_position_offsets(&input, &selected_prompt_position_offsets, 4, 10_000.0)
        .expect("native MLX should apply each selected prompt position");
    let mut expected_token_outputs = Vec::new();
    for (token_index, prompt_position_offset) in [5, 6, 7].into_iter().enumerate() {
        let token_start = i32::try_from(token_index).expect("the test token index should fit");
        let token_input = runtime
            .slice(
                &input,
                &[0, 0, token_start, 0],
                &[1, 1, token_start + 1, 4],
                &[1, 1, 1, 1],
            )
            .expect("the test token should slice");
        expected_token_outputs.push(
            runtime
                .rope(&token_input, 4, 10_000.0, prompt_position_offset)
                .expect("the scalar native MLX rotary embedding should build"),
        );
    }
    let expected_token_output_references = expected_token_outputs.iter().collect::<Vec<_>>();
    let expected_output = runtime
        .concatenate_axis(&expected_token_output_references, 2)
        .expect("the scalar native rotary outputs should concatenate");
    let actual_output = runtime
        .astype(&actual_output, MlxDtype::Float32)
        .expect("the dynamic native rotary output should cast to float32");
    let expected_output = runtime
        .astype(&expected_output, MlxDtype::Float32)
        .expect("the scalar native rotary output should cast to float32");

    assert_eq!(
        actual_output
            .to_vec_f32()
            .expect("the dynamic native rotary output should evaluate"),
        expected_output
            .to_vec_f32()
            .expect("the scalar native rotary output should evaluate"),
    );
}

#[tokio::test]
async fn should_select_speculative_prefill_chunks_on_the_gpu() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the selection test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let importance_scores = runtime
        .array_from_f32(
            &[
                0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3,
            ],
            &[16],
        )
        .expect("the controlled importance scores should be valid");

    let selected_token_positions = qwen3_5_select_speculative_prefill_token_positions_on_gpu(
        &runtime,
        &importance_scores,
        50,
        4,
        4,
    )
    .expect("GPU speculative-prefill selection should succeed");
    let selected_token_positions = runtime
        .astype(&selected_token_positions, MlxDtype::UInt32)
        .expect("selected positions should cast to uint32");

    assert_eq!(
        runtime
            .copy_u32_values(&selected_token_positions)
            .expect("the bounded selected positions should copy"),
        vec![4, 5, 6, 7, 12, 13, 14, 15],
    );

    let partial_final_chunk_scores = runtime
        .array_from_f32(&[0.1, 0.2, 0.3, 0.4, 0.5], &[5])
        .expect("the partial final selection chunk should be valid");
    let partial_final_chunk_positions = qwen3_5_select_speculative_prefill_token_positions_on_gpu(
        &runtime,
        &partial_final_chunk_scores,
        100,
        3,
        1,
    )
    .expect("GPU selection should remove padded final positions");
    let partial_final_chunk_positions = runtime
        .astype(&partial_final_chunk_positions, MlxDtype::UInt32)
        .expect("partial final positions should cast to uint32");
    assert_eq!(
        runtime
            .copy_u32_values(&partial_final_chunk_positions)
            .expect("the bounded partial final positions should copy"),
        vec![0, 1, 2, 3, 4],
    );

    let unaligned_mandatory_trailing_scores = runtime
        .array_from_f32(&[0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], &[10])
        .expect("the unaligned mandatory trailing scores should upload");
    let unaligned_mandatory_trailing_positions =
        qwen3_5_select_speculative_prefill_token_positions_on_gpu(
            &runtime,
            &unaligned_mandatory_trailing_scores,
            10,
            4,
            5,
        )
        .expect("GPU selection should retain every unaligned mandatory trailing token");
    let unaligned_mandatory_trailing_positions = runtime
        .astype(&unaligned_mandatory_trailing_positions, MlxDtype::UInt32)
        .expect("unaligned mandatory trailing positions should cast to uint32");
    assert_eq!(
        runtime
            .copy_u32_values(&unaligned_mandatory_trailing_positions)
            .expect("unaligned mandatory trailing positions should copy"),
        (4_u32..10).collect::<Vec<_>>(),
    );
}

#[tokio::test]
async fn should_gather_sparse_prompt_token_indices_on_the_gpu() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the sparse input test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let full_prompt_token_indices = runtime
        .array_from_i32(&[11, 22, 33, 44, 55], &[1, 5])
        .expect("the full prompt token indices should be valid");
    let selected_prompt_position_offsets = runtime
        .array_from_i32(&[1, 4], &[2])
        .expect("the selected prompt positions should be valid");

    let selected_token_indices = runtime
        .take_axis(
            &full_prompt_token_indices,
            &selected_prompt_position_offsets,
            1,
        )
        .expect("MLX should gather the selected prompt token indices");
    let selected_token_indices_as_float = runtime
        .astype(&selected_token_indices, MlxDtype::Float32)
        .expect("the gathered token indices should cast for verification");

    assert_eq!(selected_token_indices.shape(), vec![1, 2]);
    assert_eq!(
        selected_token_indices_as_float
            .to_vec_f32()
            .expect("the gathered token indices should evaluate"),
        vec![22.0, 55.0],
    );
}
