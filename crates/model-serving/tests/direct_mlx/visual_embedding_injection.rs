use astronomical_model_serving::qwen3_5_moe_inject_visual_embeddings;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_replace_chunk_image_pad_runs_with_the_matching_visual_embedding_slice() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the visual injection test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let text_embeddings = runtime
        .array_from_f32(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            &[1, 6, 2],
        )
        .expect("the text embeddings should be valid");
    let visual_embeddings = runtime
        .array_from_f32(
            &[
                100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0, 140.0, 141.0,
            ],
            &[5, 2],
        )
        .expect("the visual embeddings should be valid");
    let chunk_token_ids = [10, 248_056, 248_056, 11, 248_056, 12];

    let (injected_embeddings, injected_visual_embedding_count) =
        qwen3_5_moe_inject_visual_embeddings(
            &runtime,
            &text_embeddings,
            &chunk_token_ids,
            &visual_embeddings,
            1,
            248_056,
        )
        .expect("the chunk image-pad runs should receive ordered visual embeddings");

    assert_eq!(injected_visual_embedding_count, 3);
    assert_eq!(
        injected_embeddings
            .to_vec_f32()
            .expect("the injected embeddings should evaluate as float32"),
        vec![
            0.0, 1.0, 110.0, 111.0, 120.0, 121.0, 6.0, 7.0, 130.0, 131.0, 10.0, 11.0,
        ]
    );
}

#[tokio::test]
async fn should_continue_the_visual_embedding_cursor_across_two_prefill_chunks() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the visual injection test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let visual_embeddings = runtime
        .array_from_f32(
            &[
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            ],
            &[6, 2],
        )
        .expect("the visual embeddings should be valid");
    let first_chunk_token_ids = [10, 248_056, 248_056, 248_056];
    let first_chunk_text_embeddings = runtime
        .array_from_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 4, 2])
        .expect("the first chunk text embeddings should be valid");
    let (first_injected_embeddings, first_injected_count) = qwen3_5_moe_inject_visual_embeddings(
        &runtime,
        &first_chunk_text_embeddings,
        &first_chunk_token_ids,
        &visual_embeddings,
        0,
        248_056,
    )
    .expect("the first chunk should consume the first three visual embeddings");
    let second_chunk_token_ids = [248_056, 248_056, 248_056, 11];
    let second_chunk_text_embeddings = runtime
        .array_from_f32(
            &[200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0],
            &[1, 4, 2],
        )
        .expect("the second chunk text embeddings should be valid");
    let (second_injected_embeddings, second_injected_count) = qwen3_5_moe_inject_visual_embeddings(
        &runtime,
        &second_chunk_text_embeddings,
        &second_chunk_token_ids,
        &visual_embeddings,
        first_injected_count,
        248_056,
    )
    .expect("the second chunk should continue the visual embedding cursor");

    assert_eq!(first_injected_count, 3);
    assert_eq!(second_injected_count, 3);
    assert_eq!(
        first_injected_embeddings
            .to_vec_f32()
            .expect("the first injected embeddings should evaluate as float32"),
        vec![0.0, 1.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    );
    assert_eq!(
        second_injected_embeddings
            .to_vec_f32()
            .expect("the second injected embeddings should evaluate as float32"),
        vec![106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 206.0, 207.0]
    );
}
