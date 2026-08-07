use astronomical_model_serving::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_collect_complete_persistent_prompt_cache_boundary_tensors_in_prompt_order() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the checkpoint collector test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let mut collector =
        Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(vec![2_048, 4_096], 2, 2_048)
            .expect("two ordered checkpoint positions should create a collector");
    let convolution_boundary_states = vec![
        runtime
            .zeros(&[1, 3, 2], MlxDtype::BFloat16)
            .expect("the first convolution boundary should be valid"),
        runtime
            .zeros(&[1, 3, 2], MlxDtype::BFloat16)
            .expect("the second convolution boundary should be valid"),
    ];
    let recurrent_boundary_states = vec![
        runtime
            .zeros(&[1, 2, 2, 2], MlxDtype::Float32)
            .expect("the first recurrent boundary should be valid"),
        runtime
            .zeros(&[1, 2, 2, 2], MlxDtype::Float32)
            .expect("the second recurrent boundary should be valid"),
    ];

    collector
        .record_linear_attention_layer(7, convolution_boundary_states, recurrent_boundary_states)
        .expect("one complete linear layer should populate both checkpoints");
    let completed_checkpoints = collector
        .complete()
        .expect("every checkpoint should contain the expected tensor count");

    assert_eq!(completed_checkpoints.len(), 2);
    assert_eq!(
        completed_checkpoints[0].completed_prefill_chunck_tokens,
        2_048
    );
    assert_eq!(
        completed_checkpoints[1].completed_prefill_chunck_tokens,
        4_096
    );
    for completed_checkpoint in completed_checkpoints {
        assert!(
            completed_checkpoint
                .recurrent_snapshot_tensors
                .contains_key("layer_7_linear.convolution")
        );
        assert!(
            completed_checkpoint
                .recurrent_snapshot_tensors
                .contains_key("layer_7_linear.gated_delta_recurrent")
        );
    }
}

#[tokio::test]
async fn should_reject_mismatched_or_incomplete_boundary_tensors() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the checkpoint collector validation limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let convolution_boundary_state = runtime
        .zeros(&[1, 3, 2], MlxDtype::BFloat16)
        .expect("the convolution boundary should be valid");
    let recurrent_boundary_state = runtime
        .zeros(&[1, 2, 2, 2], MlxDtype::Float32)
        .expect("the recurrent boundary should be valid");
    let mut collector =
        Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(vec![2_048], 4, 2_048)
            .expect("one checkpoint position should create a collector");

    assert!(
        collector
            .record_linear_attention_layer(
                7,
                vec![
                    convolution_boundary_state
                        .retain()
                        .expect("the convolution boundary should retain")
                ],
                vec![],
            )
            .is_err()
    );
    collector
        .record_linear_attention_layer(
            7,
            vec![convolution_boundary_state],
            vec![recurrent_boundary_state],
        )
        .expect("the first complete layer should record");
    assert!(collector.complete().is_err());
}
