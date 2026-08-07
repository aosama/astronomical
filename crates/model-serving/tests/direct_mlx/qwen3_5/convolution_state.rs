use astronomical_model_serving::ConvolutionState;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_checkpoint_exact_convolution_state_at_each_requested_boundary() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the convolution checkpoint test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let mixed_queries_keys_values = runtime
        .array_from_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 8, 2],
        )
        .expect("the complete convolution input should be valid");
    let mut checkpoint_aware_convolution_state = ConvolutionState::empty_with_shape(4, 2);

    let checkpoint_update = checkpoint_aware_convolution_state
        .update_with_boundary_checkpoints(&runtime, &mixed_queries_keys_values, 8, &[2, 5])
        .expect("the convolution update should retain both requested boundaries");

    assert_eq!(checkpoint_update.convolution_input.shape(), vec![1, 11, 2]);
    let expected_boundary_states = [
        vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ];
    for (boundary_convolution_state, expected_boundary_state) in checkpoint_update
        .boundary_convolution_states
        .iter()
        .zip(expected_boundary_states)
    {
        assert_eq!(
            boundary_convolution_state
                .to_vec_f32()
                .expect("the boundary convolution state should evaluate"),
            expected_boundary_state
        );
    }
    assert_eq!(
        checkpoint_aware_convolution_state
            .state()
            .expect("the final convolution state should be installed")
            .to_vec_f32()
            .expect("the final convolution state should evaluate"),
        vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    );
}
