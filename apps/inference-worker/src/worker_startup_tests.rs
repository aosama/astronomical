use astronomical_test_harness::offline_test;

use crate::worker_startup::derive_serving_mlx_memory_ceiling_bytes;
use crate::worker_startup_error::WorkerStartupError;

offline_test!(
    should_reject_working_sets_that_cannot_produce_a_positive_serving_ceiling,
    {
        for recommended_gpu_working_set_bytes in [0, 1_000_000_000, 2 * 1024 * 1024 * 1024] {
            let startup_error =
                derive_serving_mlx_memory_ceiling_bytes(recommended_gpu_working_set_bytes)
                    .expect_err("a zero serving ceiling must fail worker startup");

            assert!(matches!(
                &startup_error,
                WorkerStartupError::RecommendedGpuWorkingSetTooSmall
            ));
            assert!(
                startup_error
                    .to_string()
                    .contains("cannot produce a positive serving memory ceiling")
            );
        }
    }
);

offline_test!(
    should_accept_a_working_set_that_produces_a_positive_serving_ceiling,
    {
        assert_eq!(
            derive_serving_mlx_memory_ceiling_bytes(2 * 1024 * 1024 * 1024 + 1)
                .expect("a positive serving ceiling should be derived"),
            1
        );
    }
);
