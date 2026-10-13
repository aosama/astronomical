use astronomical_runtime_integration::MlxRuntimeError;
use astronomical_test_harness::offline_test;

use crate::worker_startup_error::{WorkerProcessError, WorkerStartupError};

offline_test!(
    should_include_the_exact_mlx_cause_in_a_worker_startup_failure,
    {
        let worker_process_error =
            WorkerProcessError::Startup(WorkerStartupError::ReadMlxRecommendedGpuWorkingSet(
                MlxRuntimeError::RuntimeOperation {
                    operation: "read MLX GPU recommended working set",
                    description: "synthetic device-info key failure".to_owned(),
                },
            ));

        let worker_failure_diagnostic = worker_process_error.to_string();

        assert!(
            worker_failure_diagnostic.contains("failed to read MLX recommended GPU working set")
        );
        assert!(worker_failure_diagnostic.contains("read MLX GPU recommended working set"));
        assert!(worker_failure_diagnostic.contains("synthetic device-info key failure"));
    }
);
