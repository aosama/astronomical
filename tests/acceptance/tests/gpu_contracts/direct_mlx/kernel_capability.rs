//! Direct-MLX contracts for the real custom-kernel capability probes.
//!
//! The probe journey must finish inside a bounded timeout, report the real
//! sorted-expert weighted-sum kernel as supported on a GPU that can run it,
//! and keep unprobed families fail-closed.

use std::time::Duration;

use astronomical_model_serving::{
    CustomKernelVerdict, CustomMetalKernelFamily, KernelUnsupportedReason, PerformanceAttribution,
    SortedExpertWeightedSumProbe, WorkerKernelCapabilities, worker_process_kernel_capabilities,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("kernel-capability test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

#[tokio::test]
async fn should_probe_the_real_sorted_expert_weighted_sum_kernel_once_per_process() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let bounded_probe_journey = tokio::time::timeout(Duration::from_secs(120), async {
        let runtime = test_runtime();
        let weighted_sum_probe = SortedExpertWeightedSumProbe::new(&runtime);
        let mut performance_attribution = PerformanceAttribution::disabled();

        let capabilities = WorkerKernelCapabilities::probe_custom_kernels(
            &[&weighted_sum_probe],
            &mut performance_attribution,
        );

        assert_eq!(
            capabilities.verdict(CustomMetalKernelFamily::SortedExpertWeightedSum),
            CustomKernelVerdict::Supported,
            "a GPU that executes the bounded probe with correct output values must keep the custom kernel"
        );
        assert_eq!(
            capabilities.verdict(CustomMetalKernelFamily::GatedDeltaSequence),
            CustomKernelVerdict::Unsupported(KernelUnsupportedReason::Unprobed),
            "families without a probe in this build stay fail-closed"
        );
    })
    .await;
    assert!(
        bounded_probe_journey.is_ok(),
        "the real probe journey must finish within the 120-second bound"
    );
}

#[tokio::test]
async fn should_reuse_worker_process_capabilities_across_model_loads() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let bounded_capability_journey = tokio::time::timeout(Duration::from_secs(120), async {
        let runtime = test_runtime();
        let mut performance_attribution = PerformanceAttribution::disabled();

        let first_capabilities =
            worker_process_kernel_capabilities(&runtime, &mut performance_attribution);
        let second_capabilities =
            worker_process_kernel_capabilities(&runtime, &mut performance_attribution);

        assert!(
            std::ptr::eq(first_capabilities, second_capabilities),
            "a second model load in the same worker process must reuse the retained verdicts"
        );
        assert!(
            first_capabilities
                .is_custom_kernel_supported(CustomMetalKernelFamily::SortedExpertWeightedSum)
        );
    })
    .await;
    assert!(
        bounded_capability_journey.is_ok(),
        "the process-capability journey must finish within the 120-second bound"
    );
}
