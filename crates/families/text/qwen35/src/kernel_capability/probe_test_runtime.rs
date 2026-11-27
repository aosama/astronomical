//! The runtime a probe test launches its probe on, or `None` on a machine
//! without a Metal GPU, where the test then passes without a launch.

use runtime_integration::{MlxMemoryLimits, MlxRuntime};

pub(in crate::kernel_capability) fn probe_test_runtime(test_name: &str) -> Option<MlxRuntime> {
    if test_harness::skip_without(test_harness::PlatformRequirement::MetalGpu, test_name) {
        return None;
    }
    Some(
        MlxRuntime::initialize(
            MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
                .expect("the probe test memory limits are valid"),
        )
        .expect("the direct MLX runtime initializes"),
    )
}
