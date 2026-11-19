//! The small MLX runtime the GPU tests in this crate share.
//!
//! Half a gigabyte is ample for the synthetic layers these tests build and
//! keeps a runaway test from taking the machine's memory.

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

pub fn initialize_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}
