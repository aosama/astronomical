//! The one runtime every non-ignored graphics-processor contract in this test
//! binary initializes.
//!
//! The MLX runtime configures its memory limits once per process and refuses a
//! different limit afterwards, so the contracts that run together by default
//! must share one value; keeping it in one function makes that invariant
//! structural instead of a convention each file has to remember. The ignored
//! throughput measurements initialize a larger limit for complete-layer pages
//! and therefore run in their own invocation through
//! `cargo test-jangtq2-decode-throughput`, never together with these contracts.

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

/// Large enough for the materializing reference decode on a quarter-height
/// top-8 page (about four full-width uint32 temporaries per projection); the
/// fused contracts size their real-shape page against this value.
const CONTRACT_MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;

pub(super) fn contract_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(CONTRACT_MEMORY_LIMIT_BYTES, CONTRACT_MEMORY_LIMIT_BYTES)
            .expect("the jangtq2 contract memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}
