//! GPU-machine probes for the inference worker's startup memory policy.
//!
//! The native probe is ignored by default and requires
//! `allow_large_wired_memory_probe: true` in the explicit test-run configuration.
//! Run `cargo xtask test-run --package astronomical-inference-worker --test gpu_tests
//! --ignored --configuration PATH` to grant that permission for one run.
//! Permission checks run without touching graphics-processor state.

#[path = "gpu/wired_limit_claim_probe.rs"]
mod wired_limit_claim_probe;
