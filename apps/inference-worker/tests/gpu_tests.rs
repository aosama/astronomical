//! GPU-machine probes for the inference worker's startup memory policy.
//!
//! Everything under this target touches real GPU state on the host and is
//! opt-in through `ASTRONOMICAL_RUN_WIRED_CLAIM_PROBE=1`; without the
//! variable the probes skip so offline suites stay sealed from machine
//! state.

#[path = "gpu/wired_limit_claim_probe.rs"]
mod wired_limit_claim_probe;
