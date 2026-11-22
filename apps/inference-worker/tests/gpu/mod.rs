//! GPU-machine probe targets for the inference worker.

mod wired_claim_probe_permission;
// The probe walks Metal's wired residency, which exists only on macOS.
#[cfg(target_os = "macos")]
mod wired_limit_claim_probe;
