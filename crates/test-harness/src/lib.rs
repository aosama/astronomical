//! Shared test harness: deterministic-time and timeout scaffolding, plus
//! workspace binary resolution, for the repository's test tiers.
//!
//! Vocabulary (issue #1208): the lane that runs every unit and integration
//! test is the **integration** lane: scripted and deterministic, no real
//! model artifacts, no external network; loopback and Unix Domain Socket
//! inter-process communication are local transport. Scope words name the
//! seam: **unit** (in-process, fakes), **integration** (stubs or fixture
//! binaries), **end-to-end (E2E)** (the real `astronomicald` binary over real
//! HTTP), **acceptance** (real model artifacts).
//!
//! Tests are plain Book-style functions; hang protection lives at the runner
//! level (CI job timeouts) per issue #1333. The harness keeps the helpers:
//! run_with_timeout for explicit bounded waits, and workspace binary
//! resolution.

pub mod sparse_decode_full_fit;
pub mod test_run;
pub mod timeout;
pub mod workspace;

#[cfg(feature = "flux-reference-oracle")]
pub mod flux_reference_oracle;

pub use timeout::run_with_timeout;
pub use workspace::workspace_binary;

/// Default test timeout in seconds, per the repository testing rules.
pub const DEFAULT_TEST_TIMEOUT_SECONDS: u64 = 120;

// The wrapper macros of the earlier lane design were retired with issue
// #1333: tests are plain Book-style #[test] / #[tokio::test] functions. Hang
// protection lives at the runner level (CI job timeouts); bodies that must
// bound a specific wait wrap that wait explicitly. The paused-clock contract
// stays on the attribute.

#[cfg(feature = "test-fixtures")]
pub mod fixtures;
