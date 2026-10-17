//! Shared test harness: deterministic-time and timeout scaffolding, plus
//! workspace binary resolution, for the repository's test tiers.
//!
//! Vocabulary (issue #1208): **offline** names the property — sealed and
//! deterministic, no real network beyond localhost, no real model artifacts,
//! everything scripted. Scope words name the seam: **unit** (in-process,
//! fakes), **integration** (stubs or fixture binaries), **end-to-end (E2E)**
//! (the real `astronomicald` binary over real HTTP), **acceptance** (real
//! model artifacts).
//!
//! Tests are plain Book-style functions; hang protection lives at the runner
//! level (CI job timeouts) per issue #1333. The harness keeps the physics
//! helpers: run_with_timeout for explicit bounded waits, ensure_network_denied
//! for network-denial contracts, and workspace binary resolution.

pub mod network_denial;
pub mod timeout;
pub mod workspace;

#[cfg(feature = "flux-reference-oracle")]
pub mod flux_reference_oracle;

pub use network_denial::ensure_network_denied;
pub use timeout::run_with_timeout;
pub use workspace::workspace_binary;

/// Default test timeout in seconds, per the repository testing rules.
pub const DEFAULT_TEST_TIMEOUT_SECONDS: u64 = 120;

// The offline_* wrapper macros were retired with issue #1333: tests are plain
// Book-style #[test] / #[tokio::test] functions. Hang protection lives at the
// runner level (CI job timeouts); bodies that must bound a specific wait wrap
// that wait explicitly. Network-denial tests call ensure_network_denied()
// directly; the paused-clock contract stays on the attribute.

#[cfg(feature = "test-fixtures")]
pub mod fixtures;
