//! Durable acceptance-evidence roots shared by real-model acceptance journeys.

use std::path::PathBuf;

use astronomical_test_harness::test_run::TestRunContext;

/// Resolves the durable evidence root for one journey's lane so worker logs
/// and performance reports survive a timeout or panic (tempdir drops them).
/// The package's binaries share this module, and each binary consumes a
/// subset, so unused-in-this-binary is expected rather than a defect.
#[allow(dead_code)]
pub(crate) fn acceptance_evidence_root(journey_evidence_directory_name: &str) -> PathBuf {
    if let Some(configured_evidence_directory) =
        TestRunContext::current().configuration.evidence_directory
    {
        return configured_evidence_directory;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target/acceptance-evidence")
        .join(journey_evidence_directory_name)
}
