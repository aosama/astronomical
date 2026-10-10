//! Durable acceptance-evidence roots shared by real-model acceptance journeys.

use std::env;
use std::path::PathBuf;

/// Resolves the durable evidence root for one journey's lane so worker logs
/// and performance reports survive a timeout or panic (tempdir drops them).
pub(crate) fn acceptance_evidence_root(journey_evidence_directory_name: &str) -> PathBuf {
    if let Ok(configured_evidence_directory) =
        env::var("ASTRONOMICAL_ACCEPTANCE_EVIDENCE_DIRECTORY")
    {
        return PathBuf::from(configured_evidence_directory);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target/acceptance-evidence")
        .join(journey_evidence_directory_name)
}
