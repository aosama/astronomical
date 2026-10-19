//! Unit tests for the notices check verdict: the regenerated notices text is
//! the contract, and a behind input digest memo must not fail the check.

use crate::notices::{HarvestVerdict, harvest_verdict};

#[test]
fn should_report_stale_when_regenerated_notices_text_differs_from_committed_text() {
    let verdict = harvest_verdict(
        b"apache-2.0\nmit\n",
        b"apache-2.0\n",
        // Memo state is irrelevant once the text differs: the committed file
        // no longer matches what the current dependency tree would generate.
        true,
    );
    assert_eq!(verdict, HarvestVerdict::NoticesTextStale);
}

#[test]
fn should_report_current_with_behind_memo_when_notices_text_matches() {
    // The recurring incident shape: a manifest edit flipped input_sha256
    // (workspace version bump, tooling metadata, comment) while the generated
    // license text stayed byte-identical. This outcome passes the check.
    let verdict = harvest_verdict(b"apache-2.0\nmit\n", b"apache-2.0\nmit\n", false);
    assert_eq!(
        verdict,
        HarvestVerdict::Current {
            memo_matches_inputs: false,
        }
    );
}

#[test]
fn should_report_current_with_current_memo_when_text_and_memo_both_match() {
    let verdict = harvest_verdict(b"apache-2.0\n", b"apache-2.0\n", true);
    assert_eq!(
        verdict,
        HarvestVerdict::Current {
            memo_matches_inputs: true,
        }
    );
}
