//! Unit tests for the notices check verdict: the regenerated notices text is
//! the contract, and a behind input digest memo must not fail the check.

use crate::notices::{HarvestVerdict, harvest_verdict, stale_notices_line_differences};

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

#[test]
fn should_name_the_lines_each_side_lacks_when_notices_are_stale() {
    let line_differences =
        stale_notices_line_differences("- cc 1.10.0\n- mit\n", "- cc 1.4.3\n- mit\n", 20);

    assert_eq!(line_differences, vec!["+ - cc 1.10.0", "- - cc 1.4.3"]);
}

#[test]
fn should_report_no_differences_for_identical_notices() {
    assert!(stale_notices_line_differences("a\nb\n", "a\nb\n", 20).is_empty());
}

#[test]
fn should_report_a_repeated_line_when_only_one_copy_moved() {
    let line_differences = stale_notices_line_differences("mit\nmit\n", "mit\n", 20);

    assert_eq!(line_differences, vec!["+ mit"]);
}

#[test]
fn should_cap_each_side_of_a_large_difference() {
    let regenerated_text = "added-1\nadded-2\nadded-3\n";

    let line_differences = stale_notices_line_differences(regenerated_text, "", 2);

    assert_eq!(line_differences, vec!["+ added-1", "+ added-2"]);
}
