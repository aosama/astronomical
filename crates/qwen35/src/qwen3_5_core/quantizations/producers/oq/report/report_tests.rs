use serde_json::json;

use super::*;

fn report_bytes(report: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(report).expect("synthetic report serializes")
}

#[test]
fn should_read_provenance_from_a_complete_report() {
    let provenance = OqImatrixProvenance::from_json_bytes(&report_bytes(&json!({
        "enabled": true,
        "cache_path": "synthetic/cache.npz",
        "cache_reused": false,
        "entry_count": 12,
        "calib_dataset": "synthetic_calibration",
        "applied": ["a", "b", "c"],
        "missing": ["lm_head"],
        "mismatched": [],
        "zero_count_experts": 0,
        "collection": {"dataset": "synthetic_calibration"}
    })))
    .expect("complete report should parse");

    assert_eq!(
        provenance,
        OqImatrixProvenance {
            calibration_dataset: Some("synthetic_calibration".to_owned()),
            is_imatrix_enabled: Some(true),
            entry_count: Some(12),
            applied_module_count: Some(3),
            missing_module_count: Some(1),
            mismatched_module_count: Some(0),
        }
    );
}

#[test]
fn should_accept_bare_counts_in_place_of_module_lists() {
    let provenance = OqImatrixProvenance::from_json_bytes(&report_bytes(&json!({
        "applied": 470, "missing": 1
    })))
    .expect("count-only report should parse");

    assert_eq!(provenance.applied_module_count, Some(470));
    assert_eq!(provenance.missing_module_count, Some(1));
    assert_eq!(provenance.mismatched_module_count, None);
}

#[test]
fn should_read_an_empty_object_as_all_absent() {
    let provenance =
        OqImatrixProvenance::from_json_bytes(b"{}").expect("empty report should parse");

    assert_eq!(provenance, OqImatrixProvenance::default());
}

#[test]
fn should_treat_fields_of_unexpected_types_as_absent() {
    let provenance = OqImatrixProvenance::from_json_bytes(&report_bytes(&json!({
        "calib_dataset": 7, "enabled": "yes", "entry_count": -3, "applied": "many"
    })))
    .expect("mistyped fields should not fail the read");

    assert_eq!(provenance, OqImatrixProvenance::default());
}

#[test]
fn should_reject_a_report_larger_than_the_bound() {
    let oversized_report = vec![b' '; MAXIMUM_OQ_REPORT_BYTES + 1];

    let report_error = OqImatrixProvenance::from_json_bytes(&oversized_report)
        .expect_err("oversized report must be refused");

    assert!(matches!(
        report_error,
        OqImatrixReportError::ReportTooLarge { .. }
    ));
}

#[test]
fn should_accept_a_report_exactly_at_the_bound() {
    let mut boundary_report = b"{}".to_vec();
    boundary_report.resize(MAXIMUM_OQ_REPORT_BYTES, b' ');

    assert!(OqImatrixProvenance::from_json_bytes(&boundary_report).is_ok());
}

#[test]
fn should_reject_malformed_json_and_non_object_documents() {
    assert!(matches!(
        OqImatrixProvenance::from_json_bytes(b"{not json"),
        Err(OqImatrixReportError::DeserializeReport(_))
    ));
    assert!(matches!(
        OqImatrixProvenance::from_json_bytes(b"[1, 2]"),
        Err(OqImatrixReportError::ReportIsNotAnObject)
    ));
}
