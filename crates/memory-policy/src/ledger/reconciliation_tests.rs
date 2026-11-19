use super::LedgerReconciliation;

#[test]
fn should_report_no_disagreement_when_the_reading_matches() {
    let reconciliation = LedgerReconciliation::new(1_000, 1_000, 400, 400);
    assert_eq!(reconciliation.unaccounted_bytes, 0);
    assert_eq!(reconciliation.overcounted_bytes, 0);
    assert_eq!(reconciliation.disagreement_bytes(), 0);
    assert_eq!(reconciliation.kept_over_capacity_bytes, 0);
}

#[test]
fn should_name_memory_in_use_beyond_what_the_ledger_expects() {
    let reconciliation = LedgerReconciliation::new(1_000, 1_250, 400, 500);
    assert_eq!(reconciliation.unaccounted_bytes, 250);
    assert_eq!(reconciliation.overcounted_bytes, 0);
    assert_eq!(reconciliation.disagreement_bytes(), 250);
}

#[test]
fn should_name_memory_the_ledger_expects_that_is_not_in_use() {
    let reconciliation = LedgerReconciliation::new(1_000, 700, 400, 500);
    assert_eq!(reconciliation.unaccounted_bytes, 0);
    assert_eq!(reconciliation.overcounted_bytes, 300);
    assert_eq!(reconciliation.disagreement_bytes(), 300);
}

#[test]
fn should_name_weights_kept_beyond_the_capacity_and_nothing_at_the_boundary() {
    assert_eq!(
        LedgerReconciliation::new(0, 0, 501, 500).kept_over_capacity_bytes,
        1
    );
    assert_eq!(
        LedgerReconciliation::new(0, 0, 500, 500).kept_over_capacity_bytes,
        0
    );
    assert_eq!(
        LedgerReconciliation::new(0, 0, 0, 500).kept_over_capacity_bytes,
        0
    );
}
