use super::gate_verdict;

#[test]
fn should_report_a_passed_gate() {
    assert_eq!(gate_verdict(true), "PASS");
}

#[test]
fn should_report_a_failed_gate_as_a_stop() {
    assert_eq!(gate_verdict(false), "STOP");
}
