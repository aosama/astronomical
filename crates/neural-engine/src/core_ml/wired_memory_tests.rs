use super::{system_wired_bytes, wired_growth_bytes};

#[test]
fn should_read_some_wired_memory_on_a_running_mac() {
    assert!(system_wired_bytes() > 0);
}

#[test]
fn should_report_the_growth_between_two_readings() {
    assert_eq!(wired_growth_bytes(100, 350), 250);
}

#[test]
fn should_report_no_growth_when_the_readings_match() {
    assert_eq!(wired_growth_bytes(100, 100), 0);
}

#[test]
fn should_report_no_growth_when_memory_was_returned() {
    assert_eq!(wired_growth_bytes(350, 100), 0);
}

#[test]
fn should_report_no_growth_when_the_later_query_failed() {
    assert_eq!(wired_growth_bytes(350, 0), 0);
}
