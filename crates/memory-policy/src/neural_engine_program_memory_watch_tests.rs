use std::time::Duration;

use super::ProgramMemoryWatch;

const GIBIBYTE: u64 = 1024 * 1024 * 1024;
const MEBIBYTE: u64 = 1024 * 1024;

const fn at_milliseconds(milliseconds: u64) -> Duration {
    Duration::from_millis(milliseconds)
}

#[test]
fn should_not_settle_before_the_driver_has_wired_the_weights() {
    let mut watch = ProgramMemoryWatch::new(10 * GIBIBYTE);

    assert!(!watch.observe(0, at_milliseconds(0)));
    assert!(!watch.observe(0, at_milliseconds(5000)));
    assert!(!watch.observe(9 * GIBIBYTE, at_milliseconds(6000)));
    assert!(!watch.observe(9 * GIBIBYTE, at_milliseconds(9000)));
}

#[test]
fn should_settle_once_the_reading_stays_flat_above_the_weights() {
    let mut watch = ProgramMemoryWatch::new(10 * GIBIBYTE);

    assert!(!watch.observe(13 * GIBIBYTE, at_milliseconds(1000)));
    assert!(!watch.observe(13 * GIBIBYTE, at_milliseconds(2499)));
    assert!(watch.observe(13 * GIBIBYTE, at_milliseconds(2500)));
    assert_eq!(watch.wired_bytes(), 13 * GIBIBYTE);
}

#[test]
fn should_settle_at_a_reading_of_exactly_the_weights() {
    let mut watch = ProgramMemoryWatch::new(10 * GIBIBYTE);

    watch.observe(10 * GIBIBYTE, at_milliseconds(0));

    assert!(watch.observe(10 * GIBIBYTE, at_milliseconds(1500)));
    assert_eq!(watch.wired_bytes(), 10 * GIBIBYTE);
}

#[test]
fn should_wait_again_after_a_further_rise() {
    let mut watch = ProgramMemoryWatch::new(10 * GIBIBYTE);
    watch.observe(11 * GIBIBYTE, at_milliseconds(1000));

    assert!(!watch.observe(13 * GIBIBYTE, at_milliseconds(2400)));
    assert!(!watch.observe(13 * GIBIBYTE, at_milliseconds(3000)));
    assert!(watch.observe(13 * GIBIBYTE, at_milliseconds(3900)));
    assert_eq!(watch.wired_bytes(), 13 * GIBIBYTE);
}

#[test]
fn should_keep_the_highest_reading_when_memory_falls_back() {
    let mut watch = ProgramMemoryWatch::new(10 * GIBIBYTE);
    watch.observe(13 * GIBIBYTE, at_milliseconds(1000));

    assert!(watch.observe(12 * GIBIBYTE, at_milliseconds(2500)));
    assert_eq!(watch.wired_bytes(), 13 * GIBIBYTE);
}

#[test]
fn should_ignore_a_rise_of_exactly_a_sixty_fourth_of_large_weights() {
    let mut watch = ProgramMemoryWatch::new(64 * GIBIBYTE);
    watch.observe(70 * GIBIBYTE, at_milliseconds(1000));

    assert!(watch.observe(71 * GIBIBYTE, at_milliseconds(2500)));
    assert_eq!(watch.wired_bytes(), 70 * GIBIBYTE);
}

#[test]
fn should_ignore_ordinary_churn_on_small_weights() {
    let mut watch = ProgramMemoryWatch::new(600 * MEBIBYTE);
    watch.observe(1500 * MEBIBYTE, at_milliseconds(1000));

    assert!(watch.observe(1550 * MEBIBYTE, at_milliseconds(2500)));
    assert_eq!(watch.wired_bytes(), 1500 * MEBIBYTE);
}

#[test]
fn should_fall_back_to_the_weights_when_nothing_was_measured() {
    let watch = ProgramMemoryWatch::new(10 * GIBIBYTE);

    assert_eq!(watch.wired_bytes(), 10 * GIBIBYTE);
}

#[test]
fn should_fall_back_to_the_weights_when_growth_reads_lower() {
    let mut watch = ProgramMemoryWatch::new(10 * GIBIBYTE);
    watch.observe(4 * GIBIBYTE, at_milliseconds(1000));

    assert_eq!(watch.wired_bytes(), 10 * GIBIBYTE);
}
