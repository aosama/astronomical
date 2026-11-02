use std::time::Duration;

use super::TimingSummary;

fn milliseconds(values: &[u64]) -> Vec<Duration> {
    let mut samples = Vec::with_capacity(values.len());
    for value in values {
        samples.push(Duration::from_millis(*value));
    }
    samples
}

#[test]
fn should_order_samples_before_taking_percentiles() {
    let summary = TimingSummary::from_samples(&milliseconds(&[9, 1, 5, 3, 7, 2, 8, 4, 6, 10]))
        .expect("samples exist");

    assert_eq!(summary.sample_count, 10);
    assert_eq!(summary.minimum, Duration::from_millis(1));
    assert_eq!(summary.median, Duration::from_millis(5));
    assert_eq!(summary.ninetieth_percentile, Duration::from_millis(9));
    assert_eq!(summary.maximum, Duration::from_millis(10));
}

#[test]
fn should_report_the_only_sample_for_every_statistic() {
    let summary = TimingSummary::from_samples(&milliseconds(&[4])).expect("one sample");

    assert_eq!(summary.minimum, summary.maximum);
    assert_eq!(summary.median, Duration::from_millis(4));
    assert_eq!(summary.ninetieth_percentile, Duration::from_millis(4));
}

#[test]
fn should_report_nothing_without_samples() {
    assert_eq!(TimingSummary::from_samples(&[]), None);
}

#[test]
fn should_convert_the_median_into_teraflops() {
    let summary = TimingSummary::from_samples(&milliseconds(&[2])).expect("one sample");

    assert!((summary.median_teraflops(4.0e10) - 20.0).abs() < 1e-9);
}
