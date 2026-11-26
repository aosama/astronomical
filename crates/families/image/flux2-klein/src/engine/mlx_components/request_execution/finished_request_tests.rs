//! Beside-source coverage for the facts a finished request keeps for its performance report.

use super::{FinishedRequest, bounded_failure_description, memory_readings};
use crate::dimensions::Flux2KleinImageDimensions;

const TRANSPORT_LIMIT_BYTES: u64 = 32 * 1024 * 1024;

fn finished_request_with(dimensions: Option<Flux2KleinImageDimensions>) -> FinishedRequest {
    FinishedRequest {
        request_id: 7,
        seed: 12_345,
        dimensions,
        start_memory: None,
        final_memory: None,
    }
}

#[test]
fn should_report_the_image_size_of_the_request() {
    let dimensions = Flux2KleinImageDimensions::validate(64, 128, TRANSPORT_LIMIT_BYTES)
        .expect("a 64 by 128 image is a valid size");

    let finished_request = finished_request_with(Some(dimensions));

    assert_eq!(finished_request.image_size_pixels(), (64, 128));
}

#[test]
fn should_report_zero_by_zero_when_the_request_never_reached_its_dimensions() {
    let finished_request = finished_request_with(None);

    assert_eq!(finished_request.image_size_pixels(), (0, 0));
}

#[test]
fn should_split_a_memory_reading_into_three_present_values() {
    assert_eq!(
        memory_readings(Some((10, 20, 30))),
        (Some(10), Some(20), Some(30))
    );
}

#[test]
fn should_report_three_absent_values_when_memory_was_not_read() {
    assert_eq!(memory_readings(None), (None, None, None));
}

#[test]
fn should_keep_a_failure_description_of_exactly_512_characters() {
    let description = "é".repeat(512);

    let bounded_description = bounded_failure_description(Some(&description));

    assert_eq!(bounded_description.as_deref(), Some(description.as_str()));
}

#[test]
fn should_cut_a_longer_failure_description_at_512_characters_not_bytes() {
    let description = "é".repeat(513);

    let bounded_description = bounded_failure_description(Some(&description))
        .expect("a present description stays present");

    assert_eq!(bounded_description.chars().count(), 512);
}

#[test]
fn should_pass_an_absent_failure_description_through() {
    assert_eq!(bounded_failure_description(None), None);
}
