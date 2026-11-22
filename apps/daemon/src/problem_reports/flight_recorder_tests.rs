use std::time::{Duration, Instant};

use super::{FlightEvent, FlightRecorder, MAXIMUM_KEPT_EVENTS};

#[test]
fn should_list_events_oldest_first_with_seconds_before_the_report() {
    let flight_recorder = FlightRecorder::default();
    let report_time = Instant::now();
    flight_recorder.record_at(
        report_time - Duration::from_secs(40),
        FlightEvent::ModelLoadStarted,
    );
    flight_recorder.record_at(
        report_time - Duration::from_secs(31),
        FlightEvent::ModelLoaded,
    );
    flight_recorder.record_at(
        report_time - Duration::from_secs(2),
        FlightEvent::ReplyFinished {
            prompt_token_count: 9100,
            generated_token_count: 384,
        },
    );
    assert_eq!(
        flight_recorder.lines_before(report_time),
        [
            "40 s before: model load started",
            "31 s before: model loaded",
            "2 s before: reply finished (9100 tokens in, 384 out)",
        ]
    );
}

#[test]
fn should_keep_only_the_newest_events() {
    let flight_recorder = FlightRecorder::default();
    let report_time = Instant::now();
    for _ in 0..MAXIMUM_KEPT_EVENTS {
        flight_recorder.record_at(report_time, FlightEvent::ReplyFailed);
    }
    flight_recorder.record_at(report_time, FlightEvent::EngineExited);
    let event_lines = flight_recorder.lines_before(report_time);
    assert_eq!(event_lines.len(), MAXIMUM_KEPT_EVENTS);
    assert_eq!(
        event_lines.last().map(String::as_str),
        Some("0 s before: engine exited")
    );
}

#[test]
fn should_show_the_memory_limit_in_gb_of_two_to_the_thirty_and_nothing_when_empty() {
    let flight_recorder = FlightRecorder::default();
    assert!(flight_recorder.lines_before(Instant::now()).is_empty());
    let report_time = Instant::now();
    flight_recorder.record_at(
        report_time,
        FlightEvent::MemoryLimitChanged {
            memory_limit_bytes: 39_191_576_576,
        },
    );
    assert_eq!(
        flight_recorder.lines_before(report_time),
        ["0 s before: memory limit set to 36.5 GB"]
    );
}

#[test]
fn should_never_report_a_negative_time_for_an_event_after_the_report_time() {
    let flight_recorder = FlightRecorder::default();
    let report_time = Instant::now();
    flight_recorder.record_at(
        report_time + Duration::from_secs(5),
        FlightEvent::ReplyFailed,
    );
    assert_eq!(
        flight_recorder.lines_before(report_time),
        ["0 s before: reply failed"]
    );
}
