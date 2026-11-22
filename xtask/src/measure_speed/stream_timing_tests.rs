use std::time::Duration;

use super::StreamTiming;

fn token_line(text: &str) -> String {
    format!(r#"data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}"#)
}

#[test]
fn should_measure_reply_speed_between_the_first_and_last_token() {
    let mut stream_timing = StreamTiming::default();
    stream_timing.record_line(
        r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#,
        Duration::from_millis(100),
    );
    stream_timing.record_line(&token_line("Hello"), Duration::from_millis(2_000));
    stream_timing.record_line(
        r#"data: {"choices":[{"delta":{"reasoning_content":"hm"}}]}"#,
        Duration::from_millis(2_500),
    );
    stream_timing.record_line(&token_line("!"), Duration::from_millis(12_000));
    stream_timing.record_line(
        r#"data: {"choices":[],"usage":{"prompt_tokens":10122,"completion_tokens":501}}"#,
        Duration::from_millis(12_010),
    );
    stream_timing.record_line("data: [DONE]", Duration::from_millis(12_020));

    let measurement = stream_timing.measurement();

    assert_eq!(measurement.prompt_tokens, 10_122);
    assert_eq!(measurement.completion_tokens, 501);
    assert!((measurement.seconds_to_first_token - 2.0).abs() < 1e-9);
    assert!(
        (measurement.reply_tokens_per_second.expect("a rate") - 50.0).abs() < 1e-9,
        "500 tokens over 10 seconds"
    );
}

#[test]
fn should_report_no_rate_when_fewer_than_two_tokens_arrived() {
    let mut stream_timing = StreamTiming::default();
    stream_timing.record_line(&token_line("Hi"), Duration::from_millis(500));
    stream_timing.record_line(
        r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}"#,
        Duration::from_millis(510),
    );

    assert_eq!(stream_timing.measurement().reply_tokens_per_second, None);
}

#[test]
fn should_ignore_comments_malformed_lines_and_empty_deltas() {
    let mut stream_timing = StreamTiming::default();
    for ignored_line in [
        ": keep-alive",
        "event: message",
        "data: {not json",
        r#"data: {"choices":[{"delta":{"content":""}}]}"#,
    ] {
        stream_timing.record_line(ignored_line, Duration::from_millis(100));
    }

    let measurement = stream_timing.measurement();
    assert_eq!(measurement.completion_tokens, 0);
    assert!(measurement.seconds_to_first_token.abs() < 1e-9);
    assert_eq!(measurement.reply_tokens_per_second, None);
}
