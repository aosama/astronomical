use super::DailyCountLine;
use crate::problem_reports::MacDescription;

fn line_for_test() -> DailyCountLine {
    DailyCountLine {
        app_version: "0.2.127".to_owned(),
        mac: MacDescription {
            chip_name: "Apple M5 Pro".to_owned(),
            memory_gb: 48,
            macos_version: "27.2".to_owned(),
        },
    }
}

#[test]
fn should_read_as_the_four_values_and_nothing_else() {
    assert_eq!(
        line_for_test().readable_line(),
        "Astronomical 0.2.127 · macOS 27.2 · Apple M5 Pro · 48 GB"
    );
}

#[test]
fn should_send_the_same_four_values_with_no_identifier_or_location() {
    let envelope_body = line_for_test().envelope_body("00ab", 1_700_000_000);
    let event_line = envelope_body
        .lines()
        .nth(2)
        .expect("the event is the third line");
    let event: serde_json::Value = serde_json::from_str(event_line).expect("the event is JSON");
    assert_eq!(event["release"], "astronomical@0.2.127");
    assert_eq!(event["level"], "info");
    assert_eq!(event["fingerprint"], serde_json::json!(["daily_count"]));
    assert_eq!(
        event["tags"],
        serde_json::json!({
            "kind": "daily_count", "chip": "Apple M5 Pro", "memory_gb": "48", "macos": "27.2"
        })
    );
    assert_eq!(event["user"], serde_json::json!({"ip_address": "0.0.0.0"}));
    assert_eq!(event["sdk"]["settings"]["infer_ip"], "never");
    let mut event_keys: Vec<&String> = event.as_object().expect("an object").keys().collect();
    event_keys.sort();
    assert_eq!(
        event_keys,
        [
            "event_id",
            "fingerprint",
            "level",
            "message",
            "platform",
            "release",
            "sdk",
            "tags",
            "timestamp",
            "user"
        ],
        "a count carries no extra data and nothing that identifies the Mac"
    );
}
