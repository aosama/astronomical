//! Hermetic coverage for the completion attribution log.
//!
//! The completion attribution log captures what a chat generation emitted —
//! function names and the arguments JSON — so tool-call argument pollution and
//! foreign-dialect regressions are diagnosable instead of invisible. These
//! tests cover the bounding logic, the enabled/disabled gating, and the JSONL
//! serialization contract, mirroring the generation performance log tests.

use std::fs;

use astronomical_supervisor::{
    CompletedToolCall, CompletionArgumentsRecord, CompletionAttributionLog,
    CompletionToolCallRecord,
};

/// A literary payload used as oversized tool-call arguments so the bounding
/// path is exercised with source text rather than random tokens.
fn romeo_arguments_payload(repeats: usize) -> String {
    let passage =
        "Juliet: O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name.";
    passage.repeat(repeats)
}

#[test]
fn should_record_full_arguments_when_under_the_size_cap() {
    let record = CompletionToolCallRecord::from_arguments(
        /* tool_call_index */ 0,
        /* function_name */ "read",
        /* arguments_json */ r#"{"path":"romeo-and-juliet.md"}"#,
    );
    let CompletionArgumentsRecord {
        size_bytes,
        sha256,
        json,
        truncated,
    } = record.arguments;
    assert!(!truncated, "small arguments must not be truncated");
    assert_eq!(
        json, r#"{"path":"romeo-and-juliet.md"}"#,
        "under-cap arguments must be recorded verbatim"
    );
    assert!(!sha256.is_empty(), "a sha256 must always be recorded");
    assert_eq!(
        size_bytes,
        r#"{"path":"romeo-and-juliet.md"}"#.len(),
        "size must be the original argument length"
    );
}

#[test]
fn should_truncate_and_hash_arguments_when_over_the_size_cap() {
    let oversized_arguments = romeo_arguments_payload(/* repeats */ 1_000);
    let record = CompletionToolCallRecord::from_arguments(
        /* tool_call_index */ 1,
        /* function_name */ "find_character",
        /* arguments_json */ &oversized_arguments,
    );
    let CompletionArgumentsRecord {
        size_bytes,
        sha256,
        json,
        truncated,
    } = record.arguments;
    assert!(truncated, "over-cap arguments must be marked truncated");
    assert_eq!(
        size_bytes,
        oversized_arguments.len(),
        "size must reflect the original argument length, not the truncation"
    );
    assert!(
        json.len() < oversized_arguments.len(),
        "the recorded json must be shorter than the original"
    );
    assert!(!sha256.is_empty(), "a sha256 must always be recorded");
    // The sha256 must be of the full original arguments, not the truncation,
    // so two identical over-cap calls correlate regardless of truncation.
    let expected_sha = sha256_hex(oversized_arguments.as_bytes());
    assert_eq!(sha256, expected_sha);
}

#[test]
fn should_write_one_completion_row_with_tool_calls_when_enabled() {
    let temp_dir = tempfile::tempdir().expect("test directory should be created");
    let mut log = CompletionAttributionLog::open(temp_dir.path(), true)
        .expect("enabled completion log should open");

    log.record_completion(
        /* timestamp_millis */ 1_700_000_000_000,
        /* request_id */ 42,
        /* model_id */ "ornith-1.5-35b",
        /* completion_reason */ "tool_calls",
        &[
            CompletedToolCall {
                tool_call_index: 0,
                function_name: "read".to_owned(),
                arguments_json: r#"{"path":"romeo-and-juliet.md"}"#.to_owned(),
            },
            CompletedToolCall {
                tool_call_index: 1,
                function_name: "find_character".to_owned(),
                arguments_json: r#"{"name":"Romeo"}"#.to_owned(),
            },
        ],
    );

    let completion_path = temp_dir.path().join("completion.jsonl");
    let contents =
        fs::read_to_string(&completion_path).expect("completion log file should be readable");
    let lines: Vec<&str> = contents.trim().lines().collect();
    assert_eq!(lines.len(), 1, "should write exactly one JSON line");

    let parsed: serde_json::Value =
        serde_json::from_str(lines[0]).expect("completion log line should be valid JSON");
    assert_eq!(parsed["request_id"], 42);
    assert_eq!(parsed["model_id"], "ornith-1.5-35b");
    assert_eq!(parsed["completion_reason"], "tool_calls");
    assert_eq!(parsed["tool_calls"][0]["tool_call_index"], 0);
    assert_eq!(parsed["tool_calls"][0]["function_name"], "read");
    assert_eq!(parsed["tool_calls"][0]["arguments"]["truncated"], false);
    assert_eq!(
        parsed["tool_calls"][0]["arguments"]["json"],
        r#"{"path":"romeo-and-juliet.md"}"#
    );
    assert_eq!(parsed["tool_calls"][1]["function_name"], "find_character");
}

#[test]
fn should_write_an_empty_tool_call_list_for_end_of_sequence() {
    let temp_dir = tempfile::tempdir().expect("test directory should be created");
    let mut log = CompletionAttributionLog::open(temp_dir.path(), true)
        .expect("enabled completion log should open");

    log.record_completion(
        /* timestamp_millis */ 1_700_000_000_000,
        /* request_id */ 7,
        /* model_id */ "ornith-1.5-35b",
        /* completion_reason */ "end_of_sequence",
        &[],
    );

    let completion_path = temp_dir.path().join("completion.jsonl");
    let contents =
        fs::read_to_string(&completion_path).expect("completion log file should be readable");
    let parsed: serde_json::Value =
        serde_json::from_str(contents.trim()).expect("completion log line should be valid JSON");
    assert_eq!(parsed["completion_reason"], "end_of_sequence");
    assert!(
        parsed["tool_calls"].is_array(),
        "tool_calls must always serialize as an array"
    );
    assert_eq!(parsed["tool_calls"].as_array().map(Vec::len), Some(0));
}

#[test]
fn should_write_nothing_when_attribution_is_disabled() {
    let temp_dir = tempfile::tempdir().expect("test directory should be created");
    let mut disabled = CompletionAttributionLog::open(temp_dir.path(), false)
        .expect("disabled completion log should open");
    disabled.record_completion(
        /* timestamp_millis */ 1_700_000_000_000,
        /* request_id */ 99,
        /* model_id */ "ornith-1.5-35b",
        /* completion_reason */ "tool_calls",
        &[CompletedToolCall {
            tool_call_index: 0,
            function_name: "read".to_owned(),
            arguments_json: r#"{"path":"romeo-and-juliet.md"}"#.to_owned(),
        }],
    );
    assert!(
        !temp_dir.path().join("completion.jsonl").exists(),
        "a disabled log must not create the completion file"
    );
}

#[test]
fn should_never_leak_local_paths_into_completion_rows() {
    let temp_dir = tempfile::tempdir().expect("test directory should be created");
    let mut log = CompletionAttributionLog::open(temp_dir.path(), true)
        .expect("enabled completion log should open");

    log.record_completion(
        /* timestamp_millis */ 1_700_000_000_000,
        /* request_id */ 5,
        /* model_id */ "ornith-1.5-35b",
        /* completion_reason */ "tool_calls",
        &[CompletedToolCall {
            tool_call_index: 0,
            function_name: "read".to_owned(),
            arguments_json: r#"{"path":"romeo-and-juliet.md"}"#.to_owned(),
        }],
    );

    let completion_path = temp_dir.path().join("completion.jsonl");
    let serialized =
        fs::read_to_string(&completion_path).expect("completion log file should be readable");
    assert!(
        !serialized.contains("/fictional/"),
        "completion rows must not carry synthetic local paths"
    );
    assert!(
        !serialized.contains("Users"),
        "completion rows must not carry home-directory paths"
    );
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}
