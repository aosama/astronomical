//! Beside-source tests for measured speeds: plausible whole-token rates, a
//! named Mac and a numeric version are kept; anything else is refused, never
//! shown as a figure.

use serde_json::{Value, json};

use crate::download_catalog::{DownloadCatalog, DownloadCatalogError};

const VALID_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn measured_speed_with(field_overrides: &[(&str, Value)]) -> Value {
    let mut measured_speed = json!({
        "measured_on": "Mac with an M5 Pro and 48 GB",
        "astronomical_version": "0.2.129",
        "tokens_per_second": {
            "essay_short_prompt": 96,
            "essay_long_prompt": 90,
            "code_short_prompt": 125,
            "code_long_prompt": 101
        }
    });
    for (field_name, field_value) in field_overrides {
        measured_speed[field_name] = field_value.clone();
    }
    measured_speed
}

fn parse_entry_with_speed(measured_speed: Value) -> Result<DownloadCatalog, DownloadCatalogError> {
    let catalog_json = json!({
        "schema_version": 5,
        "entries": [{
            "huggingface_id": "astronomical-test/Fast",
            "revision": VALID_REVISION,
            "family": "qwen3_5",
            "approximate_size_bytes": 4_000_000_000_u64,
            "approximate_memory_bytes": 5_000_000_000_u64,
            "role": "chat",
            "public": true,
            "measured_speed": measured_speed
        }]
    })
    .to_string();
    DownloadCatalog::parse_json(&catalog_json)
}

#[test]
fn should_keep_measured_speeds_with_where_they_were_measured() {
    let parsed_catalog =
        parse_entry_with_speed(measured_speed_with(&[])).expect("a measured speed should parse");
    let measured_speed = parsed_catalog.entries()[0]
        .measured_speed()
        .expect("the entry should carry its speed");

    assert_eq!(measured_speed.measured_on(), "Mac with an M5 Pro and 48 GB");
    assert_eq!(measured_speed.astronomical_version(), "0.2.129");
    assert_eq!(measured_speed.tokens_per_second().code_short_prompt, 125);
    assert_eq!(measured_speed.tokens_per_second().essay_long_prompt, 90);
}

#[test]
fn should_accept_rates_at_both_ends_of_the_plausible_range() {
    for edge_rate in [1, 1_000] {
        let measured_speed = measured_speed_with(&[(
            "tokens_per_second",
            json!({"essay_short_prompt": edge_rate, "essay_long_prompt": edge_rate, "code_short_prompt": edge_rate, "code_long_prompt": edge_rate}),
        )]);
        assert!(
            parse_entry_with_speed(measured_speed).is_ok(),
            "{edge_rate} tokens a second is plausible"
        );
    }
}

#[test]
fn should_refuse_a_zero_or_implausible_rate() {
    for implausible_rate in [0, 1_001] {
        let measured_speed = measured_speed_with(&[(
            "tokens_per_second",
            json!({"essay_short_prompt": 96, "essay_long_prompt": implausible_rate, "code_short_prompt": 125, "code_long_prompt": 101}),
        )]);
        let parse_error =
            parse_entry_with_speed(measured_speed).expect_err("an implausible rate is refused");
        assert!(matches!(
            parse_error,
            DownloadCatalogError::InvalidMeasuredSpeed { entry_index: 0 }
        ));
    }
}

#[test]
fn should_refuse_an_unnamed_mac_or_a_version_that_is_not_numeric() {
    for (field_name, field_value) in [
        ("measured_on", json!(" ")),
        ("measured_on", json!("M".repeat(65))),
        ("astronomical_version", json!("latest")),
        ("astronomical_version", json!("0..129")),
    ] {
        let parse_error = parse_entry_with_speed(measured_speed_with(&[(field_name, field_value)]))
            .expect_err("a speed must say where it was measured");
        assert!(matches!(
            parse_error,
            DownloadCatalogError::InvalidMeasuredSpeed { entry_index: 0 }
        ));
    }
}

#[test]
fn should_refuse_a_missing_task_or_a_fractional_rate() {
    let missing_task = measured_speed_with(&[(
        "tokens_per_second",
        json!({"essay_short_prompt": 96, "essay_long_prompt": 90, "code_short_prompt": 125}),
    )]);
    let fractional_rate = measured_speed_with(&[(
        "tokens_per_second",
        json!({"essay_short_prompt": 96.5, "essay_long_prompt": 90, "code_short_prompt": 125, "code_long_prompt": 101}),
    )]);

    assert!(matches!(
        parse_entry_with_speed(missing_task),
        Err(DownloadCatalogError::Parse(_))
    ));
    assert!(matches!(
        parse_entry_with_speed(fractional_rate),
        Err(DownloadCatalogError::Parse(_))
    ));
}
