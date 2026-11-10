//! Beside-source tests for the catalog presentation texts: the short name, the
//! version label that tells builds of one model apart, and the one-line summary.

use serde_json::{Value, json};

use super::{
    AuthoredEntryPresentation, MAXIMUM_SHORT_NAME_BYTES, MAXIMUM_SUMMARY_BYTES,
    MAXIMUM_VERSION_LABEL_BYTES, parse_entry_presentation,
};
use crate::library::{DownloadCatalog, DownloadCatalogError};

const VALID_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn presented_entry(repository_name: &str, presentation_fields: &[(&str, &str)]) -> Value {
    let mut catalog_entry = json!({
        "huggingface_id": format!("astronomical-test/{repository_name}"),
        "revision": VALID_REVISION,
        "display_name": "Example model 4-bit",
        "family": "qwen3_5",
        "approximate_size_bytes": 4_000_000_000_u64,
        "approximate_memory_bytes": 2_000_000_000_u64,
        "role": "chat",
        "public": true,
    });
    for (field_name, field_text) in presentation_fields {
        catalog_entry[field_name] = json!(field_text);
    }
    catalog_entry
}

fn parse_catalog_with(catalog_entries: &[Value]) -> Result<DownloadCatalog, DownloadCatalogError> {
    DownloadCatalog::parse_json(
        &json!({"schema_version": 4, "entries": catalog_entries}).to_string(),
    )
}

#[test]
fn should_expose_the_short_name_version_label_and_summary_of_an_entry() {
    let download_catalog = parse_catalog_with(&[presented_entry(
        "example-compact",
        &[
            ("short_name", "Example 9B"),
            ("version_label", "Compact"),
            ("summary", "A quick everyday assistant."),
        ],
    )])
    .expect("an entry with all presentation texts should parse");

    let catalog_entry = &download_catalog.entries()[0];
    assert_eq!(catalog_entry.short_name(), Some("Example 9B"));
    assert_eq!(catalog_entry.version_label(), Some("Compact"));
    assert_eq!(catalog_entry.summary(), Some("A quick everyday assistant."));
}

#[test]
fn should_leave_absent_and_whitespace_only_presentation_texts_absent() {
    let download_catalog = parse_catalog_with(&[
        presented_entry("example-plain", &[]),
        presented_entry(
            "example-blank",
            &[("short_name", "  "), ("summary", "\u{20}")],
        ),
    ])
    .expect("entries without presentation texts should parse");

    for catalog_entry in download_catalog.entries() {
        assert_eq!(catalog_entry.short_name(), None);
        assert_eq!(catalog_entry.version_label(), None);
        assert_eq!(catalog_entry.summary(), None);
    }
}

#[test]
fn should_accept_each_presentation_text_at_its_byte_bound_and_reject_one_byte_more() {
    let bounded_fields = [
        ("short_name", MAXIMUM_SHORT_NAME_BYTES),
        ("summary", MAXIMUM_SUMMARY_BYTES),
    ];
    for (field_name, maximum_bytes) in bounded_fields {
        let text_at_bound = "a".repeat(maximum_bytes);
        let text_over_bound = "a".repeat(maximum_bytes + 1);
        assert!(
            parse_catalog_with(&[presented_entry("example", &[(field_name, &text_at_bound)])])
                .is_ok(),
            "{field_name} at its bound should parse"
        );
        assert!(
            parse_catalog_with(&[presented_entry(
                "example",
                &[(field_name, &text_over_bound)]
            )])
            .is_err(),
            "{field_name} over its bound should be rejected"
        );
    }
    let label_over_bound = "a".repeat(MAXIMUM_VERSION_LABEL_BYTES + 1);
    assert!(matches!(
        parse_catalog_with(&[presented_entry(
            "example",
            &[
                ("short_name", "Example"),
                ("version_label", &label_over_bound)
            ]
        )]),
        Err(DownloadCatalogError::InvalidVersionLabel { entry_index: 0 })
    ));
}

#[test]
fn should_reject_control_characters_with_the_error_of_the_offending_text() {
    assert!(matches!(
        parse_catalog_with(&[presented_entry("example", &[("short_name", "Exam\nple")])]),
        Err(DownloadCatalogError::InvalidShortName { entry_index: 0 })
    ));
    assert!(matches!(
        parse_catalog_with(&[presented_entry(
            "example",
            &[("summary", "Quick\tassistant")]
        )]),
        Err(DownloadCatalogError::InvalidSummary { entry_index: 0 })
    ));
}

#[test]
fn should_reject_a_version_label_without_a_short_name() {
    let orphan_label = AuthoredEntryPresentation {
        short_name: None,
        version_label: Some("Compact"),
        summary: None,
    };

    assert!(matches!(
        parse_entry_presentation(&orphan_label, 3),
        Err(DownloadCatalogError::InvalidVersionLabel { entry_index: 3 })
    ));
}

#[test]
fn should_accept_builds_of_one_model_that_differ_only_by_version_label() {
    let download_catalog = parse_catalog_with(&[
        presented_entry(
            "example-compact",
            &[("short_name", "Example 9B"), ("version_label", "Compact")],
        ),
        presented_entry(
            "example-sharper",
            &[("short_name", "Example 9B"), ("version_label", "Sharper")],
        ),
    ])
    .expect("two builds with different version labels should parse");

    assert_eq!(download_catalog.entry_count(), 2);
}

#[test]
fn should_reject_a_repeated_short_name_and_version_label_ignoring_case() {
    let repeated_version = parse_catalog_with(&[
        presented_entry(
            "example-compact",
            &[("short_name", "Example 9B"), ("version_label", "Compact")],
        ),
        presented_entry(
            "example-compact-again",
            &[("short_name", "example 9b"), ("version_label", "COMPACT")],
        ),
    ]);
    let repeated_unlabelled_name = parse_catalog_with(&[
        presented_entry("example-one", &[("short_name", "Example 9B")]),
        presented_entry("example-two", &[("short_name", "Example 9B")]),
    ]);

    assert!(matches!(
        repeated_version,
        Err(DownloadCatalogError::DuplicateModelVersion { entry_index: 1 })
    ));
    assert!(matches!(
        repeated_unlabelled_name,
        Err(DownloadCatalogError::DuplicateModelVersion { entry_index: 1 })
    ));
}

#[test]
fn should_give_every_bundled_entry_a_short_name_and_a_summary() {
    let bundled_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");

    for catalog_entry in bundled_catalog.entries() {
        assert!(
            catalog_entry.short_name().is_some() && catalog_entry.summary().is_some(),
            "{} needs a short name and a summary for the Models page",
            catalog_entry.huggingface_id()
        );
    }
}
