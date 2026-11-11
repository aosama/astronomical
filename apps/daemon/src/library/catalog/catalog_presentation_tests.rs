//! Beside-source tests for the catalog presentation: one name per model, the
//! model ID, and the one-line summary that says what the model is for.

use serde_json::{Value, json};

use super::{MAXIMUM_SUMMARY_BYTES, parse_entry_presentation};
use crate::library::{DownloadCatalog, DownloadCatalogError};

const VALID_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn catalog_entry_with(huggingface_id: &str, extra_fields: &[(&str, &str)]) -> Value {
    let mut catalog_entry = json!({
        "huggingface_id": huggingface_id,
        "revision": VALID_REVISION,
        "family": "qwen3_5",
        "approximate_size_bytes": 4_000_000_000_u64,
        "approximate_memory_bytes": 2_000_000_000_u64,
        "role": "chat",
        "public": true,
    });
    for (field_name, field_text) in extra_fields {
        catalog_entry[field_name] = json!(field_text);
    }
    catalog_entry
}

fn parse_catalog_with(catalog_entries: &[Value]) -> Result<DownloadCatalog, DownloadCatalogError> {
    DownloadCatalog::parse_json(
        &json!({"schema_version": 5, "entries": catalog_entries}).to_string(),
    )
}

#[test]
fn should_name_an_entry_by_the_leaf_of_its_hugging_face_identity() {
    let download_catalog = parse_catalog_with(&[catalog_entry_with(
        "example-organization/Example-9B-MLX-6bit-vision",
        &[("summary", "A quick everyday assistant.")],
    )])
    .expect("an entry with a summary should parse");

    let catalog_entry = &download_catalog.entries()[0];
    assert_eq!(catalog_entry.model_id(), "Example-9B-MLX-6bit-vision");
    assert_eq!(catalog_entry.summary(), Some("A quick everyday assistant."));
}

/// The model ID is the only name people see, so two organizations publishing
/// one repository name would be two models no one could tell apart. Case is
/// ignored because the folders they download into would collide on a Mac.
#[test]
fn should_reject_a_model_id_an_earlier_entry_already_uses_ignoring_case() {
    for later_huggingface_id in [
        "second-organization/Example-9B",
        "second-organization/example-9b",
    ] {
        let parse_result = parse_catalog_with(&[
            catalog_entry_with("first-organization/Example-9B", &[]),
            catalog_entry_with(later_huggingface_id, &[]),
        ]);

        assert!(
            matches!(
                parse_result,
                Err(DownloadCatalogError::DuplicateModelId { entry_index: 1 })
            ),
            "{later_huggingface_id} must collide with first-organization/Example-9B: {parse_result:?}"
        );
    }
}

#[test]
fn should_accept_model_ids_that_differ_only_after_a_shared_prefix() {
    let download_catalog = parse_catalog_with(&[
        catalog_entry_with("example-organization/Example-9B", &[]),
        catalog_entry_with("example-organization/Example-9B-vision", &[]),
    ])
    .expect("distinct model IDs should parse");

    let model_ids: Vec<&str> = download_catalog
        .entries()
        .iter()
        .map(super::super::DownloadCatalogEntry::model_id)
        .collect();
    assert_eq!(model_ids, ["Example-9B", "Example-9B-vision"]);
}

/// The names this catalog used to carry are gone from its schema; the parser
/// refuses unknown fields, so a catalog still carrying one fails loudly
/// instead of silently showing a second name nowhere.
#[test]
fn should_reject_each_retired_name_field() {
    for retired_field in ["display_name", "short_name", "version_label"] {
        let parse_result = parse_catalog_with(&[catalog_entry_with(
            "example-organization/Example-9B",
            &[(retired_field, "Example 9B")],
        )]);

        assert!(
            matches!(parse_result, Err(DownloadCatalogError::Parse(_))),
            "{retired_field} must be refused: {parse_result:?}"
        );
    }
}

#[test]
fn should_leave_an_absent_or_whitespace_only_summary_absent() {
    for authored_summary in [None, Some(""), Some("  "), Some("\u{20}")] {
        let presentation = parse_entry_presentation(authored_summary, 0)
            .expect("an absent or blank summary is allowed");

        assert_eq!(presentation.summary, None, "{authored_summary:?}");
    }
}

#[test]
fn should_accept_a_summary_at_its_byte_bound_and_reject_one_byte_more() {
    let summary_at_bound = "a".repeat(MAXIMUM_SUMMARY_BYTES);
    let summary_over_bound = "a".repeat(MAXIMUM_SUMMARY_BYTES + 1);

    assert!(parse_entry_presentation(Some(&summary_at_bound), 0).is_ok());
    assert!(matches!(
        parse_entry_presentation(Some(&summary_over_bound), 3),
        Err(DownloadCatalogError::InvalidSummary { entry_index: 3 })
    ));
}

#[test]
fn should_reject_a_summary_holding_control_characters() {
    for summary_with_control in ["Quick\tassistant", "Quick\nassistant"] {
        assert!(matches!(
            parse_entry_presentation(Some(summary_with_control), 0),
            Err(DownloadCatalogError::InvalidSummary { entry_index: 0 })
        ));
    }
}

/// Every model the release ships must keep a distinct name; this reads the
/// bundled catalog itself rather than a fixture.
#[test]
fn should_give_every_bundled_catalog_model_a_distinct_model_id() {
    let bundled_catalog = DownloadCatalog::load_bundled().expect("the bundled catalog parses");

    let mut model_ids: Vec<String> = bundled_catalog
        .entries()
        .iter()
        .map(|catalog_entry| catalog_entry.model_id().to_ascii_lowercase())
        .collect();
    let model_count = model_ids.len();
    model_ids.sort();
    model_ids.dedup();

    assert!(model_count > 0);
    assert_eq!(model_ids.len(), model_count);
}
