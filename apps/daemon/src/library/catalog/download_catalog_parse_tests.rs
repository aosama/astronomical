//! Beside-source tests for the decomposed download-catalog parse units.
//!
//! Entry documents are built from a compact JSON baseline with per-test
//! overrides, mirroring `download_catalog_tests.rs`. The precedence tests pin
//! the original authored check order, which the extracted units must
//! reproduce exactly.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use super::{
    DownloadCatalogCapabilitiesDocument, DownloadCatalogEntryDocument, DownloadCatalogError,
    DownloadCatalogFamily, MAXIMUM_DESCRIPTION_BYTES, MAXIMUM_DISPLAY_NAME_BYTES,
    MAXIMUM_DOWNLOAD_CATALOG_BYTES, MAXIMUM_DOWNLOAD_CATALOG_ENTRY_COUNT,
    MAXIMUM_JAVASCRIPT_SAFE_INTEGER, MAXIMUM_QUANTIZATION_LABEL_BYTES,
    MAXIMUM_UPSTREAM_LICENSE_BYTES, parse_and_shape_check_document, parse_catalog_entry,
    parse_entry_capabilities, parse_optional_entry_description,
    parse_optional_entry_quantization_label, parse_optional_entry_upstream_license,
    validate_entry_display_metadata, validate_entry_identity,
};

const VALID_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn baseline_entry_document() -> Value {
    json!({
        "huggingface_id": "astronomical-test/example-qwen",
        "revision": VALID_REVISION,
        "display_name": "Example model",
        "family": "qwen3_5",
        "approximate_size_bytes": 4_000_000_000_u64,
        "public": true,
    })
}

fn entry_document_with(field_overrides: &[(&str, Value)]) -> DownloadCatalogEntryDocument {
    let mut catalog_entry = baseline_entry_document();
    for (field_name, field_value) in field_overrides {
        catalog_entry[field_name] = field_value.clone();
    }
    serde_json::from_value(catalog_entry).expect("the entry document should deserialize")
}

fn baseline_entry_document_unchanged() -> DownloadCatalogEntryDocument {
    serde_json::from_value(baseline_entry_document())
        .expect("the baseline entry document should deserialize")
}

fn empty_normalized_huggingface_ids() -> BTreeSet<String> {
    BTreeSet::new()
}

/// Trailing JSON whitespace is legal, so a document padded with spaces to an
/// exact byte size still parses — making the byte bound directly testable.
fn catalog_document_padded_to_byte_size(total_byte_size: usize) -> String {
    let unpadded_catalog_document = r#"{"schema_version":2,"entries":[]}"#;
    let padding_byte_count = total_byte_size - unpadded_catalog_document.len();
    format!(
        "{unpadded_catalog_document}{}",
        " ".repeat(padding_byte_count)
    )
}

fn catalog_document_with_entry_count(entry_count: usize) -> String {
    let mut entry_values = Vec::with_capacity(entry_count);
    for entry_index in 0..entry_count {
        entry_values.push(json!({
            "huggingface_id": format!("astronomical-test/example-{entry_index}"),
            "revision": VALID_REVISION,
            "display_name": "Example model",
            "family": "qwen3_5",
            "approximate_size_bytes": 4_000_000_000_u64,
            "public": true,
        }));
    }
    json!({"schema_version": 2, "entries": entry_values}).to_string()
}

#[test]
fn should_accept_documents_at_both_startup_resource_boundaries() {
    let document_at_byte_bound =
        catalog_document_padded_to_byte_size(MAXIMUM_DOWNLOAD_CATALOG_BYTES);
    let document_at_entry_bound =
        catalog_document_with_entry_count(MAXIMUM_DOWNLOAD_CATALOG_ENTRY_COUNT);

    assert!(parse_and_shape_check_document(&document_at_byte_bound).is_ok());
    assert!(parse_and_shape_check_document(&document_at_entry_bound).is_ok());
}

#[test]
fn should_reject_documents_beyond_resource_bounds_or_with_bad_shapes() {
    let document_beyond_byte_bound =
        catalog_document_padded_to_byte_size(MAXIMUM_DOWNLOAD_CATALOG_BYTES + 1);
    assert!(matches!(
        parse_and_shape_check_document(&document_beyond_byte_bound),
        Err(DownloadCatalogError::DocumentTooLarge)
    ));

    let document_beyond_entry_bound =
        catalog_document_with_entry_count(MAXIMUM_DOWNLOAD_CATALOG_ENTRY_COUNT + 1);
    assert!(matches!(
        parse_and_shape_check_document(&document_beyond_entry_bound),
        Err(DownloadCatalogError::TooManyEntries)
    ));

    assert!(matches!(
        parse_and_shape_check_document("not json at all"),
        Err(DownloadCatalogError::Parse(_))
    ));
    assert!(matches!(
        parse_and_shape_check_document(r#"{"schema_version":3,"entries":[]}"#),
        Err(DownloadCatalogError::UnsupportedSchemaVersion { schema_version: 3 })
    ));
}

#[test]
fn should_report_the_byte_bound_before_any_json_syntax_failure() {
    let oversized_non_json_document = "x".repeat(MAXIMUM_DOWNLOAD_CATALOG_BYTES + 1);

    let outcome = parse_and_shape_check_document(&oversized_non_json_document);

    assert!(matches!(
        outcome,
        Err(DownloadCatalogError::DocumentTooLarge)
    ));
}

#[test]
fn should_accept_a_first_seen_identity_and_record_its_normalization() {
    let mut normalized_huggingface_ids = empty_normalized_huggingface_ids();

    let outcome = validate_entry_identity(
        &baseline_entry_document_unchanged(),
        0,
        &mut normalized_huggingface_ids,
    );

    assert!(outcome.is_ok());
    assert!(normalized_huggingface_ids.contains("astronomical-test/example-qwen"));
}

#[test]
fn should_reject_unsafe_ids_moving_revisions_and_duplicate_identities() {
    let unsafe_identity = entry_document_with(&[("huggingface_id", json!("single-component"))]);
    let moving_revision = entry_document_with(&[("revision", json!("main"))]);

    assert!(matches!(
        validate_entry_identity(&unsafe_identity, 0, &mut empty_normalized_huggingface_ids()),
        Err(DownloadCatalogError::InvalidHuggingFaceId { entry_index: 0 })
    ));
    assert!(matches!(
        validate_entry_identity(&moving_revision, 7, &mut empty_normalized_huggingface_ids()),
        Err(DownloadCatalogError::InvalidRevision { entry_index: 7 })
    ));

    let mut already_seen = empty_normalized_huggingface_ids();
    already_seen.insert("astronomical-test/example-qwen".to_owned());
    assert!(matches!(
        validate_entry_identity(&baseline_entry_document_unchanged(), 0, &mut already_seen),
        Err(DownloadCatalogError::DuplicateHuggingFaceId)
    ));
}

#[test]
fn should_report_id_safety_and_duplicates_before_revision_failures() {
    let mut already_seen = empty_normalized_huggingface_ids();
    already_seen.insert("astronomical-test/example-qwen".to_owned());

    // An unsafe id is reported even though the revision is also invalid, and a
    // case-colliding duplicate is reported before the revision is inspected.
    let unsafe_identity_with_bad_revision = entry_document_with(&[
        ("huggingface_id", json!("organization/model/extra")),
        ("revision", json!("main")),
    ]);
    let case_collision_with_bad_revision = entry_document_with(&[
        ("huggingface_id", json!("Astronomical-Test/Example-Qwen")),
        ("revision", json!("main")),
    ]);

    assert!(matches!(
        validate_entry_identity(
            &unsafe_identity_with_bad_revision,
            0,
            &mut empty_normalized_huggingface_ids()
        ),
        Err(DownloadCatalogError::InvalidHuggingFaceId { .. })
    ));
    assert!(matches!(
        validate_entry_identity(&case_collision_with_bad_revision, 0, &mut already_seen),
        Err(DownloadCatalogError::DuplicateHuggingFaceId)
    ));
}

#[test]
fn should_accept_display_metadata_at_the_exact_metadata_boundaries() {
    let name_at_byte_bound = entry_document_with(&[(
        "display_name",
        json!("m".repeat(MAXIMUM_DISPLAY_NAME_BYTES)),
    )]);
    let size_at_integer_bound = entry_document_with(&[(
        "approximate_size_bytes",
        json!(MAXIMUM_JAVASCRIPT_SAFE_INTEGER),
    )]);

    assert!(validate_entry_display_metadata(&name_at_byte_bound, 0).is_ok());
    assert!(validate_entry_display_metadata(&size_at_integer_bound, 0).is_ok());
}

#[test]
fn should_reject_unsafe_display_names_sizes_and_private_models() {
    let invalid_display_names = [
        (
            "beyond the byte bound",
            json!("m".repeat(MAXIMUM_DISPLAY_NAME_BYTES + 1)),
        ),
        ("whitespace only", json!("   ")),
        ("with a control character", json!("Bad\nName")),
    ];
    for (case_description, display_name) in invalid_display_names {
        let invalid_entry = entry_document_with(&[("display_name", display_name)]);
        assert!(
            matches!(
                validate_entry_display_metadata(&invalid_entry, 3),
                Err(DownloadCatalogError::InvalidDisplayName { entry_index: 3 })
            ),
            "a display name {case_description} must fail closed"
        );
    }

    let invalid_sizes = [json!(0), json!(MAXIMUM_JAVASCRIPT_SAFE_INTEGER + 1)];
    for approximate_size_bytes in invalid_sizes {
        let invalid_entry =
            entry_document_with(&[("approximate_size_bytes", approximate_size_bytes)]);
        assert!(matches!(
            validate_entry_display_metadata(&invalid_entry, 0),
            Err(DownloadCatalogError::InvalidApproximateSize { .. })
        ));
    }

    let private_model = entry_document_with(&[("public", json!(false))]);
    assert!(matches!(
        validate_entry_display_metadata(&private_model, 0),
        Err(DownloadCatalogError::ModelNotPublic { .. })
    ));
}

#[test]
fn should_report_display_name_failures_before_visibility_failures() {
    let whitespace_named_private_model =
        entry_document_with(&[("display_name", json!("   ")), ("public", json!(false))]);

    let outcome = validate_entry_display_metadata(&whitespace_named_private_model, 0);

    assert!(matches!(
        outcome,
        Err(DownloadCatalogError::InvalidDisplayName { .. })
    ));
}

#[test]
fn should_default_absent_capabilities_to_no_badges() {
    let outcome = parse_entry_capabilities(None, 0);

    let capabilities = outcome.expect("absent capabilities should default");
    assert!(!capabilities.supports_reasoning);
    assert!(!capabilities.supports_vision);
    assert!(!capabilities.supports_tool_calls);
    assert_eq!(capabilities.context_window, None);
    assert_eq!(capabilities.max_output_tokens, None);
    assert!(!capabilities.supports_image_generation);
    assert!(!capabilities.supports_embeddings);
}

#[test]
fn should_reject_capabilities_documents_without_any_badge() {
    let badgeless_capabilities_document: DownloadCatalogCapabilitiesDocument =
        serde_json::from_value(json!({})).expect("an empty object should deserialize");

    let outcome = parse_entry_capabilities(Some(&badgeless_capabilities_document), 4);

    assert!(matches!(
        outcome,
        Err(DownloadCatalogError::InvalidCapabilities { entry_index: 4 })
    ));
}

#[test]
fn should_accept_capabilities_with_any_single_badge() {
    let single_badge_documents = [
        json!({"supports_reasoning": true}),
        json!({"supports_vision": true}),
        json!({"supports_tool_calls": true}),
        json!({"context_window": 8192}),
        json!({"max_output_tokens": 4096}),
        json!({"supports_image_generation": true}),
        json!({"supports_embeddings": true}),
    ];
    for capabilities_document_value in single_badge_documents {
        let capabilities_document: DownloadCatalogCapabilitiesDocument =
            serde_json::from_value(capabilities_document_value)
                .expect("the badge document should deserialize");
        let outcome = parse_entry_capabilities(Some(&capabilities_document), 0);
        assert!(outcome.is_ok());
    }

    let context_window_only: DownloadCatalogCapabilitiesDocument =
        serde_json::from_value(json!({"context_window": 8192})).expect("should deserialize");
    let capabilities = parse_entry_capabilities(Some(&context_window_only), 0)
        .expect("a context-window-only badge set should pass");
    assert_eq!(capabilities.context_window, Some(8192));
}

#[test]
fn should_accept_descriptions_at_the_exact_description_boundary() {
    let absent_description = baseline_entry_document_unchanged();
    let whitespace_only = entry_document_with(&[("description", json!("   "))]);
    let at_byte_bound =
        entry_document_with(&[("description", json!("d".repeat(MAXIMUM_DESCRIPTION_BYTES)))]);

    assert!(matches!(
        parse_optional_entry_description(&absent_description, 0),
        Ok(None)
    ));
    assert!(matches!(
        parse_optional_entry_description(&whitespace_only, 0),
        Ok(None)
    ));
    assert_eq!(
        parse_optional_entry_description(&at_byte_bound, 0).unwrap(),
        Some("d".repeat(MAXIMUM_DESCRIPTION_BYTES))
    );
}

#[test]
fn should_reject_descriptions_beyond_the_bound_or_with_control_characters() {
    let beyond_byte_bound = entry_document_with(&[(
        "description",
        json!("d".repeat(MAXIMUM_DESCRIPTION_BYTES + 1)),
    )]);
    let with_control_character = entry_document_with(&[("description", json!("Bad\nDescription"))]);

    assert!(matches!(
        parse_optional_entry_description(&beyond_byte_bound, 2),
        Err(DownloadCatalogError::InvalidDescription { entry_index: 2 })
    ));
    assert!(matches!(
        parse_optional_entry_description(&with_control_character, 0),
        Err(DownloadCatalogError::InvalidDescription { .. })
    ));
}

#[test]
fn should_accept_quantization_labels_at_the_exact_label_boundary() {
    let absent_label = baseline_entry_document_unchanged();
    let whitespace_only = entry_document_with(&[("quantization_label", json!("  "))]);
    let at_byte_bound = entry_document_with(&[(
        "quantization_label",
        json!("q".repeat(MAXIMUM_QUANTIZATION_LABEL_BYTES)),
    )]);

    assert!(matches!(
        parse_optional_entry_quantization_label(&absent_label, 0),
        Ok(None)
    ));
    assert!(matches!(
        parse_optional_entry_quantization_label(&whitespace_only, 0),
        Ok(None)
    ));
    assert_eq!(
        parse_optional_entry_quantization_label(&at_byte_bound, 0).unwrap(),
        Some("q".repeat(MAXIMUM_QUANTIZATION_LABEL_BYTES))
    );
}

#[test]
fn should_reject_quantization_labels_beyond_the_bound_or_with_control_characters() {
    let beyond_byte_bound = entry_document_with(&[(
        "quantization_label",
        json!("q".repeat(MAXIMUM_QUANTIZATION_LABEL_BYTES + 1)),
    )]);
    let with_control_character =
        entry_document_with(&[("quantization_label", json!("4-bit\nlabel"))]);

    assert!(matches!(
        parse_optional_entry_quantization_label(&beyond_byte_bound, 0),
        Err(DownloadCatalogError::InvalidQuantizationLabel { .. })
    ));
    assert!(matches!(
        parse_optional_entry_quantization_label(&with_control_character, 5),
        Err(DownloadCatalogError::InvalidQuantizationLabel { entry_index: 5 })
    ));
}

#[test]
fn should_accept_upstream_licenses_at_the_exact_license_boundary() {
    let absent_license = baseline_entry_document_unchanged();
    let whitespace_only = entry_document_with(&[("upstream_license", json!(" "))]);
    let at_byte_bound = entry_document_with(&[(
        "upstream_license",
        json!("l".repeat(MAXIMUM_UPSTREAM_LICENSE_BYTES)),
    )]);

    assert!(matches!(
        parse_optional_entry_upstream_license(&absent_license, 0),
        Ok(None)
    ));
    assert!(matches!(
        parse_optional_entry_upstream_license(&whitespace_only, 0),
        Ok(None)
    ));
    assert_eq!(
        parse_optional_entry_upstream_license(&at_byte_bound, 0).unwrap(),
        Some("l".repeat(MAXIMUM_UPSTREAM_LICENSE_BYTES))
    );
}

#[test]
fn should_reject_upstream_licenses_beyond_the_bound_or_with_control_characters() {
    let beyond_byte_bound = entry_document_with(&[(
        "upstream_license",
        json!("l".repeat(MAXIMUM_UPSTREAM_LICENSE_BYTES + 1)),
    )]);
    let with_control_character = entry_document_with(&[("upstream_license", json!("apache\n2.0"))]);

    assert!(matches!(
        parse_optional_entry_upstream_license(&beyond_byte_bound, 0),
        Err(DownloadCatalogError::InvalidUpstreamLicense { .. })
    ));
    assert!(matches!(
        parse_optional_entry_upstream_license(&with_control_character, 1),
        Err(DownloadCatalogError::InvalidUpstreamLicense { entry_index: 1 })
    ));
}

#[test]
fn should_parse_a_complete_entry_into_the_public_entry_type() {
    let complete_entry = entry_document_with(&[
        ("description", json!("A complete fictional entry")),
        (
            "capabilities",
            json!({"supports_reasoning": true, "context_window": 8192}),
        ),
        ("quantization_label", json!("4-bit (affine)")),
        ("upstream_license", json!("apache-2.0")),
        ("included_paths", json!(["LICENSE.md", "transformer/"])),
    ]);

    let outcome = parse_catalog_entry(complete_entry, 0, &mut empty_normalized_huggingface_ids());

    let catalog_entry = outcome.expect("a complete entry should parse");
    assert_eq!(
        catalog_entry.huggingface_id(),
        "astronomical-test/example-qwen"
    );
    assert_eq!(catalog_entry.revision(), VALID_REVISION);
    assert_eq!(catalog_entry.display_name(), "Example model");
    assert_eq!(catalog_entry.family(), DownloadCatalogFamily::Qwen3_5);
    assert_eq!(catalog_entry.approximate_size_bytes(), 4_000_000_000);
    assert_eq!(
        catalog_entry.description(),
        Some("A complete fictional entry")
    );
    assert!(catalog_entry.capabilities().supports_reasoning);
    assert_eq!(catalog_entry.capabilities().context_window, Some(8192));
    assert_eq!(catalog_entry.quantization_label(), Some("4-bit (affine)"));
    assert_eq!(catalog_entry.upstream_license(), Some("apache-2.0"));
    let selection = catalog_entry.download_path_selection();
    assert!(selection.includes("LICENSE.md"));
    assert!(selection.includes("transformer/model.safetensors"));
}

#[test]
fn should_report_identity_failures_before_display_description_and_capability_failures() {
    let multi_invalid_entry = entry_document_with(&[
        ("huggingface_id", json!("single-component")),
        ("display_name", json!("   ")),
        ("description", json!("Bad\nDescription")),
        ("capabilities", json!({})),
    ]);

    let outcome = parse_catalog_entry(
        multi_invalid_entry,
        0,
        &mut empty_normalized_huggingface_ids(),
    );

    assert!(matches!(
        outcome,
        Err(DownloadCatalogError::InvalidHuggingFaceId { .. })
    ));
}

#[test]
fn should_report_description_failures_before_capability_failures() {
    let invalid_description_with_badgeless_capabilities = entry_document_with(&[
        ("description", json!("Bad\nDescription")),
        ("capabilities", json!({})),
    ]);

    let outcome = parse_catalog_entry(
        invalid_description_with_badgeless_capabilities,
        0,
        &mut empty_normalized_huggingface_ids(),
    );

    assert!(matches!(
        outcome,
        Err(DownloadCatalogError::InvalidDescription { .. })
    ));
}

#[test]
fn should_report_quantization_label_failures_before_upstream_license_failures() {
    let both_labels_invalid = entry_document_with(&[
        ("quantization_label", json!("4-bit\nlabel")),
        ("upstream_license", json!("apache\n2.0")),
    ]);

    let outcome = parse_catalog_entry(
        both_labels_invalid,
        0,
        &mut empty_normalized_huggingface_ids(),
    );

    assert!(matches!(
        outcome,
        Err(DownloadCatalogError::InvalidQuantizationLabel { .. })
    ));
}

#[test]
fn should_report_included_path_failures_last_in_the_entry_order() {
    let invalid_included_paths = entry_document_with(&[("included_paths", json!(["../weights"]))]);

    let outcome = parse_catalog_entry(
        invalid_included_paths,
        9,
        &mut empty_normalized_huggingface_ids(),
    );

    assert!(matches!(
        outcome,
        Err(DownloadCatalogError::InvalidIncludedPaths { entry_index: 9 })
    ));
}
