//! Beside-source tests for the build links: a build names its lead by model ID
//! and says what sets it apart, and the whole catalog is checked so every build
//! folds under a lead listed in the same place.

use serde_json::{Value, json};

use crate::download_catalog::{DownloadCatalog, DownloadCatalogError};

use super::DownloadCatalogBuildDifference;

const VALID_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn catalog_entry(model_name: &str, field_overrides: &[(&str, Value)]) -> Value {
    let mut entry_value = json!({
        "huggingface_id": format!("astronomical-test/{model_name}"),
        "revision": VALID_REVISION,
        "family": "qwen3_5",
        "approximate_size_bytes": 4_000_000_000_u64,
        "approximate_memory_bytes": 5_000_000_000_u64,
        "role": "chat",
        "public": true,
    });
    for (field_name, field_value) in field_overrides {
        entry_value[field_name] = field_value.clone();
    }
    entry_value
}

fn parse_catalog(entry_values: Vec<Value>) -> Result<DownloadCatalog, DownloadCatalogError> {
    let catalog_json = json!({ "schema_version": 5, "entries": entry_values }).to_string();
    DownloadCatalog::parse_json(&catalog_json)
}

fn build_fields(lead_model_id: &str, difference: &str) -> [(&'static str, Value); 2] {
    [
        ("build_of", json!(lead_model_id)),
        ("build_difference", json!(difference)),
    ]
}

#[test]
fn should_link_a_build_to_a_lead_authored_after_it() {
    let parsed_catalog = parse_catalog(vec![
        catalog_entry("Ornith-6bit", &build_fields("Ornith-4bit", "more_detail")),
        catalog_entry("Ornith-4bit", &[("role", json!("coding"))]),
    ])
    .expect("a chat build under a coding lead should parse");

    let build = parsed_catalog.entries()[0]
        .build()
        .expect("the first entry should be a build");
    assert_eq!(build.lead_model_id(), "Ornith-4bit");
    assert_eq!(
        build.difference(),
        DownloadCatalogBuildDifference::MoreDetail
    );
    assert_eq!(build.difference().as_str(), "more_detail");
    assert!(parsed_catalog.entries()[1].build().is_none());
}

#[test]
fn should_read_every_difference_kind() {
    for (wire_name, difference) in [
        ("more_detail", DownloadCatalogBuildDifference::MoreDetail),
        ("draft_model", DownloadCatalogBuildDifference::DraftModel),
        (
            "streaming_layout",
            DownloadCatalogBuildDifference::StreamingLayout,
        ),
    ] {
        let parsed_catalog = parse_catalog(vec![
            catalog_entry("Lead", &[]),
            catalog_entry("Build", &build_fields("Lead", wire_name)),
        ])
        .expect("each difference kind should parse");
        let build = parsed_catalog.entries()[1].build().expect("a build");
        assert_eq!(build.difference(), difference);
        assert_eq!(difference.as_str(), wire_name);
    }
}

#[test]
fn should_refuse_a_lead_without_a_difference_and_a_difference_without_a_lead() {
    for half_declared in [
        [("build_of", json!("Lead"))],
        [("build_difference", json!("draft_model"))],
    ] {
        let parse_error = parse_catalog(vec![
            catalog_entry("Lead", &[]),
            catalog_entry("Build", &half_declared),
        ])
        .expect_err("half a build link should be refused");
        assert!(matches!(
            parse_error,
            DownloadCatalogError::InvalidBuild { entry_index: 1 }
        ));
    }
}

#[test]
fn should_refuse_an_empty_lead_name_and_an_unknown_difference() {
    let empty_lead_error = parse_catalog(vec![catalog_entry(
        "Build",
        &build_fields(" ", "draft_model"),
    )])
    .expect_err("an empty lead name should be refused");
    assert!(matches!(
        empty_lead_error,
        DownloadCatalogError::InvalidBuild { entry_index: 0 }
    ));

    let unknown_difference_error = parse_catalog(vec![
        catalog_entry("Lead", &[]),
        catalog_entry("Build", &build_fields("Lead", "faster")),
    ])
    .expect_err("an unknown difference should be refused");
    assert!(matches!(
        unknown_difference_error,
        DownloadCatalogError::Parse(_)
    ));
}

#[test]
fn should_refuse_a_lead_no_entry_has_matching_case_exactly() {
    let parse_error = parse_catalog(vec![
        catalog_entry("Lead", &[]),
        catalog_entry("Build", &build_fields("lead", "more_detail")),
    ])
    .expect_err("a lead must be named exactly");
    assert!(matches!(
        parse_error,
        DownloadCatalogError::UnknownBuildLead { entry_index: 1 }
    ));
}

#[test]
fn should_refuse_a_build_of_itself() {
    let parse_error = parse_catalog(vec![catalog_entry(
        "Build",
        &build_fields("Build", "more_detail"),
    )])
    .expect_err("an entry cannot be a build of itself");
    assert!(matches!(
        parse_error,
        DownloadCatalogError::InvalidBuild { entry_index: 0 }
    ));
}

#[test]
fn should_refuse_a_build_listed_apart_from_its_lead() {
    for listed_apart_flag in ["experimental", "get_started"] {
        let mut build_overrides = build_fields("Lead", "more_detail").to_vec();
        build_overrides.push((listed_apart_flag, json!(true)));
        let parse_error = parse_catalog(vec![
            catalog_entry("Lead", &[]),
            catalog_entry("Build", &build_overrides),
        ])
        .expect_err("a build listed in its own place cannot fold under a lead");
        assert!(
            matches!(
                parse_error,
                DownloadCatalogError::InvalidBuild { entry_index: 1 }
            ),
            "{listed_apart_flag} should be refused, got {parse_error:?}"
        );
    }
}

#[test]
fn should_refuse_a_lead_that_is_a_build_or_listed_elsewhere() {
    let lead_overrides: [Vec<(&str, Value)>; 3] = [
        build_fields("Root", "draft_model").to_vec(),
        vec![("experimental", json!(true))],
        vec![
            ("role", json!("image_generation")),
            ("family", json!("flux2_klein")),
        ],
    ];
    for lead_override in lead_overrides {
        let parse_error = parse_catalog(vec![
            catalog_entry("Root", &[]),
            catalog_entry("Lead", &lead_override),
            catalog_entry("Build", &build_fields("Lead", "more_detail")),
        ])
        .expect_err("the lead must be a plain entry in the same list");
        assert!(
            matches!(
                parse_error,
                DownloadCatalogError::InvalidBuildLead { entry_index: 2 }
            ),
            "{lead_override:?} should be refused, got {parse_error:?}"
        );
    }
}

#[test]
fn should_link_every_bundled_build_to_a_lead_in_the_chat_list() {
    let bundled_catalog =
        DownloadCatalog::load_bundled().expect("the bundled catalog should parse");
    let builds_with_leads: Vec<(&str, &str)> = bundled_catalog
        .entries()
        .iter()
        .filter_map(|catalog_entry| {
            catalog_entry
                .build()
                .map(|build| (catalog_entry.model_id(), build.lead_model_id()))
        })
        .collect();

    assert_eq!(builds_with_leads.len(), 11);
    assert!(builds_with_leads.iter().all(|(_, lead_model_id)| {
        [
            "Ornith-1.5-9B-MLX-4bit-vision",
            "Ornith-1.5-35B-A3B-OptiQ-4bit",
        ]
        .contains(lead_model_id)
    }));
}
