//! Beside-source tests for the suggestion rules and their wording, on a
//! catalog shaped like the shipped one: a small and a large chat model on the
//! ladder, builds of each, and a picture model.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use crate::download_catalog::DownloadCatalog;

use super::{ModelSuggestion, SuggestionInputs, suggest_models};

const GIGABYTE: f64 = 1_073_741_824.0;

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test sizes are small positive gigabyte counts"
)]
fn gigabytes(gigabyte_count: f64) -> u64 {
    (gigabyte_count * GIGABYTE).round() as u64
}

fn measured_speed(essay_rate: u16, code_rate: u16) -> Value {
    json!({
        "measured_on": "Mac with an M5 Pro and 48 GB",
        "astronomical_version": "0.2.129",
        "tokens_per_second": {
            "essay_short_prompt": essay_rate, "essay_long_prompt": essay_rate,
            "code_short_prompt": code_rate, "code_long_prompt": code_rate
        }
    })
}

fn catalog_entry(model_name: &str, entry_fields: &Value) -> Value {
    let mut entry_value = json!({
        "huggingface_id": format!("astronomical-test/{model_name}"),
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "family": "qwen3_5",
        "approximate_size_bytes": 4_000_000_000_u64,
        "role": "chat",
        "public": true,
    });
    for (field_name, field_value) in entry_fields
        .as_object()
        .expect("entry fields are an object")
    {
        if !field_value.is_null() {
            entry_value[field_name] = field_value.clone();
        }
    }
    entry_value
}

/// The ladder catalog with per-model field overrides; a null override removes the field.
fn ladder_catalog(overrides: &Value) -> DownloadCatalog {
    let base_entries = [
        (
            "Small",
            json!({"get_started": true, "approximate_memory_bytes": gigabytes(6.6), "minimum_memory_bytes": gigabytes(1.7)}),
        ),
        (
            "Small-6bit",
            json!({"approximate_memory_bytes": gigabytes(8.7), "build_of": "Small", "build_difference": "more_detail"}),
        ),
        (
            "Large",
            json!({"role": "coding", "get_started": true, "approximate_memory_bytes": gigabytes(23.0), "minimum_memory_bytes": gigabytes(4.1)}),
        ),
        (
            "Large-draft",
            json!({"approximate_memory_bytes": gigabytes(24.0), "build_of": "Large", "build_difference": "draft_model"}),
        ),
        (
            "Large-6bit",
            json!({"approximate_memory_bytes": gigabytes(28.3), "build_of": "Large", "build_difference": "more_detail"}),
        ),
        (
            "Large-8bit",
            json!({"approximate_memory_bytes": gigabytes(36.4), "build_of": "Large", "build_difference": "more_detail"}),
        ),
        (
            "Painter",
            json!({"role": "image_generation", "family": "flux2_klein", "get_started": true, "approximate_memory_bytes": gigabytes(6.2)}),
        ),
    ];
    let entry_values: Vec<Value> = base_entries
        .into_iter()
        .map(|(model_name, mut entry_fields)| {
            if let Some(model_overrides) = overrides.get(model_name).and_then(Value::as_object) {
                for (field_name, field_value) in model_overrides {
                    entry_fields[field_name] = field_value.clone();
                }
            }
            catalog_entry(model_name, &entry_fields)
        })
        .collect();
    DownloadCatalog::parse_json(&json!({"schema_version": 5, "entries": entry_values}).to_string())
        .expect("the ladder catalog should parse")
}

fn suggestions_for(
    catalog: &DownloadCatalog,
    claimed_model_names: &[&str],
    serving_ceiling_bytes: Option<u64>,
) -> Vec<ModelSuggestion> {
    let claimed_huggingface_ids: BTreeSet<String> = claimed_model_names
        .iter()
        .map(|model_name| format!("astronomical-test/{model_name}"))
        .collect();
    suggest_models(&SuggestionInputs {
        entries: catalog.entries(),
        claimed_huggingface_ids: &claimed_huggingface_ids,
        serving_ceiling_bytes,
    })
}

fn tags_and_models(suggestions: &[ModelSuggestion]) -> Vec<(&str, &str)> {
    suggestions
        .iter()
        .map(|suggestion| (suggestion.tag, suggestion.model_id.as_str()))
        .collect()
}

#[test]
fn should_offer_a_stronger_model_its_detailed_build_and_pictures_to_a_large_mac_with_the_small_model()
 {
    let suggestions = suggestions_for(
        &ladder_catalog(&json!({})),
        &["Small"],
        Some(gigabytes(36.5)),
    );

    assert_eq!(
        tags_and_models(&suggestions),
        [
            ("A stronger model", "Large"),
            ("Closer to the original", "Large-6bit"),
            ("Make pictures", "Painter")
        ]
    );
    assert!(
        suggestions[0]
            .reason
            .contains("than Small, and it still runs entirely")
    );
    assert!(
        suggestions[1]
            .reason
            .contains("It needs 5.3 GB more memory.")
    );
    assert_eq!(suggestions[0].huggingface_id, "astronomical-test/Large");
}

#[test]
fn should_offer_a_detailed_build_and_the_large_model_streaming_to_a_small_mac() {
    let suggestions = suggestions_for(
        &ladder_catalog(&json!({})),
        &["Small"],
        Some(gigabytes(11.3)),
    );

    assert_eq!(
        tags_and_models(&suggestions),
        [
            ("Closer to the original", "Small-6bit"),
            ("The strongest that runs here", "Large"),
            ("Make pictures", "Painter")
        ]
    );
    assert!(
        suggestions[1].reason.contains("read from the SSD"),
        "a streaming model is never sold as fast"
    );
}

#[test]
fn should_offer_the_lightest_model_on_a_mac_where_nothing_fits_and_say_how_it_runs() {
    let streaming = suggestions_for(
        &ladder_catalog(&json!({})),
        &["Painter"],
        Some(gigabytes(5.3)),
    );
    let without_floor = suggestions_for(
        &ladder_catalog(&json!({"Small": {"minimum_memory_bytes": null}})),
        &["Painter"],
        Some(gigabytes(5.3)),
    );

    assert_eq!(
        tags_and_models(&streaming),
        [
            ("Best fit for this Mac", "Small"),
            ("The strongest that runs here", "Large")
        ]
    );
    assert!(streaming[0].reason.contains("from the SSD"));
    assert!(without_floor[0].reason.contains("very slow"));
}

#[test]
fn should_not_offer_a_model_whose_build_is_already_here_or_asked_for() {
    let suggestions = suggestions_for(
        &ladder_catalog(&json!({})),
        &["Large-draft", "Painter"],
        Some(gigabytes(36.5)),
    );

    assert_eq!(
        tags_and_models(&suggestions),
        [("Closer to the original", "Large-6bit")]
    );
}

#[test]
fn should_suggest_nothing_while_the_memory_ceiling_is_unknown() {
    assert!(suggestions_for(&ladder_catalog(&json!({})), &[], None).is_empty());
    assert!(suggestions_for(&ladder_catalog(&json!({})), &[], Some(0)).is_empty());
}

#[test]
fn should_offer_more_detail_only_when_it_leaves_a_tenth_of_the_memory_free() {
    let roomy_ceiling_bytes = (gigabytes(28.3) * 10).div_ceil(9);
    let at_the_edge = suggestions_for(
        &ladder_catalog(&json!({})),
        &["Painter"],
        Some(roomy_ceiling_bytes),
    );
    let just_short = suggestions_for(
        &ladder_catalog(&json!({})),
        &["Painter"],
        Some(roomy_ceiling_bytes - gigabytes(1.0)),
    );

    assert_eq!(
        tags_and_models(&at_the_edge)[1],
        ("Closer to the original", "Large-6bit")
    );
    assert_eq!(
        tags_and_models(&just_short),
        [("Best fit for this Mac", "Large")]
    );
}

#[test]
fn should_make_a_draft_build_quicker_at_everything_the_best_fit_itself() {
    let catalog = ladder_catalog(&json!({
        "Large": {"measured_speed": measured_speed(71, 68)},
        "Large-draft": {"measured_speed": measured_speed(81, 97)}
    }));
    let suggestions = suggestions_for(&catalog, &["Painter"], Some(gigabytes(36.5)));

    assert_eq!(
        tags_and_models(&suggestions)[0],
        ("Best fit for this Mac", "Large-draft")
    );
    assert_eq!(
        suggestions
            .iter()
            .filter(|suggestion| suggestion.model_id == "Large-draft")
            .count(),
        1,
        "one card, not two"
    );
    assert!(suggestions[0].reason.ends_with(
        "Its draft model makes replies about 14% quicker when writing and 43% quicker with code."
    ));
}

#[test]
fn should_offer_a_quicker_build_beside_the_one_here_and_disclose_slower_writing() {
    let quicker_at_everything = ladder_catalog(&json!({
        "Large": {"measured_speed": measured_speed(71, 68)},
        "Large-draft": {"measured_speed": measured_speed(81, 97)}
    }));
    let quicker_for_code = ladder_catalog(&json!({
        "Large": {"measured_speed": measured_speed(96, 96)},
        "Large-draft": {"measured_speed": measured_speed(89, 125)}
    }));
    let everything = suggestions_for(
        &quicker_at_everything,
        &["Large", "Painter"],
        Some(gigabytes(36.5)),
    );
    let code_only = suggestions_for(
        &quicker_for_code,
        &["Large", "Painter"],
        Some(gigabytes(36.5)),
    );

    assert_eq!(tags_and_models(&everything)[0], ("Quicker", "Large-draft"));
    assert!(
        everything[0]
            .reason
            .contains("quicker with code than Large, because a draft model is built in.")
    );
    assert_eq!(
        tags_and_models(&code_only)[0],
        ("Quicker for code", "Large-draft")
    );
    assert!(code_only[0].reason.contains("about 30% quicker than Large"));
    assert!(
        code_only[0]
            .reason
            .ends_with("Writing an essay is about 7% slower.")
    );
}

#[test]
fn should_not_offer_a_draft_build_inside_the_spread_without_figures_or_keeping_less_detail() {
    let inside_spread = ladder_catalog(&json!({
        "Large": {"measured_speed": measured_speed(96, 100)},
        "Large-draft": {"measured_speed": measured_speed(96, 109)}
    }));
    let unmeasured_reference =
        ladder_catalog(&json!({"Large-draft": {"measured_speed": measured_speed(89, 125)}}));
    let keeps_less = ladder_catalog(&json!({
        "Large": {"measured_speed": measured_speed(71, 68)},
        "Large-draft": {"measured_speed": measured_speed(95, 111), "approximate_memory_bytes": gigabytes(21.0)}
    }));

    for catalog in [inside_spread, unmeasured_reference, keeps_less] {
        let suggestions = suggestions_for(&catalog, &["Large", "Painter"], Some(gigabytes(36.5)));
        assert!(
            suggestions
                .iter()
                .all(|suggestion| suggestion.model_id != "Large-draft"),
            "{suggestions:?}"
        );
    }
}

#[test]
fn should_never_offer_a_draft_build_for_speed_when_it_would_not_fit_with_room() {
    let catalog = ladder_catalog(&json!({
        "Large": {"measured_speed": measured_speed(96, 96)},
        "Large-draft": {"measured_speed": measured_speed(89, 125), "approximate_memory_bytes": gigabytes(26.0)}
    }));
    let suggestions = suggestions_for(&catalog, &["Large", "Painter"], Some(gigabytes(26.0)));

    assert!(
        suggestions
            .iter()
            .all(|suggestion| suggestion.model_id != "Large-draft"),
        "a streaming draft build drafts nothing yet"
    );
}

#[test]
fn should_say_how_much_slower_a_build_with_more_detail_replies_when_both_are_measured() {
    let catalog = ladder_catalog(&json!({
        "Small": {"measured_speed": measured_speed(50, 50)},
        "Small-6bit": {"measured_speed": measured_speed(36, 36)}
    }));
    let suggestions = suggestions_for(&catalog, &["Small"], Some(gigabytes(11.3)));

    assert!(
        suggestions[0]
            .reason
            .ends_with("It needs 2.1 GB more memory, and replies about 28% slower.")
    );
}

#[test]
fn should_not_offer_more_detail_when_a_build_here_is_at_least_as_large() {
    let catalog =
        ladder_catalog(&json!({"Large-draft": {"approximate_memory_bytes": gigabytes(29.0)}}));
    let suggestions = suggestions_for(&catalog, &["Large-draft", "Painter"], Some(gigabytes(36.5)));

    assert!(
        suggestions.is_empty(),
        "a 29 GB build already keeps the 28.3 GB build's detail: {suggestions:?}"
    );
}

// The parser already refuses an experimental Get started pick, so
// the ladder is exactly the Get started chat and coding picks.
#[test]
fn should_climb_only_the_get_started_picks() {
    let catalog = ladder_catalog(&json!({"Large": {"get_started": null}}));
    let suggestions = suggestions_for(&catalog, &["Painter"], Some(gigabytes(36.5)));

    assert_eq!(
        tags_and_models(&suggestions),
        [
            ("Best fit for this Mac", "Small"),
            ("Closer to the original", "Small-6bit")
        ]
    );
}
