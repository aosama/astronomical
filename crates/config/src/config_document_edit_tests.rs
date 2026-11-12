use serde_json::json;

use super::{
    config_schema_bytes, read_config_document, reset_config_document, save_config_document,
    validate_config_document,
};
use crate::AstronomicalConfigError;

fn valid_document() -> serde_json::Value {
    json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": { "model_directories": [] },
        "chunking": { "experimental_neural_engine_prefill_enabled": true }
    })
}

fn state_directory_with(config_body: &str) -> tempfile::TempDir {
    let state_directory = tempfile::tempdir().expect("a scratch directory");
    std::fs::write(state_directory.path().join("config.json"), config_body)
        .expect("the config is written");
    state_directory
}

fn current_digest(state_directory: &tempfile::TempDir) -> Option<String> {
    read_config_document(state_directory.path())
        .expect("the document reads")
        .document_digest
}

#[test]
fn should_read_a_valid_document_with_its_digest() {
    let state_directory = state_directory_with(&valid_document().to_string());

    let snapshot = read_config_document(state_directory.path()).expect("the document reads");

    assert_eq!(snapshot.document, Some(valid_document()));
    assert_eq!(snapshot.read_error, None);
    assert_eq!(
        snapshot.document_digest.map(|digest| digest.len()),
        Some(64)
    );
}

#[test]
fn should_report_an_absent_file_without_creating_one() {
    let state_directory = tempfile::tempdir().expect("a scratch directory");

    let snapshot = read_config_document(state_directory.path()).expect("the read succeeds");

    assert_eq!(snapshot.document, None);
    assert_eq!(snapshot.document_digest, None);
    assert!(!state_directory.path().join("config.json").exists());
}

#[test]
fn should_report_an_unreadable_document_so_an_editor_can_still_offer_a_reset() {
    let state_directory = state_directory_with("{ not json");

    let snapshot = read_config_document(state_directory.path()).expect("the read succeeds");

    assert_eq!(snapshot.document, None);
    assert!(snapshot.read_error.is_some());
    assert!(snapshot.document_digest.is_some());
}

#[test]
fn should_save_a_valid_document_and_read_it_back() {
    let state_directory = state_directory_with(&valid_document().to_string());
    let expected_digest = current_digest(&state_directory);
    let mut edited_document = valid_document();
    edited_document["runtime"]["maximum_mlx_memory_gb"] = json!(35);

    let saved_digest = save_config_document(
        state_directory.path(),
        expected_digest.as_deref(),
        &edited_document,
    )
    .expect("the document saves");

    let snapshot = read_config_document(state_directory.path()).expect("the document reads");
    assert_eq!(snapshot.document, Some(edited_document));
    assert_eq!(
        snapshot.document_digest.as_deref(),
        Some(saved_digest.as_str())
    );
    assert!(
        state_directory
            .path()
            .join("astronomical-config.schema.json")
            .exists()
    );
}

#[test]
fn should_refuse_to_save_over_an_edit_made_since_the_document_was_read() {
    let state_directory = state_directory_with(&valid_document().to_string());
    let stale_digest = current_digest(&state_directory);
    let hand_edited = valid_document().to_string() + "\n";
    std::fs::write(state_directory.path().join("config.json"), &hand_edited)
        .expect("the hand edit is written");

    let save_result = save_config_document(
        state_directory.path(),
        stale_digest.as_deref(),
        &valid_document(),
    );

    assert!(matches!(
        save_result,
        Err(AstronomicalConfigError::ConfigChangedDuringUpdate)
    ));
    assert_eq!(
        std::fs::read_to_string(state_directory.path().join("config.json"))
            .expect("the file reads"),
        hand_edited
    );
}

#[test]
fn should_refuse_a_document_with_an_unknown_field_and_leave_the_file_alone() {
    let original = valid_document().to_string();
    let state_directory = state_directory_with(&original);
    let expected_digest = current_digest(&state_directory);
    let mut edited_document = valid_document();
    edited_document["runtime"]["no_such_setting"] = json!(true);

    let save_result = save_config_document(
        state_directory.path(),
        expected_digest.as_deref(),
        &edited_document,
    );

    assert!(matches!(
        save_result,
        Err(AstronomicalConfigError::ParseConfigFile { .. })
    ));
    assert_eq!(
        std::fs::read_to_string(state_directory.path().join("config.json"))
            .expect("the file reads"),
        original
    );
}

#[test]
fn should_refuse_a_relative_model_directory() {
    let state_directory = tempfile::tempdir().expect("a scratch directory");
    let mut edited_document = valid_document();
    edited_document["runtime"]["model_directories"] = json!(["relative/path"]);

    let validation_result = validate_config_document(state_directory.path(), &edited_document);

    assert!(matches!(
        validation_result,
        Err(AstronomicalConfigError::PathMustBeAbsolute { .. })
    ));
}

#[test]
fn should_refuse_a_value_at_the_edge_of_its_range() {
    let state_directory = tempfile::tempdir().expect("a scratch directory");
    let mut edited_document = valid_document();
    edited_document["diagnostics"] = json!({ "retained_log_files": 0 });

    assert!(validate_config_document(state_directory.path(), &edited_document).is_err());

    edited_document["diagnostics"] = json!({ "retained_log_files": 1 });

    assert!(validate_config_document(state_directory.path(), &edited_document).is_ok());
}

#[test]
fn should_reset_to_first_run_defaults_and_keep_the_previous_bytes() {
    let original = valid_document().to_string();
    let state_directory = state_directory_with(&original);
    let expected_digest = current_digest(&state_directory);

    reset_config_document(state_directory.path(), expected_digest.as_deref())
        .expect("the document resets");

    let snapshot = read_config_document(state_directory.path()).expect("the document reads");
    let document = snapshot.document.expect("the reset document parses");
    assert_eq!(document["runtime"]["model_directories"], json!([]));
    assert_eq!(
        document["chunking"]["experimental_neural_engine_prefill_enabled"],
        serde_json::Value::Null
    );
    assert_eq!(
        std::fs::read_to_string(state_directory.path().join("config.previous.json"))
            .expect("the previous document was kept"),
        original
    );
}

#[test]
fn should_reset_an_unreadable_document() {
    let state_directory = state_directory_with("{ not json");
    let expected_digest = current_digest(&state_directory);

    reset_config_document(state_directory.path(), expected_digest.as_deref())
        .expect("an unreadable document still resets");

    let snapshot = read_config_document(state_directory.path()).expect("the document reads");
    assert!(snapshot.document.is_some());
    assert_eq!(snapshot.read_error, None);
}

#[test]
fn should_refuse_to_reset_a_document_changed_since_it_was_read() {
    let state_directory = state_directory_with(&valid_document().to_string());
    let stale_digest = current_digest(&state_directory);
    std::fs::write(state_directory.path().join("config.json"), "{}\n")
        .expect("the hand edit is written");

    assert!(matches!(
        reset_config_document(state_directory.path(), stale_digest.as_deref()),
        Err(AstronomicalConfigError::ConfigChangedDuringUpdate)
    ));
    assert!(!state_directory.path().join("config.previous.json").exists());
}

#[test]
fn should_describe_every_setting_including_the_neural_engine_opt_in() {
    let schema: serde_json::Value =
        serde_json::from_slice(config_schema_bytes()).expect("the bundled schema parses");

    assert_eq!(
        schema["$defs"]["chunking"]["properties"]["experimental_neural_engine_prefill_enabled"]["default"],
        json!(false)
    );
}

/// A value the schema permits for one leaf setting, or `None` for a nested object.
fn permitted_example(schema_node: &serde_json::Value) -> Option<serde_json::Value> {
    if let Some(enum_values) = schema_node["enum"].as_array() {
        return enum_values.first().cloned();
    }
    if schema_node["anyOf"].is_array() {
        return Some(serde_json::Value::Null);
    }
    match schema_node["type"].as_str() {
        Some("boolean") => Some(json!(false)),
        Some("integer") | Some("number") => Some(
            schema_node["minimum"]
                .as_i64()
                .map_or(json!(1), serde_json::Value::from),
        ),
        Some("string") => Some(json!("/absolute/fictional/path")),
        Some("array") => Some(json!([])),
        _ => None,
    }
}

fn schema_definition<'schema>(
    schema: &'schema serde_json::Value,
    reference: &serde_json::Value,
) -> &'schema serde_json::Value {
    let definition_name = reference["$ref"]
        .as_str()
        .expect("the section is a reference")
        .trim_start_matches("#/$defs/");
    &schema["$defs"][definition_name]
}

fn every_setting_set(
    schema: &serde_json::Value,
    definition: &serde_json::Value,
) -> serde_json::Map<String, serde_json::Value> {
    let mut configured_settings = serde_json::Map::new();
    let properties = definition["properties"]
        .as_object()
        .expect("the section has properties");
    for (setting_name, setting_node) in properties {
        if setting_node["$ref"].is_string() {
            let nested = schema_definition(schema, setting_node);
            configured_settings.insert(
                setting_name.clone(),
                serde_json::Value::Object(every_setting_set(schema, nested)),
            );
        } else if let Some(example) = permitted_example(setting_node) {
            configured_settings.insert(setting_name.clone(), example);
        }
    }
    configured_settings
}

/// The Settings and Tuning pages offer whatever the schema describes, so a setting the
/// strict parser rejects would be a control that can only ever fail to save.
#[test]
fn should_accept_every_setting_the_schema_describes() {
    let schema: serde_json::Value =
        serde_json::from_slice(config_schema_bytes()).expect("the bundled schema parses");
    let state_directory = tempfile::tempdir().expect("a scratch directory");
    let mut document = valid_document();
    for section in ["runtime", "prompt_cache", "chunking", "diagnostics"] {
        document[section] = serde_json::Value::Object(every_setting_set(
            &schema,
            schema_definition(&schema, &schema["properties"][section]),
        ));
    }
    document["models"] = json!({
        "some-model": serde_json::Value::Object(every_setting_set(&schema, &schema["$defs"]["model"]))
    });

    let validation_result = validate_config_document(state_directory.path(), &document);

    assert!(
        validation_result.is_ok(),
        "{validation_result:?} for {document:#}"
    );
}

/// The console shows the schema's built-in default model and fallback folder names to
/// the person, so they must be the ones the code really falls back to.
#[test]
fn should_describe_the_fallbacks_the_code_really_uses() {
    let schema: serde_json::Value =
        serde_json::from_slice(config_schema_bytes()).expect("the bundled schema parses");
    let runtime_properties = &schema["$defs"]["runtime"]["properties"];

    assert_eq!(
        runtime_properties["default_model"]["x-astronomical-builtin-default"],
        crate::BUILTIN_DEFAULT_MODEL_ID
    );
    assert_eq!(
        runtime_properties["neural_engine_program_directory"]["x-astronomical-instance-folder-default"],
        crate::NEURAL_ENGINE_PROGRAM_DIRECTORY_NAME
    );
    assert_eq!(
        runtime_properties["model_download_directory"]["x-astronomical-instance-folder-default"],
        "models"
    );
}
