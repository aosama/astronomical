use super::*;

#[test]
fn should_leave_default_model_unset_when_omitted() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("missing config should load");

    assert_eq!(astronomical_config.default_model(), None);
}

#[test]
fn should_round_trip_default_model_id() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":["/models/astronomical"],"default_model":"Qwen3.5-2B-4bit"}}"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(astronomical_config.default_model(), Some("Qwen3.5-2B-4bit"));
}

#[test]
fn should_atomically_update_default_model_without_losing_other_config_fields() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let original_config_bytes = br#"{
      "model_directories": ["/models/astronomical"],
      "maximum_mlx_memory_gb": 32,
      "chunking": {}
    }"#;
    write_config(
        temporary_home_directory.path(),
        std::str::from_utf8(original_config_bytes).expect("fixture should be UTF-8"),
    );

    let config_update = write_default_model(
        temporary_home_directory.path().join(".astronomical"),
        Some("Qwen3.5-2B-4bit"),
    )
    .expect("default model should be persisted");

    assert_eq!(
        config_update.prior_config_bytes,
        Some(original_config_bytes.to_vec())
    );
    let persisted_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("the persisted config should load");
    assert_eq!(persisted_config.default_model(), Some("Qwen3.5-2B-4bit"));
    assert_eq!(
        persisted_config
            .maximum_mlx_memory_bytes()
            .expect("the memory ceiling should survive the update"),
        Some(32_000_000_000)
    );
}

#[test]
fn should_clear_default_model() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[],"default_model":"Qwen3.5-2B-4bit"}}"#,
    );

    write_default_model(temporary_home_directory.path().join(".astronomical"), None)
        .expect("clearing the default model should persist");

    let cleared_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("the cleared config should load");
    assert_eq!(cleared_config.default_model(), None);
}

#[test]
fn should_reject_blank_default_model_id() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let update_result = write_default_model(
        temporary_home_directory.path().join(".astronomical"),
        Some("  "),
    );

    assert!(matches!(
        update_result,
        Err(AstronomicalConfigError::InvalidDefaultModel { .. })
    ));
}
