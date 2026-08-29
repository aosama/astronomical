use astronomical_config::AstronomicalConfig;

use super::write_config;

#[test]
fn should_disable_completion_attribution_by_default() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "$schema":"./astronomical-config.schema.json",
          "schema_version":1,
          "runtime":{"model_directories":[]}
        }"#,
    );
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(!astronomical_config.completion_attribution_enabled());
}

#[test]
fn should_enable_completion_attribution_when_explicitly_configured() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "$schema":"./astronomical-config.schema.json",
          "schema_version":1,
          "runtime":{"model_directories":[]},
          "diagnostics":{"completion_attribution_enabled":true}
        }"#,
    );
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(astronomical_config.completion_attribution_enabled());
}
