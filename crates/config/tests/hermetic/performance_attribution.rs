use astronomical_config::AstronomicalConfig;

use super::write_config;

#[test]
fn should_disable_performance_attribution_when_not_configured() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{ "model_directories": ["/models/ornith"] }"#,
    );
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(!astronomical_config.performance_attribution_enabled());
}

#[test]
fn should_enable_performance_attribution_when_configured() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "model_directories": ["/models/ornith"],
          "performance_attribution_enabled": true
        }"#,
    );
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(astronomical_config.performance_attribution_enabled());
}

#[test]
fn should_disable_performance_attribution_when_explicitly_configured_as_false() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "model_directories": ["/models/ornith"],
          "performance_attribution_enabled": false
        }"#,
    );
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(!astronomical_config.performance_attribution_enabled());
}

#[test]
fn should_reject_null_performance_attribution_setting() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "model_directories": ["/models/ornith"],
          "performance_attribution_enabled": null
        }"#,
    );

    AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
        .expect_err("null must not be accepted as an omitted performance attribution setting");
}
